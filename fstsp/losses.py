"""
fstsp.losses -- the RL objective pieces, un-monkeypatched.

advantage() below IS the old three-layer stack flattened into one dispatch:

    fstsp_leader.advantage_leader( fstsp_advfix.advantage_fixed(
        fstsp_train_v4.advantage ))

* rep < 2 or --adv_scale pair  -> fstsp_train_v4.advantage, verbatim
* rep >= 2 and --adv_scale group -> fstsp_advfix.advantage_fixed, verbatim
* --leader_alpha > 1 (or inf)  -> the Leader Reward transform
  (Wang et al., arXiv:2405.13947) applied on top, verbatim

leader_alpha <= 1 is bit-exact base behaviour; adv_scale="group" is inert at
rep=1.  scripts/check_equivalence.py asserts bit-equality against the old
patched chain on synthetic cost tensors.
"""

import math

import torch
import torch.nn.functional as F

from .dp import SplitDPTorchV4
from .rollout import rollout_v4


# ------------------------------------------------------------------- pieces
def masked_bce(logits, target, mask, max_pos_weight=5.0):
    with torch.no_grad():
        pos = (target * mask).sum()
        tot = mask.sum().clamp_min(1.0)
        pw = ((tot - pos) / (pos + 1e-6)).clamp(1.0, max_pos_weight)
    l = -target * F.logsigmoid(logits) * pw - (1 - target) * F.logsigmoid(-logits)
    return (l * mask).sum() / mask.sum().clamp_min(1.0)


def _advantage_pair(C, dp_at, Kk, rep, a):
    """fstsp_train_v4.advantage, verbatim."""
    B, S = C.shape
    N = dp_at.shape[-1]
    G = (C[:, :, None] - dp_at) if a.rtg else C[:, :, None].expand(B, S, N)
    G = G.reshape(B, Kk, rep, N)
    Cg = C.reshape(B, Kk, rep)
    if rep >= 2:
        base = (G.sum(2, keepdim=True) - G) / (rep - 1)
        sd = Cg.std(2, unbiased=False, keepdim=True)
        ref = Cg.mean(2, keepdim=True)
    else:
        base = G.mean(1, keepdim=True)
        sd = Cg.std(1, unbiased=False, keepdim=True)
        ref = Cg.mean(1, keepdim=True)
    adv = G - base
    if a.adv_norm:
        sc = torch.maximum(sd, 0.01 * ref.abs() + 1e-6)
        adv = adv / sc.unsqueeze(-1)
    return adv.reshape(B, S, N)


def _advantage_group(C, dp_at, Kk, rep, a):
    """fstsp_advfix.advantage_fixed for the rep>=2 group scale, verbatim."""
    B, S = C.shape
    N = dp_at.shape[-1]
    G = (C[:, :, None] - dp_at) if a.rtg else C[:, :, None].expand(B, S, N)
    G = G.reshape(B, Kk, rep, N)
    Cg = C.reshape(B, Kk, rep)

    base = (G.sum(2, keepdim=True) - G) / (rep - 1)      # unchanged: within-start
    sd = Cg.std(dim=(1, 2), unbiased=False).view(B, 1, 1)
    ref = Cg.mean(dim=(1, 2)).view(B, 1, 1)

    adv = G - base
    if a.adv_norm:
        sc = torch.maximum(sd, 0.01 * ref.abs() + 1e-6)
        adv = adv / sc.unsqueeze(-1)
    return adv.reshape(B, S, N)


def advantage(C, dp_at, Kk, rep, a):
    """Base advantage (pair/group dispatch) + Leader Reward transform."""
    scale = getattr(a, "adv_scale", "group")
    if rep < 2 or scale == "pair":
        adv = _advantage_pair(C, dp_at, Kk, rep, a)
    else:
        adv = _advantage_group(C, dp_at, Kk, rep, a)

    al = float(getattr(a, "leader_alpha", 0.0) or 0.0)
    if al <= 1.0:                                           # off
        return adv
    B = C.shape[0]
    idx = torch.arange(B, device=C.device)
    lead = C.argmin(dim=1)                                  # [B] lowest cost
    if math.isinf(al):                                      # Alg. 2
        out = torch.zeros_like(adv)
        out[idx, lead] = adv[idx, lead]
        return out
    out = adv / al                                          # Alg. 1, line 7
    out[idx, lead] = adv[idx, lead]                         # lines 8-9
    return out


# --------------------------------------------------------- self-imitation
_MOVE_CACHE = {}


def build_moves(N, kinds):
    key = (N, tuple(kinds))
    if key in _MOVE_CACHE:
        return _MOVE_CACHE[key]
    base = list(range(N))
    seen, P = {tuple(base)}, []
    if "2opt" in kinds:
        for i in range(N - 1):
            for j in range(i + 1, N):
                p = base[:i] + base[i:j + 1][::-1] + base[j + 1:]
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    P.append(p)
    for L, kk in ((1, "oropt1"), (2, "oropt2"), (3, "oropt3")):
        if kk not in kinds:
            continue
        for i in range(N - L + 1):
            seg, rest = base[i:i + L], base[:i] + base[i + L:]
            for j in range(len(rest) + 1):
                p = rest[:j] + seg + rest[j:]
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    P.append(p)
    out = torch.tensor(P, dtype=torch.long)
    _MOVE_CACHE[key] = out
    return out


@torch.no_grad()
def dp_cost(orders, batch_rows, N, dtype=torch.float32):
    """orders [R,N]; batch_rows already expanded to R rows."""
    R = orders.shape[0]
    dev = orders.device
    eng = SplitDPTorchV4(batch_rows["T"].to(dtype), batch_rows["D"].to(dtype),
                         batch_rows["elig"], batch_rows["e"].to(dtype),
                         float(batch_rows["sL"]), float(batch_rows["sR"]),
                         int(batch_rows["depot"]), Lmax=N + 2)
    for t in range(N):
        eng.append(orders[:, t])
    eng.append(torch.full((R,), int(batch_rows["depot"]), dtype=torch.long,
                          device=dev))
    return eng.dp[:, N + 1]


@torch.no_grad()
def ls_improve(orders, batch, N, moves, passes=1):
    """
    Best-improvement local search, all instances in parallel.
    orders [b,N] -> (improved [b,N], improved-cost [b], took [b] bool)
    """
    b = orders.shape[0]
    dev = orders.device
    Mv = moves.shape[0]
    rows = {k: (v.repeat_interleave(Mv, 0)
                if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == b else v)
            for k, v in batch.items()}
    one = {k: (v if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == b else v)
           for k, v in batch.items()}
    cur = orders.clone()
    curc = dp_cost(cur, one, N)
    Pe = moves.to(dev)[None].expand(b, Mv, N)
    took = torch.zeros(b, dtype=torch.bool, device=dev)
    for _ in range(max(1, int(passes))):
        cand = cur[:, None, :].expand(b, Mv, N).gather(2, Pe)
        c = dp_cost(cand.reshape(b * Mv, N), rows, N).reshape(b, Mv)
        best, bi = c.min(1)
        sel = best < curc - 1e-6
        if not bool(sel.any()):
            break
        cur[sel] = cand[sel, bi[sel]]
        curc[sel] = best[sel]
        took |= sel
    return cur, curc, took


def sil_term(model, batch, a, r_best_orders, N, dec=0):
    """
    -log pi(improved order) on the instances local search could improve.
    Returns (loss_term, n_improved, mean_gain).  0.0 when nothing improved.
    """
    B = r_best_orders.shape[0]
    nb = max(1, int(round(float(a.sil_frac) * B)))
    idx = torch.randperm(B, device=r_best_orders.device)[:nb]
    sub = {k: (v[idx] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == B
               else v) for k, v in batch.items()}
    moves = build_moves(N, tuple(a.sil_moves.split("+")))
    o0 = r_best_orders[idx]
    c0 = dp_cost(o0, sub, N)
    o1, c1, took = ls_improve(o0, sub, N, moves, passes=a.sil_passes)
    if not bool(took.any()):
        return 0.0, 0, 0.0
    keep = took.nonzero(as_tuple=True)[0]
    sub2 = {k: (v[keep] if torch.is_tensor(v) and v.ndim > 0
                and v.shape[0] == nb else v) for k, v in sub.items()}
    fr = rollout_v4(model, sub2, force_order=o1[keep], want_labels=False,
                    dec=dec, look_w=a.look_w)
    nll = -fr["logp_steps"].sum(-1).mean()
    gain = float((c0[keep] - c1[keep]).mean())
    return nll, int(keep.numel()), gain
