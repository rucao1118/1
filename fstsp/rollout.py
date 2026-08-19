"""
v4 rollout.

Three things v3 did not have:

  1. candidate_feats_v4 appends the EXACT DP lookahead (3 channels) to v3's
     10.  Same layout otherwise, new channels at the end, which is what makes
     fstsp_model_v4.inflate_v3_state_dict a true no-op warm start.

  2. force_order=.  Teacher forcing: run the decoder over a GIVEN permutation
     and return per-step log-probabilities.  That is the machinery
     self-imitation needs -- take a rollout, improve it with local search,
     then push probability mass onto the improved permutation.  Nothing else
     changes, so the same code path is used for sampling, greedy and forcing
     and there is no second decoder to keep in sync.

  3. beam_decode.  A beam over PREFIXES ranked by the split DP itself.

     Why that is principled and not just "beam search on log-probs": op(i,k)
     is a sum of non-negative times, so dp is non-decreasing in t, so

         dp[t]  <=  cost of EVERY completion of this prefix

     dp[t] is an admissible lower bound, and it is exact, not learned.  A beam
     ranked on it is a truncated best-first search on a real bound with the
     policy supplying the expansion order.  Standard NCO beam search has only
     the log-prob, which bounds nothing.  Set w_lp > 0 to blend the two;
     w_dp=1, w_lp=0 is pure bound-guided search, w_dp=0, w_lp=1 is the usual
     NCO beam.

     The beam re-uses SplitDPTorchV4.append -- the same arithmetic that scores
     the reward and produced the proven-optimal CSVs -- via select_rows for
     the expand/prune bookkeeping.  There is no second cost model anywhere.

ALIASING RULE from v3 still applies and is still load-bearing: engine buffers
are written in place, so anything read out of them that can reach an index
into a grad-carrying tensor is cloned, and the current node is carried in a
standalone `last`.
"""

import torch

from .data import apply_d4_to_batch
from .dp import SplitDPTorchV4
from .model import CAND_DIM, N_NEW_CAND

CAP = 3.0
LOOK_W = 8          # DP lookahead window.  8 is exact at n=20 (measured);
                    # 0 = untruncated; 4 halves the cost (still an upper
                    # bound); NEGATIVE = DISABLED -- channels 10-12 are held
                    # at exactly zero, which is the noLA ablation.


def _typ(batch):
    return float(batch["sL"]) / batch["feat"][:, 0, 5].clamp_min(1e-6)


def candidate_feats_v4(t, eng, Rt, T_e, D_e, ef_e, elig_e, e_e, typ_e,
                       dlog_sig, sL, sR, selectable, look_w=LOOK_W):
    """
    [M, V, CAND_DIM].  Channels 0..9 are v3's, unchanged:
      0..4  edge features (Rt, c)
      5     -T[Rt, c] / typ
      6     -b_fly / typ   (one-step "fly Rt, rendezvous at c" approximation)
      7     1[that sortie is feasible]
      8     eligibility of c
      9     sigmoid(aux drone logit of c)
    and 10..12 are new and exact:
      10    -(dp[t+1](c) - dp[t]) / typ    what appending c ACTUALLY costs
      11    1[the operation closing at c flies somebody]
      12    -(dp[t+1](c) - best legal dp[t+1]) / typ
    """
    M, V, _ = T_e.shape

    ef_row = ef_e.gather(1, Rt.view(M, 1, 1, 1).expand(M, 1, V, ef_e.shape[-1])).squeeze(1)
    T_row = T_e.gather(1, Rt.view(M, 1, 1).expand(M, 1, V)).squeeze(1)
    b_truck = T_row / typ_e[:, None]

    if t >= 1:
        Rp = eng.node[:, t - 1].clone()          # see ALIASING RULE
        Tp_row = T_e.gather(1, Rp.view(M, 1, 1).expand(M, 1, V)).squeeze(1)
        D_row = D_e.gather(1, Rt.view(M, 1, 1).expand(M, 1, V)).squeeze(1)
        d_pt = D_e.reshape(M, V * V).gather(1, (Rp * V + Rt).view(M, 1)).squeeze(1)
        span = torch.maximum(Tp_row, d_pt[:, None] + D_row)
        sL_eff = 0.0 if t == 1 else sL           # no launch service at the depot
        b_fly = (sL_eff + span + sR) / typ_e[:, None]
        elig_Rt = elig_e.gather(1, Rt.view(M, 1))
        feas = (elig_Rt & (span + sR <= e_e[:, None] + 1e-9)).to(T_e.dtype)
    else:
        b_fly = torch.full_like(b_truck, CAP)
        feas = torch.zeros_like(b_truck)

    base = torch.stack([
        *[ef_row[:, :, i] for i in range(ef_e.shape[-1])],
        -b_truck.clamp(max=CAP),
        -b_fly.clamp(max=CAP),
        feas,
        elig_e.to(T_e.dtype),
        dlog_sig,
    ], dim=-1)

    if look_w is not None and int(look_w) < 0:
        # lookahead DISABLED (the noLA ablation).  The channels are exactly
        # zero, so the padded columns of the candidate MLP receive exactly
        # zero gradient and stay at their zero init forever: the arm is the
        # v3 FUNCTION trained under the v4 recipe, which is the control the
        # ablation needs.  Skipping the computation also makes this the
        # honest speed baseline.
        look = torch.zeros(M, V, N_NEW_CAND, dtype=base.dtype,
                           device=base.device)
    else:
        look = eng.lookahead_feats(selectable, window=look_w).to(base.dtype)
    out = torch.cat([base, look], dim=-1)
    assert out.shape[-1] == CAND_DIM, \
        f"candidate_feats_v4 produced {out.shape[-1]}, model expects {CAND_DIM}"
    return out


def _encode_d4(model, batch, modes):
    """One encoder call for all |modes| framings.  [B,nm,V,dim], [B,nm,dim]."""
    B, V, _ = batch["feat"].shape
    nm = len(modes)
    if nm == 1:
        bm = apply_d4_to_batch(batch, modes[0]) if modes[0] else batch
        ne, ge = model.encode(bm["feat"], bm["edge_feat"], batch["elig"])
        return ne[:, None], ge[:, None]

    fs = torch.stack([apply_d4_to_batch(batch, m)["feat"] for m in modes], 1)
    E = batch["edge_feat"].shape[-1]
    ef = batch["edge_feat"][:, None].expand(B, nm, V, V, E).reshape(B * nm, V, V, E)
    el = batch["elig"][:, None].expand(B, nm, V).reshape(B * nm, V)
    ne, ge = model.encode(fs.reshape(B * nm, V, fs.shape[-1]), ef, el)
    return ne.reshape(B, nm, V, model.dim), ge.reshape(B, nm, model.dim)


# ---------------------------------------------------------------------------
def rollout_v4(model, batch, starts=0, rep=1, d4_group=True, greedy=False,
               temperature=1.0, start_mode="pomo", want_labels=True,
               dtype=torch.float32, force_starts=False, dec=0,
               look_w=LOOK_W, force_order=None, force_d4=0):
    """
    force_order  [B, S, N] long, or [B, N] (broadcast over the group).  Every
                 action is taken from it instead of from the policy; S comes
                 from its shape and `starts` is ignored.  logp_steps holds
                 log pi(given action | state), which is the cross-entropy term
                 self-imitation wants.

                 start_mode IS still honoured.  Under "pomo" the first entry
                 is treated as a FORCED START -- appended, but scored with no
                 log-prob -- because that is exactly what the policy was
                 trained under and what it will be evaluated under.  Putting a
                 gradient on p(first customer) here would be training a
                 distribution the model never uses at decode time, which is
                 the free-start arm that already lost at n=10 and n=13.
    """
    feat = batch["feat"]
    B, V, _ = feat.shape
    N = V - 1
    depot = int(batch["depot"])
    assert depot == N, "this pipeline assumes customers 0..N-1 and depot = N"
    dev = feat.device
    sL, sR = float(batch["sL"]), float(batch["sR"])

    forcing = force_order is not None
    if forcing:
        fo = force_order
        if fo.dim() == 2:
            fo = fo[:, None, :]
        assert fo.shape[0] == B and fo.shape[2] == N, "force_order must be [B,(S),N]"
        K, rep = 1, fo.shape[1]
        S = fo.shape[1]
        fo = fo.reshape(B * S, N).long()
    else:
        free = (start_mode == "none")
        K = 1 if (free and not force_starts) \
            else (N if starts in (0, None) else int(starts))
        rep = max(1, int(rep))
        S = K * rep
    M = B * S
    nm = min(rep, 8) if d4_group else 1

    if forcing:
        # ONE framing for the whole forced group.  In forcing mode the group
        # dimension of force_order is a list of permutations, NOT a list of D4
        # replications, so deriving framings from it the way a sampling
        # rollout does would score each order under a rotation of the instance
        # it was not produced in.  force_d4 selects the framing explicitly.
        nm, modes = 1, [int(force_d4)]
    elif not d4_group:
        modes = [0]
    elif greedy:
        modes = list(range(nm))
    else:
        modes = torch.randperm(8)[:nm].tolist()

    ne_all, ge_all = _encode_d4(model, batch, modes)

    midx = torch.zeros(S, dtype=torch.long, device=dev) if forcing else \
        (torch.arange(S, device=dev) % max(rep, 1)) % nm
    node_e = ne_all[:, midx].reshape(M, V, model.dim)
    graph_e = ge_all[:, midx].reshape(M, model.dim)
    cache = model.precompute_decoder(node_e, dec=dec)

    bidx = torch.arange(B, device=dev).repeat_interleave(S)
    T_e = batch["T"].to(dtype)[bidx]
    D_e = batch["D"].to(dtype)[bidx]
    elig_e = batch["elig"][bidx]
    e_e = batch["e"].to(dtype)[bidx]
    typ_e = _typ(batch).to(dtype)[bidx]
    ef_e = batch["edge_feat"].to(dtype)[bidx]

    dlog_sig = torch.sigmoid(model.drone_logits(node_e)).detach()

    eng = SplitDPTorchV4(T_e, D_e, elig_e, e_e, sL, sR, depot, Lmax=N + 2,
                         typ=typ_e)

    ar = torch.arange(M, device=dev)
    selectable = torch.ones(M, V, dtype=torch.bool, device=dev)
    selectable[:, depot] = False

    logp_steps = torch.zeros(M, N, device=dev, dtype=node_e.dtype)
    dp_at = torch.zeros(M, N, device=dev, dtype=dtype)
    ents, seq = [], []

    forced0 = None
    if forcing and start_mode != "none":
        first = fo[:, 0].clone()
        selectable[ar, first] = False
        seq.append(first)
        eng.append(first)
        first_emb = node_e[ar, first]
        last = first
        t0 = 1
    elif forcing:
        first_emb = node_e[:, depot, :]
        last = torch.full((M,), depot, dtype=torch.long, device=dev)
        t0 = 0
    elif start_mode == "none":
        first_emb = node_e[:, depot, :]
        last = torch.full((M,), depot, dtype=torch.long, device=dev)
        t0 = 0
        if force_starts:
            st = torch.arange(K, device=dev) % N
            forced0 = st.repeat_interleave(rep)[None, :].expand(B, S).reshape(M).clone()
    else:
        if start_mode == "pomo":
            st = torch.arange(K, device=dev) % N
        else:
            st = torch.randint(0, N, (K,), device=dev)
        first = st.repeat_interleave(rep)[None, :].expand(B, S).reshape(M).clone()
        selectable[ar, first] = False
        seq.append(first)
        eng.append(first)
        first_emb = node_e[ar, first]
        last = first
        t0 = 1

    for t in range(t0, N):
        dp_at[:, t] = eng.dp[:, t]
        dpf = eng.state_feats().to(node_e.dtype)
        cf = candidate_feats_v4(t, eng, last, T_e, D_e, ef_e, elig_e, e_e,
                                typ_e, dlog_sig, sL, sR, selectable,
                                look_w=look_w).to(node_e.dtype)

        last_emb = node_e[ar, last]
        rc = selectable.sum(1).clamp_min(1).to(node_e.dtype)
        rem = (node_e * selectable[:, :, None].to(node_e.dtype)).sum(1) / rc[:, None]

        lp = model.decode_step(node_e, graph_e, last_emb, first_emb, rem,
                               dpf, cf, selectable, temperature, cache=cache,
                               dec=dec)

        prob = lp.exp()
        ents.append(-(prob * lp.masked_fill(~selectable, 0.0)).sum(1))

        if forcing:
            a = fo[:, t]
        elif greedy:
            a = lp.argmax(-1)
        else:
            with torch.no_grad():
                ps = prob.detach().clamp_min(0.0).masked_fill(~selectable, 0.0)
                bad = ~torch.isfinite(ps).all(1) | (ps.sum(1) <= 0)
                if bool(bad.any()):
                    ps[bad] = selectable[bad].to(ps.dtype)
            a = torch.multinomial(ps, 1).squeeze(1)

        if forced0 is not None and t == 0:
            a = forced0                    # evaluation only, see v3 FORCE_STARTS

        logp_steps[:, t] = lp.gather(1, a[:, None]).squeeze(1)
        selectable[ar, a] = False
        seq.append(a)
        eng.append(a)
        if t0 == 0 and t == 0:
            first_emb = node_e[ar, a]
        last = a

    eng.append(torch.full((M,), depot, dtype=torch.long, device=dev))
    cost = eng.dp[:, N + 1]

    out = {
        "orders": torch.stack(seq, 1).reshape(B, S, N),
        "logp_steps": logp_steps.reshape(B, S, N),
        "entropy": torch.stack(ents, 1).sum(1).reshape(B, S),
        "cost": cost.reshape(B, S),
        "dp_at": dp_at.reshape(B, S, N),
        "node_emb0": ne_all[:, 0],
        "K": K, "rep": rep, "S": S, "dec": dec,
    }
    if want_labels:
        out["labels"] = eng.backtrack_labels(N).reshape(B, S, N)
    return out


# ---------------------------------------------------------------------------
# beam search over prefixes, ranked by the DP lower bound
# ---------------------------------------------------------------------------
@torch.no_grad()
def beam_decode(model, batch, beam=0, k_expand=5, w_dp=1.0, w_lp=0.1,
                d4_mode=0, dec=0, start_mode="pomo", dtype=torch.float32,
                look_w=LOOK_W, return_all=False):
    """
    beam      beams kept per instance.  0 -> N (one seed per POMO start).
    k_expand  candidates expanded per beam per step.
    w_dp,w_lp score = w_dp * dp[t]/typ - w_lp * sum logp.  Lower is better.
              w_lp = 0 is pure lower-bound search; w_dp = 0 is vanilla NCO
              beam search.  The default leans on the bound and lets the policy
              break ties, which is the whole point of having an exact bound.

    Returns cost [B], orders [B,N]; with return_all also cost [B,R] and
    orders [B,R,N] over the whole surviving beam.

    A POMO checkpoint has never been asked for p(first customer) -- step 0 was
    always forced -- so the seeds ARE the N forced starts and the beam only
    begins choosing at t = 1.  A free-start checkpoint seeds one beam at the
    depot and lets the first expansion choose.

    INVARIANT: every instance holds exactly R beams at every step, and row
    b*R + r belongs to instance b.  That is what lets the prune be a topk on a
    [B, R*k] view instead of a segmented sort.
    """
    feat = batch["feat"]
    B, V, _ = feat.shape
    N = V - 1
    depot = int(batch["depot"])
    dev = feat.device
    sL, sR = float(batch["sL"]), float(batch["sR"])
    free = (start_mode == "none")
    Wb = N if beam in (0, None) else int(beam)
    k_expand = max(1, int(k_expand))

    ne_all, ge_all = _encode_d4(model, batch, [int(d4_mode)])
    ne, ge = ne_all[:, 0], ge_all[:, 0]

    R = 1 if free else min(Wb, N)
    M = B * R
    binst = torch.arange(B, device=dev).repeat_interleave(R)

    node_e = ne[binst]
    graph_e = ge[binst]
    T_e = batch["T"].to(dtype)[binst]
    D_e = batch["D"].to(dtype)[binst]
    elig_e = batch["elig"][binst]
    e_e = batch["e"].to(dtype)[binst]
    typ_e = _typ(batch).to(dtype)[binst]
    ef_e = batch["edge_feat"].to(dtype)[binst]
    dlog_sig = torch.sigmoid(model.drone_logits(node_e))

    eng = SplitDPTorchV4(T_e, D_e, elig_e, e_e, sL, sR, depot, Lmax=N + 2,
                         typ=typ_e)
    selectable = torch.ones(M, V, dtype=torch.bool, device=dev)
    selectable[:, depot] = False
    logp_sum = torch.zeros(M, device=dev, dtype=dtype)

    if free:
        last = torch.full((M,), depot, dtype=torch.long, device=dev)
        first_emb = node_e[:, depot, :]
        seq = torch.zeros(M, 0, dtype=torch.long, device=dev)
        t0 = 0
    else:
        first = (torch.arange(R, device=dev) % N).repeat(B)
        selectable[torch.arange(M, device=dev), first] = False
        eng.append(first)
        last = first
        first_emb = node_e[torch.arange(M, device=dev), first]
        seq = first.view(M, 1)
        t0 = 1

    for t in range(t0, N):
        M = seq.shape[0]
        rows_all = torch.arange(M, device=dev)
        dpf = eng.state_feats().to(node_e.dtype)
        cf = candidate_feats_v4(t, eng, last, T_e, D_e, ef_e, elig_e, e_e,
                                typ_e, dlog_sig, sL, sR, selectable,
                                look_w=look_w).to(node_e.dtype)
        rc = selectable.sum(1).clamp_min(1).to(node_e.dtype)
        rem = (node_e * selectable[:, :, None].to(node_e.dtype)).sum(1) / rc[:, None]
        lp = model.decode_step(node_e, graph_e, node_e[rows_all, last],
                               first_emb, rem, dpf, cf, selectable, 1.0,
                               cache=None, dec=dec)

        k = min(k_expand, int(selectable.sum(1).min()))
        top_lp, top_a = lp.topk(k, dim=1)                       # [M,k]

        rows = rows_all.repeat_interleave(k)                    # [M*k]
        acts = top_a.reshape(-1)
        eng = eng.select_rows(rows)
        eng.append(acts)

        lps = logp_sum[rows] + top_lp.reshape(-1).to(dtype)
        score = w_dp * eng.dp[:, t + 1].to(dtype) / typ_e[rows].clamp_min(1e-6) \
            - w_lp * lps

        Rk = R * k
        Wn = min(Wb, Rk)
        sel = score.view(B, Rk).topk(Wn, dim=1, largest=False).indices   # [B,Wn]
        keep = (torch.arange(B, device=dev)[:, None] * Rk + sel).reshape(-1)

        seq = torch.cat([seq[rows][keep], acts[keep].view(-1, 1)], dim=1)
        eng = eng.select_rows(keep)
        selectable = selectable[rows][keep].clone()
        last = acts[keep]
        selectable[torch.arange(keep.shape[0], device=dev), last] = False
        logp_sum = lps[keep]

        node_e = node_e[rows][keep]
        graph_e = graph_e[rows][keep]
        T_e, D_e = T_e[rows][keep], D_e[rows][keep]
        elig_e, e_e = elig_e[rows][keep], e_e[rows][keep]
        typ_e, ef_e = typ_e[rows][keep], ef_e[rows][keep]
        dlog_sig = dlog_sig[rows][keep]
        first_emb = node_e[torch.arange(keep.shape[0], device=dev), last] \
            if (free and t == 0) else first_emb[rows][keep]
        R = Wn

    Mf = seq.shape[0]
    eng.append(torch.full((Mf,), depot, dtype=torch.long, device=dev))
    cost = eng.dp[:, N + 1].view(B, R)
    orders = seq.view(B, R, N)
    bc, bi = cost.min(1)
    bo = orders[torch.arange(B, device=dev), bi]
    if return_all:
        return bc, bo, cost, orders
    return bc, bo


# ---------------------------------------------------------------------------
@torch.no_grad()
def solve_budget_v4(model, batch, budget=160, start_mode="pomo", starts=0,
                    temperature=1.0, dtype=torch.float32, block=8,
                    force_starts=False, n_dec=None, look_w=LOOK_W,
                    beam=0, k_expand=0, w_dp=1.0, w_lp=0.1):
    """
    v3's solve_budget plus an optional beam pass.  k_expand > 0 runs
    beam_decode once per decoder per D4 framing that fits in the budget and
    folds the result into the same running min, so the returned cost is
    "best over everything I was allowed to spend" exactly as before.

    Budget accounting for the beam: one beam pass with `beam` beams and
    `k_expand` expansions costs beam*k_expand DP-scored partial extensions per
    step, i.e. roughly beam*k_expand constructions' worth of DP work.  Report
    it as its own row rather than pretending it is free.
    """
    B, V, _ = batch["T"].shape
    N = V - 1
    dev = batch["T"].device
    D = getattr(model, "n_dec", 1) if n_dec is None else int(n_dec)
    best_c = torch.full((B,), float("inf"), device=dev, dtype=dtype)
    best_o = torch.zeros(B, N, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)

    def take_co(c, o):
        nonlocal best_c, best_o
        sel = c.to(best_c.dtype) < best_c
        best_c = torch.where(sel, c.to(best_c.dtype), best_c)
        best_o = torch.where(sel[:, None], o, best_o)

    def take(r):
        c, i = r["cost"].min(1)
        take_co(c, r["orders"][ar, i])

    def run(K, rep, greedy, d):
        take(rollout_v4(model, batch, starts=(0 if K == N else K), rep=rep,
                        d4_group=True, greedy=greedy, temperature=temperature,
                        start_mode=start_mode, want_labels=False, dtype=dtype,
                        force_starts=force_starts, dec=d, look_w=look_w))

    def run_blocks(K, total, greedy, d):
        done = 0
        while done < total:
            r = min(block, total - done)
            run(K, r, greedy, d)
            done += r

    for d in range(D):
        if start_mode == "none" and not force_starts:
            g = min(8, budget)
            run(1, g, True, d)
            if budget - g > 0:
                run_blocks(1, budget - g, False, d)
        else:
            K = N if starts in (0, None) else int(starts)
            rg = max(1, min(8, budget // K))
            run(K, rg, True, d)
            left = budget - K * rg
            if left >= K:
                run_blocks(K, left // K, False, d)

        if k_expand and k_expand > 0:
            c, o = beam_decode(model, batch, beam=beam, k_expand=k_expand,
                               w_dp=w_dp, w_lp=w_lp, d4_mode=0, dec=d,
                               start_mode=start_mode, dtype=dtype,
                               look_w=look_w)
            take_co(c, o)

    return best_c, best_o