"""
fstsp.data -- ONE data module.

Merge of fstsp_data_v3.py + fstsp_data_v4.py + fstsp_data_depot.py, bodies
verbatim.  The single entry point is

    generate_batch(B, n_customer=.., depot_mode="center|centroid|edge|corner|
                   murray3|mix4", elig_mode="strat|rate_cap|rate|count", ...)

which is generate_batch_depot under its old name; depot_mode="center" is
bit-exact generate_batch_v4 (assert_matches_v4 proves it at import-check
time), so the old names are kept as thin aliases and every RNG stream is
unchanged.  v_drone_set / endurance_set allow OOD evaluation sets (e.g.
v=45 or e=60) without touching the module constants.
"""

import argparse
import hashlib
import json
import math
import os

import torch


"""
v3 data.  Self-contained.

Set M, strict Murray & Chu 2015:
    8 x 8 mile square, depot at the centre
    truck 25 mph Manhattan, drone in {15,25,35} mph Euclidean
    endurance in {20,40} minutes, sL = sR = 1
    customers 0..N-1, depot = N

DRONE ELIGIBILITY
    "rate"  (default)  Binomial(N, INELIG_RATE), floored at MIN_INELIG.
    "count"            exactly 1 or 2, uniform, whatever N is.

    "count" was fitted to Murray's 10-customer instances, which carry 1 or 2
    ineligible customers.  But eligibility there comes from parcel weight
    against the drone's payload -- a per-customer property -- so it is a RATE,
    and 1-2 out of 10 is just what a rate near 0.15 looks like in a small
    sample.  Counting the flag column over the 36 Set M directories gives
    12 x 1 + 24 x 2 = 60 / 36 = 1.67, i.e. 16.7%.

    Keeping "count" at n=20 would quietly dilute the constraint to 5-10% and
    at n=50 to 2-4%, so a model trained that way arrives at the real benchmark
    with no preparation.  At n=10 the two modes are nearly the same
    distribution (rate: mean ~1.6, count: 1.5), which is why the existing
    frozen n=10 set stays a legitimate hold-out.

    CHANGING THE MODE CHANGES THE RNG STREAM.  Do not regenerate an existing
    frozen test set unless you also re-run the exact solver on it.

Node features (12):
    0,1  x, y normalised to the unit square
    2    eligibility
    3    e / typ          endurance in units of the typical truck time
    4    rho              drone speed / truck speed
    5,6  sL / typ, sR / typ
    7    is-depot
    8    detour_j         mean truck detour saved by flying j   (gated by elig)
    9    reach_j          fraction of i with a feasible out-and-back (gated)
    10   nnT_j            nearest truck neighbour
    11   nnD_j            nearest drone neighbour

Edge features (5):
    0    T / typ
    1    D / e
    2    (T - D) / typ
    3    1[D <= e]
    4    op2[i,k]  min over j of {sL + max(T[i,k], D[i,j]+D[j,k]) + sR} - T[i,k]
                   the marginal cost of covering ONE extra customer by drone
                   while the truck drives the edge (i,k).  The DP's own
                   quantity, handed straight to the attention bias.

Fully vectorised: no python loop over the batch, no .item(), so on CUDA it
issues no host syncs.

add_v3_features() upgrades a batch saved with the older 8/4 dimensional
features.  That is a file-format shim, not a code dependency.
"""

import torch

FEAT_DIM = 12
EDGE_DIM = 5
FEAT_DIM_BASE = 8
EDGE_DIM_BASE = 4

OP2_CAP = 3.0

SIDE = 8.0
V_TRUCK = 25.0
V_DRONE = (15.0, 25.0, 35.0)
ENDURANCE = (20.0, 40.0)
S_LAUNCH = 1.0
S_RECOVER = 1.0

# ---- eligibility ----------------------------------------------------------
ELIG_MODE = "rate"          # "rate" | "count"
INELIG_RATE = 0.15          # calibrated on Set M: 60 ineligible / 360 customers
MIN_INELIG = 1              # keep the constraint active in every instance
# ---------------------------------------------------------------------------


def _manhattan(x):
    return (x[:, :, None, :] - x[:, None, :, :]).abs().sum(-1)


def _euclidean(x):
    return torch.sqrt(((x[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1) + 1e-12)


def _typ_from_feat(feat, sL):
    """feat[:,:,5] == sL / typical_truck_time, so typ = sL / that."""
    return float(sL) / feat[:, 0, 5].clamp_min(1e-6)


def _balanced(values, B, device, g):
    """Each value used floor(B/k) or ceil(B/k) times, in random order."""
    v = torch.tensor(values, device=device, dtype=torch.float32)
    idx = torch.arange(B, device=device) % len(values)
    return v[idx][torch.randperm(B, device=device, generator=g)]


def draw_elig(B, N, device, g, mode=None, rate=None, min_inelig=None):
    """
    [B, N+1] bool, True = the drone may serve this node.  Depot is False.
    No host sync: the count is turned into a rank threshold.
    """
    mode = ELIG_MODE if mode is None else mode
    rate = INELIG_RATE if rate is None else float(rate)
    mn = MIN_INELIG if min_inelig is None else int(min_inelig)

    rank = torch.rand(B, N, device=device, generator=g).argsort(1).argsort(1)
    if mode == "count":
        nbad = torch.randint(1, 3, (B, 1), device=device, generator=g)
    elif mode == "rate":
        u = torch.rand(B, N, device=device, generator=g)
        nbad = (u < rate).sum(1, keepdim=True).clamp(min=mn, max=N - 1)
    else:
        raise ValueError(f"elig_mode must be 'rate' or 'count', got {mode!r}")

    elig = torch.ones(B, N + 1, dtype=torch.bool, device=device)
    elig[:, :N] = rank >= nbad
    elig[:, N] = False
    return elig


# ---------------------------------------------------------------------------
@torch.no_grad()
def add_v3_features(batch, chunk=64):
    """Append node features 8..11 and edge feature 4.  Idempotent."""
    if batch["feat"].shape[-1] >= FEAT_DIM:
        return batch

    T, D = batch["T"], batch["D"]
    elig, e = batch["elig"], batch["e"]
    feat, ef = batch["feat"], batch["edge_feat"]
    sL, sR = float(batch["sL"]), float(batch["sR"])
    depot = int(batch["depot"])

    B, V, _ = T.shape
    assert depot == V - 1, "customers 0..N-1, depot = N"
    dev, dt = T.device, T.dtype
    chunk = max(1, min(chunk, int(2e7 // max(V ** 3, 1))))     # cap [m,V,V,V]
    typ = _typ_from_feat(feat, sL).to(dt)
    eye = torch.eye(V, device=dev, dtype=torch.bool)[None]

    Tm = T.masked_fill(eye, 0.0)
    denom = float(V - 1)
    mean_in = Tm.sum(1) / denom
    mean_out = Tm.sum(2) / denom
    grand = Tm.sum((1, 2)) / (denom * V)
    detour = (mean_in + mean_out - grand[:, None]) / typ[:, None]

    reach = ((2.0 * D + sR) <= e[:, None, None]).to(dt).mean(1)

    big = torch.finfo(dt).max
    nnT = T.masked_fill(eye, big).min(1).values / typ[:, None]
    nnD = D.masked_fill(eye, big).min(1).values / typ[:, None]

    keep = elig.to(dt)
    keep[:, depot] = 0.0
    node_extra = torch.stack([detour * keep, reach * keep, nnT, nnD], dim=-1)

    op2 = torch.empty(B, V, V, device=dev, dtype=dt)
    for s in range(0, B, chunk):
        sl = slice(s, min(s + chunk, B))
        Ds, Ts, Es, es = D[sl], T[sl], elig[sl], e[sl]
        m = Ds.shape[0]
        # Ds.unsqueeze(3) -> [m,V,V,1] = D[i,j];  Ds.unsqueeze(1) -> D[j,k]
        fly = Ds.unsqueeze(3) + Ds.unsqueeze(1)
        tik = Ts[:, :, None, :]
        span = torch.maximum(tik, fly)
        ok = Es[:, None, :, None].expand(m, V, V, V) & \
            (span + sR <= es.view(m, 1, 1, 1) + 1e-9)
        jr = torch.arange(V, device=dev)
        ok = ok & (jr.view(1, V, 1, 1) != jr.view(1, 1, V, 1))      # j != i
        ok = ok & (jr.view(1, 1, V, 1) != jr.view(1, 1, 1, V))      # j != k
        cand = (sL + span + sR).masked_fill(~ok, float(OP2_CAP) * 1e6)
        best = cand.min(dim=2).values
        tv = typ[sl].view(m, 1, 1)
        op2[sl] = torch.minimum((best - Ts).clamp_min(0.0), OP2_CAP * tv) / tv

    out = dict(batch)
    out["feat"] = torch.cat([feat, node_extra], dim=-1)
    out["edge_feat"] = torch.cat([ef, op2.unsqueeze(-1)], dim=-1)
    return out


@torch.no_grad()
def generate_batch_v3(batch_size, n_customer=10, profile="setM", device="cpu",
                      seed=None, balanced=True, elig_mode=None,
                      inelig_rate=None, min_inelig=None):
    if profile.lower() not in ("setm", "m", "murray"):
        raise ValueError(f"only setM is implemented, got {profile}")

    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    B, N, depot = int(batch_size), int(n_customer), int(n_customer)
    sL, sR = S_LAUNCH, S_RECOVER

    coords = torch.cat([
        torch.rand(B, N, 2, device=device, generator=g) * SIDE,
        torch.full((B, 1, 2), SIDE / 2.0, device=device)], dim=1)

    if balanced:
        v_dr = _balanced(V_DRONE, B, device, g)
        e = _balanced(ENDURANCE, B, device, g)
    else:
        v_dr = torch.tensor(V_DRONE, device=device)[
            torch.randint(0, len(V_DRONE), (B,), device=device, generator=g)]
        e = torch.tensor(ENDURANCE, device=device)[
            torch.randint(0, len(ENDURANCE), (B,), device=device, generator=g)]

    T = _manhattan(coords) / V_TRUCK * 60.0
    D = _euclidean(coords) / v_dr[:, None, None] * 60.0

    elig = draw_elig(B, N, device, g, elig_mode, inelig_rate, min_inelig)

    typ = SIDE / V_TRUCK * 60.0
    feat = torch.zeros(B, N + 1, FEAT_DIM_BASE, device=device)
    feat[:, :, 0:2] = coords / SIDE
    feat[:, :, 2] = elig.float()
    feat[:, :, 3] = (e / typ)[:, None]
    feat[:, :, 4] = (v_dr / V_TRUCK)[:, None]
    feat[:, :, 5] = sL / typ
    feat[:, :, 6] = sR / typ
    feat[:, depot, 7] = 1.0

    ef = torch.zeros(B, N + 1, N + 1, EDGE_DIM_BASE, device=device)
    ef[:, :, :, 0] = T / typ
    ef[:, :, :, 1] = D / e[:, None, None].clamp_min(1e-6)
    ef[:, :, :, 2] = (T - D) / typ
    ef[:, :, :, 3] = (D <= e[:, None, None]).float()

    return add_v3_features({
        "coords": coords, "feat": feat, "edge_feat": ef, "T": T, "D": D,
        "elig": elig, "e": e, "sL": sL, "sR": sR, "depot": depot,
        "profile": "setM",
    })


# ---------------------------------------------------------------------------
def d4_transform_feat(feat, mode):
    """
    Move the normalised coordinate channels only.  T, D and every derived
    feature are functions of L1/L2 distances, which the dihedral group
    preserves exactly, so the transformed batch is the SAME instance with the
    same optimum.
    """
    out = feat.clone()
    x, y = feat[:, :, 0], feat[:, :, 1]
    xx, yy = {
        0: (x, y),
        1: (1.0 - y, x),
        2: (1.0 - x, 1.0 - y),
        3: (y, 1.0 - x),
        4: (1.0 - x, y),
        5: (x, 1.0 - y),
        6: (y, x),
        7: (1.0 - y, 1.0 - x),
    }[int(mode)]
    out[:, :, 0], out[:, :, 1] = xx, yy
    return out


def apply_d4_to_batch(batch, mode):
    out = dict(batch)
    out["feat"] = d4_transform_feat(batch["feat"], mode)
    return out


# ==========================================================================
# v4 eligibility layer (was fstsp_data_v4.py)
# ==========================================================================
"""
v4 data.  Identical to v3 in every respect EXCEPT how the drone-ineligible set
is drawn.  Everything else -- coordinates, speeds, endurance, service times,
the 12 node features, the 5 edge features, D4 -- is re-exported from
fstsp_data_v3 unchanged, so a v3 checkpoint and a v4 checkpoint see the same
feature semantics and remain comparable.

--------------------------------------------------------------------------
THE ELIGIBILITY PROBLEM THIS FILE EXISTS TO FIX
--------------------------------------------------------------------------
Murray & Chu's Set M has 10 customers and 1 or 2 ineligible ones, i.e. 60
ineligible out of 360 = 16.7%.  Eligibility there is a per-customer property
(parcel weight against the drone's payload), so a rate is the right model and
v3's Binomial(N, 0.15) is the right *generative* extrapolation to N = 20.

The trouble is entirely in the tail, and it is a reporting problem, not a
modelling error:

    Binomial(20, 0.15):  mean 3.0, sd 1.60
    P(X >= 5) = 0.1702       P(X >= 6) = 0.0673      P(X >= 7) = 0.0219
    max observed 9/20 = 45% ineligible

Murray never publishes an instance above 2/10 = 20%.  So ~17% of the v4-sized
test set sits outside anything the benchmark demonstrates, and a referee is
entitled to ask whether the headline number is being propped up (or dragged
down) by instances the benchmark never sanctioned.  Neither answer is one you
want to have to argue.

The fix is NOT to hide the tail.  It is to separate the two jobs:

  TRAINING   should cover a BAND of rates, so the policy is robust and no
             single rate is the one it was tuned to.  mode "strat": each
             instance draws its own rate from INELIG_GRID (balanced), then
             Binomial, then clamps into [min_inelig, max_inelig].
             Coverage 5% - 25%, mean rate 15%, i.e. still centred on Set M.

  HEADLINE   test set should sit INSIDE the band the benchmark demonstrates,
             so the main table is defensible without a footnote.
             mode "rate_cap": Binomial(N, 0.15) truncated at
             max_inelig = ceil(MAX_RATE * N) = 5 at N = 20 (25%), which
             strictly contains Murray's observed 10%-20% and drops the 6.7%
             of draws that exceed it.

  ROBUSTNESS table keeps the uncapped v3 stream ("rate") and is reported
             STRATIFIED BY n_inelig.  fstsp_eval_v3.py already prints exactly
             that row (`strat(gtop, df, "n_inelig", "inel")`), so this costs
             nothing but a second data file.

What goes in the paper is then one honest sentence: eligibility is drawn
per-customer at Murray's measured rate; the headline set truncates at 25%
ineligible, the upper end of what Set M demonstrates; performance on the
untruncated tail is reported separately and does not degrade / degrades by X.

--------------------------------------------------------------------------
!!! THE RNG STREAM MOVES !!!
--------------------------------------------------------------------------
draw_elig_v4 consumes different random numbers from draw_elig.  Any .pt built
with a v4 mode is a DIFFERENT test set from one built with v3, even at the
same seed.  Its frozen reference (fstsp_ref_v3.py) must be regenerated, and
data/test_n10_v3.pt + data/test_n13_v3.pt with their PROVEN .exact.csv must
NOT be rebuilt with this module.  meta records the mode so an accident is
detectable after the fact.
"""

import math



# ---- v4 eligibility -------------------------------------------------------
ELIG_MODE_V4 = "strat"           # "strat" | "rate_cap" | "rate" | "count"
INELIG_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)   # mean 0.15, i.e. Set M centred
MAX_RATE = 0.25                  # cap: ceil(MAX_RATE * N); 5 at N=20
# ---------------------------------------------------------------------------


def cap_from_rate(N, max_rate=MAX_RATE):
    """Largest allowed ineligible count.  ceil, so N=10 -> 3 and N=20 -> 5."""
    return max(1, min(int(N) - 1, int(math.ceil(float(max_rate) * int(N)))))


@torch.no_grad()
def draw_elig_v4(B, N, device, g, mode=None, rate=None, min_inelig=None,
                 max_inelig=None, grid=None):
    """
    [B, N+1] bool, True = the drone may serve this node.  Depot is False.
    Fully vectorised, no host sync: counts become rank thresholds.

    mode
      "strat"     per-instance rate from `grid` (balanced), then Binomial,
                  then clamped into [min_inelig, max_inelig].  TRAINING.
      "rate_cap"  Binomial(N, rate) clamped into [min_inelig, max_inelig].
                  HEADLINE TEST SET.
      "rate"      v3 behaviour, no upper clamp.  ROBUSTNESS TEST SET.
      "count"     legacy 1-or-2, kept so the frozen n=10 / n=13 sets stay
                  reproducible.  Do not use above N = 10.
    """
    mode = ELIG_MODE_V4 if mode is None else mode
    rate = INELIG_RATE if rate is None else float(rate)
    mn = MIN_INELIG if min_inelig is None else int(min_inelig)
    grid = INELIG_GRID if grid is None else tuple(grid)

    if mode in ("rate", "count"):
        return draw_elig(B, N, device, g, mode=mode, rate=rate, min_inelig=mn)

    mx = cap_from_rate(N) if max_inelig is None else int(max_inelig)
    mx = max(mn, min(int(N) - 1, mx))

    rank = torch.rand(B, N, device=device, generator=g).argsort(1).argsort(1)

    if mode == "rate_cap":
        p = torch.full((B, 1), float(rate), device=device)
    elif mode == "strat":
        p = _balanced(grid, B, device, g).to(torch.float32).view(B, 1)
    else:
        raise ValueError(f"unknown elig mode {mode!r}")

    nbad = _trunc_binom(p, N, mn, mx, device, g)
    elig = torch.ones(B, N + 1, dtype=torch.bool, device=device)
    elig[:, :N] = rank >= nbad
    elig[:, N] = False
    return elig


def _trunc_binom(p, N, lo, hi, device, g):
    """
    [B,1] long ~ Binomial(N, p) CONDITIONED on lo <= X <= hi, by inverse CDF.

    Conditioning, not clamping.  Clamping to `hi` would pile P(X > hi) onto
    the single value hi -- at N=20, p=0.15, hi=5 that is a 6.7% point mass on
    exactly 5, which is both wrong and the kind of artefact a referee spots in
    a histogram.  Conditioning redistributes it over the whole legal range and
    is a statement you can write down: Binomial(N, p) | lo <= X <= hi.
    """
    B = p.shape[0]
    k = torch.arange(N + 1, device=device, dtype=torch.float32)          # [N+1]
    lg = torch.lgamma
    logc = lg(torch.tensor(N + 1.0, device=device)) - lg(k + 1.0) - lg(N - k + 1.0)
    pc = p.clamp(1e-6, 1 - 1e-6).to(torch.float32)
    logpmf = logc.view(1, -1) + k.view(1, -1) * pc.log() \
        + (N - k).view(1, -1) * (1.0 - pc).log()                         # [B,N+1]
    legal = (k >= lo) & (k <= hi)
    logpmf = logpmf.masked_fill(~legal.view(1, -1), float("-inf"))
    cdf = torch.softmax(logpmf, dim=1).cumsum(1)
    u = torch.rand(B, 1, device=device, generator=g)
    return (cdf < u).sum(1, keepdim=True).clamp(lo, hi).long()


@torch.no_grad()
def generate_batch_v4(batch_size, n_customer=10, profile="setM", device="cpu",
                      seed=None, balanced=True, elig_mode=None,
                      inelig_rate=None, min_inelig=None, max_inelig=None,
                      inelig_grid=None):
    """
    Byte-for-byte generate_batch_v3 except for the draw_elig call.  Kept as a
    copy rather than a wrapper because the eligibility draw sits in the MIDDLE
    of the RNG stream: calling v3 and patching afterwards would consume the
    wrong numbers and silently change the coordinates too.
    """
    if profile.lower() not in ("setm", "m", "murray"):
        raise ValueError(f"only setM is implemented, got {profile}")

    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    B, N, depot = int(batch_size), int(n_customer), int(n_customer)
    sL, sR = S_LAUNCH, S_RECOVER

    coords = torch.cat([
        torch.rand(B, N, 2, device=device, generator=g) * SIDE,
        torch.full((B, 1, 2), SIDE / 2.0, device=device)], dim=1)

    if balanced:
        v_dr = _balanced(V_DRONE, B, device, g)
        e = _balanced(ENDURANCE, B, device, g)
    else:
        v_dr = torch.tensor(V_DRONE, device=device)[
            torch.randint(0, len(V_DRONE), (B,), device=device, generator=g)]
        e = torch.tensor(ENDURANCE, device=device)[
            torch.randint(0, len(ENDURANCE), (B,), device=device, generator=g)]

    T = _manhattan(coords) / V_TRUCK * 60.0
    D = _euclidean(coords) / v_dr[:, None, None] * 60.0

    elig = draw_elig_v4(B, N, device, g, elig_mode, inelig_rate, min_inelig,
                        max_inelig, inelig_grid)

    typ = SIDE / V_TRUCK * 60.0
    feat = torch.zeros(B, N + 1, FEAT_DIM_BASE, device=device)
    feat[:, :, 0:2] = coords / SIDE
    feat[:, :, 2] = elig.float()
    feat[:, :, 3] = (e / typ)[:, None]
    feat[:, :, 4] = (v_dr / V_TRUCK)[:, None]
    feat[:, :, 5] = sL / typ
    feat[:, :, 6] = sR / typ
    feat[:, depot, 7] = 1.0

    ef = torch.zeros(B, N + 1, N + 1, EDGE_DIM_BASE, device=device)
    ef[:, :, :, 0] = T / typ
    ef[:, :, :, 1] = D / e[:, None, None].clamp_min(1e-6)
    ef[:, :, :, 2] = (T - D) / typ
    ef[:, :, :, 3] = (D <= e[:, None, None]).float()

    return add_v3_features({
        "coords": coords, "feat": feat, "edge_feat": ef, "T": T, "D": D,
        "elig": elig, "e": e, "sL": sL, "sR": sR, "depot": depot,
        "profile": "setM",
    })


# ---------------------------------------------------------------------------
def elig_report(N, mode=None, num=100000, seed=0, device="cpu", **kw):
    """
    The numbers to quote in the paper.  Returns a dict; prints a histogram.

        python -c "import fstsp_data_v4 as d; d.elig_report(20,'strat')"
    """
    mode = ELIG_MODE_V4 if mode is None else mode
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    el = draw_elig_v4(int(num), int(N), device, g, mode=mode, **kw)
    ine = (~el[:, :N]).sum(1)
    hist = torch.bincount(ine, minlength=N + 1)
    out = {
        "mode": mode, "n": int(N), "num": int(num),
        "mean_count": float(ine.float().mean()),
        "mean_rate_pct": float(ine.float().mean()) / N * 100.0,
        "min": int(ine.min()), "max": int(ine.max()),
        "p_over_20pct": float((ine.float() / N > 0.2001).float().mean()),
        "hist": {i: int(c) for i, c in enumerate(hist) if c},
    }
    print(f"mode={mode:9s} n={N}  mean {out['mean_count']:.2f} "
          f"({out['mean_rate_pct']:.1f}%)  range [{out['min']},{out['max']}]  "
          f"P(>20% inelig)={out['p_over_20pct']*100:.1f}%")
    print("  " + "  ".join(f"{i}:{c/num*100:.1f}%" for i, c in out["hist"].items()))
    return out




# ==========================================================================
# depot layer (was fstsp_data_depot.py)
# ==========================================================================
"""
Depot-parametric data on top of fstsp_data_v4.  Everything else -- coords,
speeds, endurance, eligibility, the 12 node features, 5 edge features, D4 --
is unchanged, so checkpoints trained here remain feature-compatible with v3/v4.

WHY THIS FILE EXISTS
    Murray & Chu 5.1: "the depot location was randomly chosen to be either the
    average of the x- and y-coordinates of the customers (near the center of
    gravity), the average of the customers' x-coordinates with a y-coordinate
    of zero, or at the southwest corner of the region (origin)."
    generate_batch_v4 hard-codes depot = (SIDE/2, SIDE/2), which is mode (a)
    only.  To claim Set M zero-shot transfer (2/3 of Set M is edge/corner
    depot) the training distribution must cover all three.

DEPOT MODES
    "center"    (SIDE/2, SIDE/2)          -- bit-exact generate_batch_v4
    "centroid"  mean of customer coords   -- Murray (a)
    "edge"      (mean customer x, 0)      -- Murray (b)
    "corner"    (0, 0)                    -- Murray (c)
    "murray3"   per-instance balanced over {centroid, edge, corner}
                                          -- the faithful Murray protocol
    "mix4"      balanced over all four    -- keeps the current center anchor
                                             in-distribution as well

RNG DISCIPLINE (same contract as fstsp_data_tau)
    The v4 draw order [coords, v_drone, e, elig] is reproduced exactly, and
    the depot placement is either deterministic (single modes: NO extra RNG)
    or drawn strictly AFTER the v4 stream (mixed modes: one randperm).
    Consequences, both verified by the self-checks below:
      * depot_mode="center" reproduces generate_batch_v4 bit for bit
        (assert_matches_v4);
      * under one seed, every mode shares identical customer coords, speeds,
        endurance and eligibility -- mode comparisons are controlled
        experiments on the SAME instances (assert_controlled).

The dihedral-D4 augmentation stays valid: all four depot positions are mapped
within the unit square by the group, and T/D are isometry-invariant, so a
transformed batch is the same instance with the same optimum.

Usage
    python fstsp_data_depot.py                         # self-checks + report
    python fstsp_data_depot.py --make_testset --n 50 --num 128 \
        --seed 50711 --depot_mode murray3 --out data/test_n50_depot3.pt
"""

import argparse
import hashlib
import json
import math
import os



SINGLE_MODES = ("center", "centroid", "edge", "corner")
MIXED_MODES = ("murray3", "mix4")
ALL_MODES = SINGLE_MODES + MIXED_MODES


def _depot_xy(cust, midx):
    """[B,N,2] customer coords + [B] mode index (0..3) -> [B,2] depot coord."""
    B = cust.shape[0]
    mean = cust.mean(dim=1)                                        # [B,2]
    xy = torch.full((B, 2), SIDE / 2.0, dtype=cust.dtype,
                    device=cust.device)                            # 0 center
    m = midx == 1                                                  # centroid
    xy[m] = mean[m]
    m = midx == 2                                                  # edge
    xy[m, 0] = mean[m, 0]
    xy[m, 1] = 0.0
    m = midx == 3                                                  # corner
    xy[m] = 0.0
    return xy


@torch.no_grad()
def generate_batch_depot(batch_size, n_customer=10, profile="setM",
                         device="cpu", seed=None, balanced=True,
                         depot_mode="center",
                         elig_mode=None, inelig_rate=None, min_inelig=None,
                         max_inelig=None, inelig_grid=None,
                         v_drone_set=None, endurance_set=None):
    """generate_batch_v4 with the depot placement made parametric.

    Copied body, not a wrapper, for the same reason fstsp_data_v4 copies v3:
    the depot coordinate enters T, D and every feature derived from them, so
    it must be set BEFORE the feature build -- and the RNG draws must stay in
    the v4 order or the stream (and the "center" bit-exactness) breaks.
    """
    if profile.lower() not in ("setm", "m", "murray"):
        raise ValueError(f"only setM is implemented, got {profile}")
    if depot_mode not in ALL_MODES:
        raise ValueError(f"depot_mode must be one of {ALL_MODES}, "
                         f"got {depot_mode!r}")

    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    B, N, depot = int(batch_size), int(n_customer), int(n_customer)
    sL, sR = S_LAUNCH, S_RECOVER

    # ---- the v4 RNG stream, draw for draw ---------------------------------
    cust = torch.rand(B, N, 2, device=device, generator=g) * SIDE

    vds = V_DRONE if v_drone_set is None else tuple(float(x) for x in v_drone_set)
    eds = ENDURANCE if endurance_set is None else tuple(float(x) for x in endurance_set)
    if balanced:
        v_dr = _balanced(vds, B, device, g)
        e = _balanced(eds, B, device, g)
    else:
        v_dr = torch.tensor(vds, device=device)[
            torch.randint(0, len(vds), (B,), device=device, generator=g)]
        e = torch.tensor(eds, device=device)[
            torch.randint(0, len(eds), (B,), device=device, generator=g)]

    elig = draw_elig_v4(B, N, device, g, elig_mode, inelig_rate, min_inelig,
                        max_inelig, inelig_grid)

    # ---- depot placement: deterministic, or drawn AFTER the v4 stream -----
    if depot_mode in SINGLE_MODES:
        midx = torch.full((B,), SINGLE_MODES.index(depot_mode),
                          dtype=torch.long, device=device)
    elif depot_mode == "murray3":
        base = 1 + torch.arange(B, device=device) % 3
        midx = base[torch.randperm(B, device=device, generator=g)]
    else:                                                          # mix4
        base = torch.arange(B, device=device) % 4
        midx = base[torch.randperm(B, device=device, generator=g)]

    coords = torch.cat([cust, _depot_xy(cust, midx)[:, None, :]], dim=1)

    # ---- everything below is generate_batch_v4 verbatim -------------------
    T = _manhattan(coords) / V_TRUCK * 60.0
    D = _euclidean(coords) / v_dr[:, None, None] * 60.0

    typ = SIDE / V_TRUCK * 60.0
    feat = torch.zeros(B, N + 1, FEAT_DIM_BASE, device=device)
    feat[:, :, 0:2] = coords / SIDE
    feat[:, :, 2] = elig.float()
    feat[:, :, 3] = (e / typ)[:, None]
    feat[:, :, 4] = (v_dr / V_TRUCK)[:, None]
    feat[:, :, 5] = sL / typ
    feat[:, :, 6] = sR / typ
    feat[:, depot, 7] = 1.0

    ef = torch.zeros(B, N + 1, N + 1, EDGE_DIM_BASE, device=device)
    ef[:, :, :, 0] = T / typ
    ef[:, :, :, 1] = D / e[:, None, None].clamp_min(1e-6)
    ef[:, :, :, 2] = (T - D) / typ
    ef[:, :, :, 3] = (D <= e[:, None, None]).float()

    out = add_v3_features({
        "coords": coords, "feat": feat, "edge_feat": ef, "T": T, "D": D,
        "elig": elig, "e": e, "sL": sL, "sR": sR, "depot": depot,
        "profile": "setM",
    })
    out["depot_mode_idx"] = midx        # extra key; make_val/testset drop or
    return out                          # record it, nothing downstream breaks


@torch.no_grad()
def classify_depot(batch, tol=1e-4):
    """Recover the mode from coords: [B] long in {0..3}, -1 if unknown.

    make_val and the frozen test sets only keep the tensor keys, so the mode
    must be recoverable from the data itself.  Corner wins over edge (both
    have y = 0), centroid wins over center on the rare near-collision.
    """
    d = int(batch["depot"])
    c = batch["coords"]
    xy, cust = c[:, d], c[:, :d]
    mean = cust.mean(1)
    out = torch.full((c.shape[0],), -1, dtype=torch.long, device=c.device)
    out[(xy - SIDE / 2.0).abs().amax(1) < tol] = 0
    ed = (xy[:, 1].abs() < tol) & ((xy[:, 0] - mean[:, 0]).abs() < tol)
    out[ed] = 2
    out[(xy - mean).abs().amax(1) < tol] = 1
    out[xy.abs().amax(1) < tol] = 3
    return out


# ---------------------------------------------------------------------------
# self-checks
# ---------------------------------------------------------------------------
def assert_matches_v4(n=20, B=64, seed=12345, device="cpu",
                      elig_mode="strat"):
    """depot_mode='center' must reproduce generate_batch_v4 bit for bit."""
    a = generate_batch_v4(B, n_customer=n, profile="setM", device=device,
                          seed=seed, balanced=True, elig_mode=elig_mode)
    b = generate_batch_depot(B, n_customer=n, profile="setM", device=device,
                             seed=seed, balanced=True, depot_mode="center",
                             elig_mode=elig_mode)
    for k in ("coords", "feat", "edge_feat", "T", "D", "e"):
        assert torch.equal(a[k], b[k]), f"'{k}' differs from generate_batch_v4"
    assert torch.equal(a["elig"], b["elig"]), "'elig' differs"
    print(f"ok: depot_mode='center' == generate_batch_v4 bit-for-bit "
          f"(n={n}, B={B}, seed={seed}, elig_mode={elig_mode})")


def assert_controlled(n=20, B=96, seed=777, device="cpu"):
    """All modes under one seed share coords / v_drone / e / elig."""
    ref = None
    for m in ALL_MODES:
        b = generate_batch_depot(B, n_customer=n, device=device, seed=seed,
                                 depot_mode=m, elig_mode="strat")
        cur = (b["coords"][:, :n], b["feat"][:, 0, 4], b["e"], b["elig"][:, :n])
        if ref is None:
            ref = cur
            continue
        for x, y, nm in zip(ref, cur, ("cust coords", "v_drone", "e", "elig")):
            assert torch.equal(x, y), f"{nm} differs under depot_mode={m}"
    print(f"ok: all modes share customer coords / v_drone / e / elig "
          f"under one seed (controlled)")


def assert_classify(n=20, B=99, seed=31, device="cpu"):
    for mode, want in (("murray3", (1, 2, 3)), ("mix4", (0, 1, 2, 3))):
        b = generate_batch_depot(B, n_customer=n, device=device, seed=seed,
                                 depot_mode=mode, elig_mode="strat")
        got = classify_depot(b)
        assert torch.equal(got, b["depot_mode_idx"].cpu().to(got.device)), \
            f"classify_depot disagrees with the generator under {mode}"
        assert set(got.tolist()) == set(want)
    print("ok: classify_depot recovers the generator's mode assignment")


@torch.no_grad()
def depot_report(n=50, B=512, seed=99, device="cpu"):
    """The observable that actually changes: the depot legs."""
    print(f"\ndepot report  (n={n}, B={B} per mode, same instances everywhere)")
    print(f"  {'mode':>9}  {'mean depot->cust T (min)':>25}  "
          f"{'mean nearest cust T':>20}")
    try:
        from fstsp_data_tau import tau_of
        have_tau = True
        print(f"  {'':>9}  {'':>25}  {'':>20}   mean tau")
    except Exception:
        have_tau = False
    for m in SINGLE_MODES:
        b = generate_batch_depot(B, n_customer=n, device=device, seed=seed,
                                 depot_mode=m, elig_mode="strat")
        d = int(b["depot"])
        Td = b["T"][:, d, :n]
        row = (f"  {m:>9}  {float(Td.mean()):>25.2f}  "
               f"{float(Td.min(1).values.mean()):>20.2f}")
        if have_tau:
            row += f"   {float(tau_of(b).mean()):.4f}"
        print(row)


# ---------------------------------------------------------------------------
# frozen test set (same on-disk format as fstsp_testset_v3)
# ---------------------------------------------------------------------------
def make_testset(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    parts, done = [], 0
    while done < args.num:
        bs = min(args.chunk, args.num - done)
        parts.append(generate_batch_depot(
            bs, n_customer=args.n, profile="setM", device="cpu",
            seed=args.seed + done, balanced=True, depot_mode=args.depot_mode,
            elig_mode=args.elig_mode, inelig_rate=args.inelig_rate,
            min_inelig=args.min_inelig,
            max_inelig=(getattr(args, "max_inelig", 0) or None),
            v_drone_set=getattr(args, "v_drone_set", None),
            endurance_set=getattr(args, "endurance_set", None)))
        done += bs

    keys = ["coords", "feat", "edge_feat", "T", "D", "elig", "e"]
    batch = {k: torch.cat([p[k] for p in parts], 0) for k in keys}
    batch.update({k: parts[0][k] for k in ("sL", "sR", "depot", "profile")})
    midx = torch.cat([p["depot_mode_idx"] for p in parts], 0)

    # classify must round-trip, otherwise stratified reporting is impossible
    got = classify_depot(batch)
    assert torch.equal(got, midx), "classify_depot does not round-trip"

    h = hashlib.sha256()
    for k in ("T", "D", "e"):
        h.update(batch[k].numpy().tobytes())
    h.update(batch["elig"].numpy().tobytes())

    ine = (~batch["elig"][:, :args.n]).sum(1)
    counts = torch.bincount(midx, minlength=4).tolist()
    meta = {"n": args.n, "num": args.num, "seed": args.seed,
            "profile": "setM", "sL": batch["sL"], "sR": batch["sR"],
            "depot": batch["depot"], "sha16": h.hexdigest()[:16], "ver": 4,
            "feat_dim": FEAT_DIM, "edge_dim": EDGE_DIM,
            "elig_mode": args.elig_mode, "inelig_rate": args.inelig_rate,
            "min_inelig": args.min_inelig,
            "max_inelig": cap_from_rate(args.n),
            "inelig_mean": float(ine.float().mean()),
            "depot_mode": args.depot_mode,
            "v_drone_set": list(getattr(args, "v_drone_set", None) or V_DRONE),
            "endurance_set": list(getattr(args, "endurance_set", None)
                                  or ENDURANCE),
            "depot_counts": dict(zip(SINGLE_MODES, counts)),
            "depot_mode_idx": midx.tolist(),
            "generator": "fstsp_data_depot.generate_batch_depot "
                         "(cpu, balanced=True)"}
    batch["meta"] = meta

    torch.save(batch, args.out)
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    e = batch["e"]
    print(json.dumps({k: v for k, v in meta.items()
                      if k != "depot_mode_idx"}, indent=2))
    print(f"e=20 {int((e == 20).sum())}  e=40 {int((e == 40).sum())}   "
          f"depot {meta['depot_counts']}")
    print(f"ineligible/inst: mean {float(ine.float().mean()):.2f}  "
          f"range [{int(ine.min())}, {int(ine.max())}]")
    print(f"written: {args.out}")
    print("\nnext: export for the HGA baseline (the depot flows through the "
          "distance matrices, run_hgatac.jl needs no change):")
    stem = os.path.splitext(os.path.basename(args.out))[0]
    print(f"  python fstsp_export_hga.py --data {args.out} --out hga/{stem}")




# =============================================================================
# unified front door
# =============================================================================
generate_batch = generate_batch_depot          # canonical name going forward
