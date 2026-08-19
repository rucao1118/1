r"""
Gurobi MILP for the FSTSP under the PIPELINE cost convention
(variant="current" in fstsp.exact, i.e. exactly what SplitDPTorch scores and
what every proven Set M number was proven against):

    truck-only leg          cost = travel time
    sortie i -> j -> k      cost = launch(i) + max(tr, dr) + sR
                            launch(i) = 0 iff i is the initial depot, else sL
                            tr = truck path time i..k (skipping j)
                            dr = D[i,j] + D[j,k]
    endurance               max(tr, dr) + sR <= e      (launch service outside)
    one drone, one customer per sortie, no overlapping sorties; a sortie may
    end where the next one launches (land-and-relaunch pays sL there).

WHAT THIS IS FOR
    Not a competitive solver.  Two paper jobs:
      1  an external reference row at n=20: solved/unsolved counts and MIPGap
         under a wall-clock cap, produced by a solver nobody in the pipeline
         wrote;
      2  the DUAL BOUND.  Even unsolved, Gurobi's best bound is a certified
         lower bound per instance, so every method's true optimality gap gets
         an upper bound:  gap(method) <= method_cost / bound - 1.

MODEL (internal nodes 0=start depot, 1..n=customers, n+1=end depot)
    x[i,j]  bin   truck arc
    y[i,j,k] bin  sortie launch i, serve j (eligible), rendezvous k
                  (created only when D[i,j]+D[j,k]+sR <= e and T[i,k]+sR <= e)
    z[i,k]  bin   = sum_j y[i,j,k]
    u[c]    cont  MTZ position, u0=0, u_end=n+1
    a,c,d   cont  truck arrival / completion / departure times, d0=0
    w[i,k,l] bin  disjunction: launch l sits before i or after k in route
                  order whenever z[i,k]=1 (interval exclusion, one drone)

    a_j >= d_i + T_ij - Mt(1-x_ij)
    c_k >= a_k ;  c_k >= a_k + sR - Mt(1-rend_k)
    c_k >= d_i + D_ij + D_jk + sR - Mt(1-y_ijk)
    d_i >= c_i + sL*launch_i          (i = 1..n; d_0 = 0: first launch free)
    c_k - d_i <= e + Mt(1-z_ik)       (endurance; drone side pruned at build)
    objective  min c_end

    Mt = heuristic upper bound (NN + 2-opt order scored by the exact split
    DP), also installed as a MIP start and as the constraint c_end <= UB.

VERIFY FIRST, ALWAYS
    python -m fstsp.baselines.gurobi_milp --selftest
    solves n in {6,7,8} random Murray-style instances to zero gap and asserts
    equality (1e-5) with fstsp.exact.exact_dp(variant="current"), which is the
    solver behind the proven n=10 optima.  Run this before trusting any row.

RUN ON A FROZEN SET
    python -m fstsp.baselines.gurobi_milp --data data/test_n20_mix4.pt \
        --sample 24 --time_limit 1800 --threads 8 --out eval/n20_gurobi.csv
"""

import argparse
import json
import math
import os
import time

import numpy as np

from ..exact import split_dp, exact_dp, exact_perm

TOL = 1e-9


# ---------------------------------------------------------------- heuristic
def nn_2opt_order(T, N, depot, passes=40):
    """Nearest neighbour + first-improvement 2-opt on the TRUCK metric.
    Only used to build the warm start / big-M; quality is irrelevant to
    correctness."""
    unvis = set(range(N))
    cur, order = depot, []
    while unvis:
        nxt = min(unvis, key=lambda j: T[cur, j])
        order.append(nxt)
        unvis.remove(nxt)
        cur = nxt
    order = np.asarray(order, dtype=np.int64)

    def tour_len(o):
        c, cu = 0.0, depot
        for j in o:
            c += T[cu, j]
            cu = j
        return c + T[cu, depot]

    best = tour_len(order)
    for _ in range(passes):
        improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                cand = np.concatenate([order[:i], order[i:j + 1][::-1],
                                       order[j + 1:]])
                c = tour_len(cand)
                if c < best - 1e-9:
                    order, best, improved = cand, c, True
        if not improved:
            break
    return order


def heuristic_ub(T, D, elig, e, N, sL, sR, depot):
    """(ub_cost, ops, order) under the pipeline convention."""
    order = nn_2opt_order(T, N, depot)
    ub, ops = split_dp(T, D, elig, e, order, sL=sL, sR=sR, depot=depot,
                       variant="current", want_ops=True)
    return float(ub), ops, order


# ------------------------------------------------------------------- model
def build_and_solve(T, D, elig, e, N, sL=1.0, sR=1.0, depot=None,
                    time_limit=0.0, threads=0, mip_gap=None, log=False,
                    warm=True, mipfocus=0):
    """
    Returns dict(status, obj, bound, gap, time_s, nodes, ub_heur).
    T, D are the pipeline's (N+1)x(N+1) matrices with depot = N (default).
    """
    import gurobipy as gp
    from gurobipy import GRB

    if depot is None:
        depot = N
    T = np.asarray(T, float)
    D = np.asarray(D, float)
    elig = np.asarray(elig, bool)
    e = float(e)

    # internal index -> pipeline index (0 and n+1 are the depot)
    def px(v):
        return depot if v in (0, N + 1) else v - 1

    Ti = np.zeros((N + 2, N + 2))
    Di = np.zeros((N + 2, N + 2))
    for a_ in range(N + 2):
        for b_ in range(N + 2):
            Ti[a_, b_] = T[px(a_), px(b_)]
            Di[a_, b_] = D[px(a_), px(b_)]
    el = np.zeros(N + 2, dtype=bool)
    for c_ in range(1, N + 1):
        el[c_] = bool(elig[c_ - 1])

    ub, ops, horder = heuristic_ub(T, D, elig, e, N, sL, sR, depot)
    Mt = ub + 1e-6
    Mp = N + 2

    C = list(range(1, N + 1))
    S0 = [0] + C                      # possible launch nodes
    KE = C + [N + 1]                  # possible rendezvous nodes

    m = gp.Model("fstsp_current")
    if not log:
        m.Params.OutputFlag = 0
    if time_limit:
        m.Params.TimeLimit = float(time_limit)
    if threads:
        m.Params.Threads = int(threads)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)
    if mipfocus:
        m.Params.MIPFocus = int(mipfocus)

    # ---- variables ---------------------------------------------------------
    arcs = [(i, j) for i in [0] + C for j in C + [N + 1] if i != j]
    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")

    ytrip = []
    for i in S0:
        for j in C:
            if not el[j] or j == i:
                continue
            for k in KE:
                if k in (i, j):
                    continue
                if Di[i, j] + Di[j, k] + sR > e + TOL:
                    continue          # drone side can never fit
                if Ti[i, k] + sR > e + TOL:
                    continue          # truck side can never fit (metric T)
                ytrip.append((i, j, k))
    y = m.addVars(ytrip, vtype=GRB.BINARY, name="y")

    zpair = sorted({(i, k) for (i, j, k) in ytrip})
    z = m.addVars(zpair, vtype=GRB.BINARY, name="z")

    u = m.addVars(C, lb=1.0, ub=float(N), vtype=GRB.CONTINUOUS, name="u")
    uu = {0: 0.0, N + 1: float(N + 1), **{c_: u[c_] for c_ in C}}

    av = m.addVars(range(1, N + 2), lb=0.0, ub=Mt, name="a")
    cv = m.addVars(range(1, N + 2), lb=0.0, ub=Mt, name="c")
    dv = m.addVars(range(0, N + 1), lb=0.0, ub=Mt, name="d")
    m.addConstr(dv[0] == 0.0)

    # ---- degree / assignment ----------------------------------------------
    visit = {c_: gp.quicksum(x[i, c_] for i in [0] + C if i != c_) for c_ in C}
    m.addConstr(gp.quicksum(x[0, j] for j in C + [N + 1]) == 1)
    m.addConstr(gp.quicksum(x[i, N + 1] for i in [0] + C) == 1)
    for c_ in C:
        m.addConstr(gp.quicksum(x[c_, k] for k in C + [N + 1] if k != c_)
                    == visit[c_])
        dserv = gp.quicksum(y[i, j, k] for (i, j, k) in ytrip if j == c_)
        m.addConstr(visit[c_] + dserv == 1, name=f"serve[{c_}]")

    # ---- MTZ ---------------------------------------------------------------
    for (i, j) in arcs:
        if j == N + 1:
            continue
        m.addConstr(u[j] >= uu[i] + 1.0 - (N + 1) * (1 - x[i, j]))

    # ---- sortie structure --------------------------------------------------
    launch = {i: gp.quicksum(z[i, k] for (ii, k) in zpair if ii == i)
              for i in S0}
    rend = {k: gp.quicksum(z[i, k] for (i, kk) in zpair if kk == k)
            for k in KE}
    for (i, k) in zpair:
        m.addConstr(z[i, k] == gp.quicksum(
            y[i, j, k] for (ii, j, kk) in ytrip if ii == i and kk == k))
        if i != 0:
            m.addConstr(z[i, k] <= visit[i])
        if k != N + 1:
            m.addConstr(z[i, k] <= visit[k])
        m.addConstr(uu[k] >= uu[i] + 1.0 - Mp * (1 - z[i, k]))
    for i in S0:
        m.addConstr(launch[i] <= 1)
    for k in KE:
        m.addConstr(rend[k] <= 1)

    # ---- one drone: no OTHER launch strictly inside an active interval -----
    lset = {i for (i, k) in zpair}          # nodes that can launch at all
    w = {}
    for (i, k) in zpair:
        for l in lset:
            if l in (i, k):
                continue
            w[i, k, l] = m.addVar(vtype=GRB.BINARY, name=f"w[{i},{k},{l}]")
            m.addConstr(uu[l] <= uu[i]
                        + Mp * (1 - z[i, k]) + Mp * (1 - launch[l])
                        + Mp * w[i, k, l])
            m.addConstr(uu[l] >= uu[k]
                        - Mp * (1 - z[i, k]) - Mp * (1 - launch[l])
                        - Mp * (1 - w[i, k, l]))

    # ---- timing -------------------------------------------------------------
    for (i, j) in arcs:
        di = dv[i] if i <= N else None
        m.addConstr(av[j] >= di + Ti[i, j] - Mt * (1 - x[i, j]))
    for k in KE:
        m.addConstr(cv[k] >= av[k])
        m.addConstr(cv[k] >= av[k] + sR - Mt * (1 - rend[k]))
    for (i, j, k) in ytrip:
        m.addConstr(cv[k] >= dv[i] + Di[i, j] + Di[j, k] + sR
                    - Mt * (1 - y[i, j, k]))
    for c_ in C:
        m.addConstr(dv[c_] >= cv[c_] + sL * launch[c_])
    for (i, k) in zpair:
        m.addConstr(cv[k] - dv[i] <= e + Mt * (1 - z[i, k]),
                    name=f"endur[{i},{k}]")

    m.addConstr(cv[N + 1] <= ub + 1e-6, name="incumbent_cap")
    m.setObjective(cv[N + 1], GRB.MINIMIZE)

    # ---- warm start from the heuristic split ------------------------------
    if warm:
        route = [depot] + [int(v) for v in horder] + [depot]

        def node_of(pos):
            return 0 if pos == 0 else (N + 1 if pos == N + 1
                                       else int(horder[pos - 1]) + 1)

        flown = {jp for (_, _, jp) in ops if jp != -1}
        tpos = [p for p in range(N + 2) if p not in flown]
        for v in x.values():
            v.Start = 0.0
        for v in y.values():
            v.Start = 0.0
        for p0, p1 in zip(tpos[:-1], tpos[1:]):
            key = (node_of(p0), node_of(p1))
            if key in x:
                x[key].Start = 1.0
        for rank, p in enumerate(tpos[1:-1], start=1):
            u[node_of(p)].Start = float(rank)
        for (ip, kp, jp) in ops:
            if jp == -1:
                continue
            key = (node_of(ip), node_of(jp), node_of(kp))
            if key in y:
                y[key].Start = 1.0

    t0 = time.perf_counter()
    m.optimize()
    el = time.perf_counter() - t0

    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
              GRB.INTERRUPTED: "interrupted",
              GRB.INFEASIBLE: "INFEASIBLE"}.get(m.Status, str(m.Status))
    obj = float(m.ObjVal) if m.SolCount else float("nan")
    bound = float(m.ObjBound) if m.Status != 3 else float("nan")
    gap = float(m.MIPGap) if m.SolCount else float("nan")
    return dict(status=status, obj=obj, bound=bound, gap=gap,
                time_s=el, nodes=int(m.NodeCount), ub_heur=ub)


# ---------------------------------------------------------------- selftest
def _rand_instance(n, rng, depot_style="mix"):
    SIDE, VT = 8.0, 25.0
    cust = rng.uniform(0, SIDE, size=(n, 2))
    if depot_style == "center":
        dxy = np.array([SIDE / 2, SIDE / 2])
    else:
        dxy = [np.array([SIDE / 2, SIDE / 2]), cust.mean(0),
               np.array([cust[:, 0].mean(), 0.0]), np.array([0.0, 0.0])][
                   int(rng.integers(0, 4))]
    coords = np.vstack([cust, dxy[None]])
    v = float(rng.choice([15.0, 25.0, 35.0]))
    e = float(rng.choice([20.0, 40.0]))
    T = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1) / VT * 60.0
    Dm = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)) \
        / v * 60.0
    elig = np.ones(n + 1, dtype=bool)
    elig[n] = False
    nbad = int(rng.integers(1, 3))
    elig[rng.choice(n, size=nbad, replace=False)] = False
    return T, Dm, elig, e


def selftest(seeds=(0, 1, 2), sizes=(6, 7, 8), time_limit=120.0, threads=0,
             perm_check=True):
    print("=" * 76)
    print("SELFTEST: MILP vs fstsp.exact.exact_dp, variant='current'")
    print("=" * 76)
    bad = 0
    for n in sizes:
        for sd in seeds:
            rng = np.random.default_rng(1000 * n + sd)
            T, Dm, elig, e = _rand_instance(n, rng)
            ref = exact_dp(T, Dm, elig, e, n, sL=1.0, sR=1.0, depot=n,
                           variant="current")
            if perm_check and n <= 6:
                pv, _ = exact_perm(T, Dm, elig, e, n, sL=1.0, sR=1.0,
                                   depot=n, variant="current")
                assert abs(pv - ref) < 1e-6, \
                    f"exact_dp vs exact_perm disagree: {ref} vs {pv}"
            res = build_and_solve(T, Dm, elig, e, n, sL=1.0, sR=1.0, depot=n,
                                  time_limit=time_limit, threads=threads,
                                  mip_gap=1e-9)
            ok = (res["status"] == "optimal"
                  and abs(res["obj"] - ref) < 1e-5)
            bad += 0 if ok else 1
            print(f"  n={n} seed={sd}: exact {ref:10.5f}   milp "
                  f"{res['obj']:10.5f}   ub {res['ub_heur']:10.5f}   "
                  f"{res['status']:>9}   {res['time_s']:6.2f}s   "
                  f"{'OK' if ok else '!! MISMATCH'}")
            assert res["ub_heur"] >= ref - 1e-6, "heuristic UB below optimum?!"
    print("=" * 76)
    if bad:
        raise SystemExit(f"{bad} mismatches -- do NOT use this MILP for any "
                         f"paper row until fixed")
    print("all matched to 1e-5.  The MILP implements the pipeline convention.")


# ------------------------------------------------------------------- runner
def load_pt(path):
    import torch
    raw = torch.load(path, map_location="cpu", weights_only=False)
    meta = raw.pop("meta", {})
    out = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in raw.items()}
    return out, meta


def stratified_sample(raw, k, seed=0):
    """k instances spread evenly over (v_drone, e) cells, in-cell random."""
    B = raw["T"].shape[0]
    v = np.round(raw["feat"][:, 0, 4] * 25.0).astype(int)
    e = raw["e"].reshape(-1).astype(int)
    rng = np.random.default_rng(seed)
    cells = {}
    for i in range(B):
        cells.setdefault((v[i], e[i]), []).append(i)
    for c in cells.values():
        rng.shuffle(c)
    out, r = [], 0
    while len(out) < min(k, B):
        for c in sorted(cells):
            if r < len(cells[c]) and len(out) < k:
                out.append(cells[c][r])
        r += 1
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", default="")
    ap.add_argument("--inst", nargs="+", type=int, default=None,
                    help="explicit instance ids; default: --sample")
    ap.add_argument("--sample", type=int, default=24,
                    help="stratified sample size over (v_drone, e)")
    ap.add_argument("--sample_seed", type=int, default=20260819)
    ap.add_argument("--time_limit", type=float, default=1800.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mipfocus", type=int, default=0,
                    help="2/3 pushes the DUAL bound, the certifying quantity")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--out", default="eval/gurobi_milp.csv")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.data:
        raise SystemExit("pass --data <frozen .pt> or --selftest")

    raw, meta = load_pt(args.data)
    B, V, _ = raw["T"].shape
    N = V - 1
    depot = int(raw["depot"])
    sL, sR = float(raw["sL"]), float(raw["sR"])
    ids = args.inst if args.inst is not None else \
        stratified_sample(raw, args.sample, args.sample_seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"data {args.data}  n={N}  running {len(ids)} of {B} instances, "
          f"cap {args.time_limit:.0f}s each")

    rows = []
    for i in ids:
        T, Dm = raw["T"][i], raw["D"][i]
        elig, e = raw["elig"][i], float(raw["e"][i])
        res = build_and_solve(T, Dm, elig, e, N, sL=sL, sR=sR, depot=depot,
                              time_limit=args.time_limit,
                              threads=args.threads, log=args.log,
                              mipfocus=args.mipfocus)
        row = dict(inst=int(i), n=N,
                   v_drone=int(round(float(raw["feat"][i, 0, 4]) * 25)),
                   e=int(e), n_inelig=int((~raw["elig"][i, :N]).sum()),
                   **res)
        row["gap_pct"] = 100.0 * row["gap"] if math.isfinite(row["gap"]) \
            else float("nan")
        rows.append(row)
        print(f"  inst {i:4d}  {row['status']:>10}  obj {row['obj']:9.4f}  "
              f"bound {row['bound']:9.4f}  gap {row['gap_pct']:6.2f}%  "
              f"{row['time_s']:7.1f}s")
        _write(rows, args, N, meta)

    _write(rows, args, N, meta)
    solved = sum(r["status"] == "optimal" for r in rows)
    gaps = [r["gap_pct"] for r in rows if math.isfinite(r["gap_pct"])]
    print(f"\nsolved {solved}/{len(rows)}   mean MIPGap "
          f"{np.mean(gaps):.2f}%   written: {args.out}")
    print("the 'bound' column certifies EVERY method: "
          "gap(method, inst) <= cost/bound - 1")


def _write(rows, args, N, meta):
    import csv
    keys = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys)
        wcsv.writeheader()
        wcsv.writerows(rows)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"data": args.data, "n": N, "time_limit": args.time_limit,
                   "convention": "current (strict Murray, pipeline)",
                   "meta": str(meta),
                   "rows": rows}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
