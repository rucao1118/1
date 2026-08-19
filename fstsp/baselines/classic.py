import argparse
import json
import os
import time
import math

import numpy as np
import pandas as pd
import torch

from ..dp import SplitDPTorch


TIE = 1e-9


def load_raw(path, device="cpu"):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    meta = raw.pop("meta", {})
    raw = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in raw.items()
    }
    return raw, meta


@torch.no_grad()
def split_cost_many(raw, inst, orders):
    """
    raw: full batch
    inst: int instance id
    orders: [R, L] tensor/ndarray, L can be <= N during repair
    """
    if isinstance(orders, np.ndarray):
        orders = torch.tensor(orders, dtype=torch.long)

    orders = orders.long()
    if orders.ndim == 1:
        orders = orders[None]

    dev = raw["T"].device
    orders = orders.to(dev)
    R, L = orders.shape

    idx = torch.full((R,), int(inst), dtype=torch.long, device=dev)
    depot = int(raw["depot"])
    sL, sR = float(raw["sL"]), float(raw["sR"])

    eng = SplitDPTorch(
        raw["T"][idx].double(),
        raw["D"][idx].double(),
        raw["elig"][idx],
        raw["e"][idx].double(),
        sL,
        sR,
        depot,
        Lmax=L + 2,
    )

    for t in range(L):
        eng.append(orders[:, t])

    eng.append(torch.full((R,), depot, dtype=torch.long, device=dev))
    return eng.dp[:, L + 1].detach().cpu().numpy()


def split_cost_one(raw, inst, order):
    return float(split_cost_many(raw, inst, np.asarray(order, dtype=np.int64))[0])



def tour_time(raw, inst, order):
    """Truck-only makespan of `order` (no drone): the Murray savings
    denominator.  Pure sum over T, depot -> order -> depot."""
    T = raw["T"][inst].detach().cpu().numpy()
    depot = int(raw["depot"])
    c, cur = 0.0, depot
    for j in order:
        c += float(T[cur, int(j)])
        cur = int(j)
    return c + float(T[cur, depot])

def nearest_order(raw, inst):
    T = raw["T"][inst].detach().cpu().numpy()
    V = T.shape[0]
    N = V - 1
    depot = int(raw["depot"])

    unvis = set(range(N))
    cur = depot
    order = []

    while unvis:
        nxt = min(unvis, key=lambda j: T[cur, j])
        order.append(nxt)
        unvis.remove(nxt)
        cur = nxt

    return np.asarray(order, dtype=np.int64)


def ortools_order(raw, inst, time_limit=1.0):
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return None, 0.0, "ortools_not_installed"

    t0 = time.perf_counter()

    T = raw["T"][inst].detach().cpu().numpy()
    V = T.shape[0]
    N = V - 1
    depot = int(raw["depot"])

    scale = 1_000_000
    C = np.rint(T * scale).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(V, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(C[i, j])

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    sec = int(time_limit)
    nanos = int((float(time_limit) - sec) * 1e9)
    params.time_limit.seconds = sec
    params.time_limit.nanos = nanos

    sol = routing.SolveWithParameters(params)
    elapsed = time.perf_counter() - t0

    if sol is None:
        return None, elapsed, "no_solution"

    idx = routing.Start(0)
    route = []

    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        if node != depot:
            route.append(node)
        idx = sol.Value(routing.NextVar(idx))

    if len(route) != N:
        return None, elapsed, "bad_route"

    return np.asarray(route, dtype=np.int64), elapsed, "ok"


# ---------------------------------------------------------------------
# Simple self-contained ALNS over permutations.
# Objective = exact split DP.
# ---------------------------------------------------------------------
def destroy_random(order, rng, q):
    pos = rng.choice(len(order), size=q, replace=False)
    pos = np.sort(pos)
    removed = order[pos].tolist()
    keep = np.delete(order, pos)
    return keep, removed


def destroy_segment(order, rng, q):
    N = len(order)
    if q >= N:
        return np.asarray([], dtype=np.int64), order.tolist()
    s = int(rng.integers(0, N - q + 1))
    removed = order[s:s + q].tolist()
    keep = np.concatenate([order[:s], order[s + q:]])
    return keep, removed


def destroy_related(raw, inst, order, rng, q):
    coords = raw["coords"][inst].detach().cpu().numpy()
    seed = int(rng.choice(order))
    xy = coords[seed]

    dist = []
    for p, node in enumerate(order):
        d = float(np.linalg.norm(coords[int(node)] - xy))
        dist.append((d, p))

    dist.sort()
    pos = np.array([p for _, p in dist[:q]], dtype=np.int64)
    pos = np.sort(pos)

    removed = order[pos].tolist()
    keep = np.delete(order, pos)
    return keep, removed


def greedy_repair(raw, inst, partial, removed, rng):
    seq = list(map(int, partial.tolist()))
    rem = list(map(int, removed))
    rng.shuffle(rem)

    for node in rem:
        cand = []
        for pos in range(len(seq) + 1):
            cand.append(seq[:pos] + [node] + seq[pos:])

        costs = split_cost_many(raw, inst, np.asarray(cand, dtype=np.int64))
        best_pos = int(costs.argmin())
        seq.insert(best_pos, node)

    return np.asarray(seq, dtype=np.int64)


def alns_one(raw, inst, init_order, time_limit=2.0, seed=0,
             min_remove=2, max_remove=6, start_temp=1.0, cooling=0.995):
    rng = np.random.default_rng(seed)

    cur = np.asarray(init_order, dtype=np.int64).copy()
    cur_c = split_cost_one(raw, inst, cur)

    best = cur.copy()
    best_c = cur_c

    temp = float(start_temp)
    t0 = time.perf_counter()
    it = 0

    destroy_ops = ["random", "segment", "related"]

    while time.perf_counter() - t0 < time_limit:
        N = len(cur)
        q = int(rng.integers(min_remove, min(max_remove, N - 1) + 1))

        op = rng.choice(destroy_ops)
        if op == "random":
            partial, removed = destroy_random(cur, rng, q)
        elif op == "segment":
            partial, removed = destroy_segment(cur, rng, q)
        else:
            partial, removed = destroy_related(raw, inst, cur, rng, q)

        cand = greedy_repair(raw, inst, partial, removed, rng)
        cand_c = split_cost_one(raw, inst, cand)

        if cand_c < best_c - TIE:
            best = cand.copy()
            best_c = cand_c

        accept = cand_c < cur_c - TIE
        if not accept:
            prob = math.exp((cur_c - cand_c) / max(temp, 1e-9))
            accept = rng.random() < prob

        if accept:
            cur = cand
            cur_c = cand_c

        temp *= cooling
        it += 1

    elapsed = time.perf_counter() - t0
    return best, best_c, elapsed, it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test_n20_v3.pt")
    ap.add_argument("--out", default="baselines/n20_baseline.csv")
    ap.add_argument("--methods", nargs="+", default=["ortools", "alns"],
                    choices=["ortools", "alns"])
    ap.add_argument("--ortools_time", type=float, default=1.0)
    ap.add_argument("--alns_time", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    raw, meta = load_raw(args.data, device="cpu")
    B, V, _ = raw["T"].shape
    N = V - 1

    print("=" * 80)
    print(f"data : {args.data}")
    print(f"n    : {N}")
    print(f"inst : {B}")
    print(f"out  : {args.out}")
    print(f"methods: {args.methods}")
    print("=" * 80)

    rows = []

    for inst in range(B):
        row = {"inst": inst}

        # ---------------- OR-Tools ----------------
        ort_order = None
        if "ortools" in args.methods:
            o, tor, status = ortools_order(raw, inst, args.ortools_time)
            if o is None:
                o = nearest_order(raw, inst)
                status = f"{status};fallback_nearest"

            c = split_cost_one(raw, inst, o)
            ort_order = o

            row.update({
                "ortools_status": status,
                "truck_only_cost": tour_time(raw, inst, o),
                "ortools_cost": c,
                "ortools_time": tor,
                "ortools_order": json.dumps([int(x) for x in o]),
            })

        # ---------------- ALNS ----------------
        if "alns" in args.methods:
            init = ort_order if ort_order is not None else nearest_order(raw, inst)

            o, c, ta, it = alns_one(
                raw,
                inst,
                init_order=init,
                time_limit=args.alns_time,
                seed=args.seed + inst,
            )

            row.update({
                "alns_cost": c,
                "alns_time": ta,
                "alns_iter": it,
                "alns_order": json.dumps([int(x) for x in o]),
            })

        # ---------------- best ----------------
        candidates = []

        if "ortools_cost" in row:
            candidates.append(("ortools", row["ortools_cost"], row["ortools_order"]))
        if "alns_cost" in row:
            candidates.append(("alns", row["alns_cost"], row["alns_order"]))

        bm, bc, bo = min(candidates, key=lambda x: x[1])

        row.update({
            "best_method": bm,
            "best_cost": bc,
            "best_order": bo,
        })

        rows.append(row)

        if (inst + 1) % 16 == 0 or inst + 1 == B:
            df_tmp = pd.DataFrame(rows)
            print(
                f"{inst + 1:4d}/{B}  "
                f"best_mean={df_tmp['best_cost'].mean():.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    report = {
        "data": args.data,
        "out": args.out,
        "n": N,
        "num": B,
        "methods": args.methods,
        "ortools_time": args.ortools_time,
        "alns_time": args.alns_time,
        "seed": args.seed,
        "meta": meta,
        "summary": {
            "ortools_mean": float(df["ortools_cost"].mean()) if "ortools_cost" in df else None,
            "alns_mean": float(df["alns_cost"].mean()) if "alns_cost" in df else None,
            "best_mean": float(df["best_cost"].mean()),
            "ortools_sec_per_inst": float(df["ortools_time"].mean()) if "ortools_time" in df else None,
            "alns_sec_per_inst": float(df["alns_time"].mean()) if "alns_time" in df else None,
        },
    }

    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print(df.describe(include="all"))
    print()
    print(f"written: {args.out}")
    print(f"written: {json_path}")


if __name__ == "__main__":
    main()