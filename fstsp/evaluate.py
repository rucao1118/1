"""
fstsp.evaluate -- the final-method table (was fstsp_post_v4.py) plus the two
things the paper section needs on top:

  * --k_expand/--beam/--w_dp/--w_lp   the DP-bound-guided beam as a strategy
    row.  --w_dp 0 --w_lp 1 is the SEARCH ABLATION (vanilla NCO beam) and
    needs NO retraining: same checkpoint, different beam score.
  * --strata                          per-stratum means (v_drone, endurance,
    n_inelig, depot mode via classify_depot), optionally with per-stratum
    gaps against a per-instance reference csv (--hga_csv/--hga_col).

Default flags reproduce fstsp_post_v4.py's rows and CSVs exactly:
construct / top{K}+LS / ILS{p}, all re-scored by the float64 split DP.

    python -m fstsp.evaluate --data data/test_n50_mix4.pt \
        --ckpt runs/n50_mix4/tail/best.pt --tag n50 --strata \
        --out eval/n50_final.csv
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .rollout import rollout_v4, beam_decode

from .dp import SplitDPTorch
from .data import FEAT_DIM, EDGE_DIM, classify_depot
from .model import FSTSPv4, inflate_v3_state_dict

# --- lifted verbatim from fstsp_eval_v4_n20 -------------------------------
def load_data(path, device):
    b = torch.load(path, map_location="cpu", weights_only=False)
    meta = b.pop("meta", {})
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    return b, meta



def check_dims(b):
    """The test set was built by fstsp_data_v3.  If v4 widened the feature
    vectors, the tensors on disk cannot be fed to a v4 encoder and every
    number below would be garbage."""
    fd = int(b["feat"].shape[-1])
    ed = int(b["edge_feat"].shape[-1])
    print(f"feat dim  : data {fd}  vs  fstsp_data_v4.FEAT_DIM {FEAT_DIM}")
    print(f"edge dim  : data {ed}  vs  fstsp_data_v4.EDGE_DIM {EDGE_DIM}")
    if fd != FEAT_DIM or ed != EDGE_DIM:
        raise SystemExit(
            "\nFEATURE DIM MISMATCH -- this test set was featurised by v3 and "
            "cannot be fed to a v4 encoder.\nRe-featurise it: rebuild feat/"
            "edge_feat with fstsp_data_v4 from the SAME coords/T/D/elig/e so "
            "the instances stay identical, then re-run.")



def build_model(ckpt_path, device):
    """
    Loads a v4 checkpoint.  A v3 checkpoint is accepted too: it is inflated
    with ZERO weight on the three added candidate channels, which makes it
    exactly the same policy as before -- useful as the baseline row, and a
    strong consistency check on this eval path (it must reproduce the v3
    eval's pomo8_min number).  Inflated rows are renamed so a v3 result can
    never be mistaken for a v4 one downstream.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck.get("args", {})
    ver = int(ck.get("ver", 3))
    m = FSTSPv4(
        in_dim=FEAT_DIM, edge_dim=EDGE_DIM,
        dim=int(a.get("dim", 128)), heads=int(a.get("heads", 8)),
        layers=int(a.get("layers", 4)), ff=int(a.get("ff", 512)),
        use_edge_bias=bool(a.get("use_edge_bias", 1)),
        use_cand_bias=bool(a.get("use_cand_bias", 1)),
        use_dp_state=bool(a.get("use_dp_state", 1)),
        n_dec=int(a.get("n_dec", 1)),
        prenorm=bool(a.get("prenorm", 0)),
    ).to(device)

    inflated = False
    try:
        m.load_state_dict(ck["model"])
    except RuntimeError as err:
        if ver >= 4:
            raise SystemExit(
                f"\n{ckpt_path} says ver={ver} but its weights do not fit a v4 "
                f"model -- this is an architecture mismatch, not a v3 file:\n{err}")
        sd = inflate_v3_state_dict(ck["model"], cand_dim=m.cand_dim,
                                   n_dec=int(a.get("n_dec", 1)), noise=0.0)
        rep = m.load_state_dict(sd, strict=False)
        inflated = True
        print(f"  v3 checkpoint inflated with zero pad "
              f"(missing {len(rep.missing_keys)}, unexpected "
              f"{len(rep.unexpected_keys)}) -- same policy as v3")
        if len(rep.missing_keys) > 4:
            print(f"  !! missing keys: {rep.missing_keys[:12]}")

    m.eval()
    return m, ck, a, ver, inflated


# ---------------------------------------------------------------------------

TIE = 1e-6

# ---------------------------------------------------------------------------
# dp_cost64 / build_moves / local_search / random_perturb / ils are VERBATIM
# copies of fstsp_eval_pomo_n20's, inlined rather than imported so this script
# does not depend on which revision of that file happens to be on disk.  They
# operate on permutations only and never touch the model, so the v3 rows and
# the v4 rows below come out of identical code.
#   diff check:  sed -n '75,242p' fstsp_eval_pomo_n20.py
# ---------------------------------------------------------------------------
@torch.no_grad()
def dp_cost64(orders2d, inst_idx, raw, rows=20000):
    R, N = orders2d.shape
    depot = int(raw["depot"])
    sL, sR = float(raw["sL"]), float(raw["sR"])
    dev = orders2d.device

    out = torch.empty(R, dtype=torch.float64, device=dev)

    for s in range(0, R, rows):
        sl = slice(s, min(s + rows, R))
        ii = inst_idx[sl]

        eng = SplitDPTorch(
            raw["T"][ii].double(),
            raw["D"][ii].double(),
            raw["elig"][ii],
            raw["e"][ii].double(),
            sL,
            sR,
            depot,
            Lmax=N + 2,
        )

        o = orders2d[sl]
        for t in range(N):
            eng.append(o[:, t])

        eng.append(torch.full(
            (o.shape[0],),
            depot,
            dtype=torch.long,
            device=dev,
        ))

        out[sl] = eng.dp[:, N + 1]

    return out


def build_moves(N, kinds):
    base = list(range(N))
    seen = {tuple(base)}
    P = []

    if "2opt" in kinds:
        for i in range(N - 1):
            for j in range(i + 1, N):
                p = base[:i] + base[i:j + 1][::-1] + base[j + 1:]
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    P.append(p)

    for L, key in ((1, "oropt1"), (2, "oropt2"), (3, "oropt3")):
        if key not in kinds:
            continue
        for i in range(N - L + 1):
            seg = base[i:i + L]
            rest = base[:i] + base[i + L:]
            for j in range(len(rest) + 1):
                p = rest[:j] + seg + rest[j:]
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    P.append(p)

    return torch.tensor(P, dtype=torch.long)


@torch.no_grad()
def local_search(orders, raw, P, rows, device, max_pass=60):
    """
    Best-improvement LS over route permutations.
    Each candidate route is scored by exact SplitDPTorch.
    """
    B, N = orders.shape
    M = P.shape[0]
    inst = torch.arange(B, device=device)

    cur = orders.clone()
    curc = dp_cost64(cur, inst, raw, rows)

    Pe = P.to(device)[None].expand(B, M, N)
    flat_idx = inst[:, None].expand(B, M).reshape(-1)

    for _ in range(max_pass):
        cand = cur[:, None, :].expand(B, M, N).gather(2, Pe)

        c = dp_cost64(
            cand.reshape(B * M, N),
            flat_idx,
            raw,
            rows,
        ).reshape(B, M)

        best, bi = c.min(1)
        take = best < curc - TIE

        if not bool(take.any()):
            break

        cur[take] = cand[take, bi[take]]
        curc[take] = best[take]

    return cur, curc


def random_perturb(order, rng, strength=2):
    N = len(order)
    x = order.copy()

    for _ in range(strength):
        if N < 8:
            i, j = sorted(rng.choice(N, 2, replace=False))
            x[i:j + 1] = x[i:j + 1][::-1]
            continue

        cuts = np.sort(rng.choice(np.arange(1, N), 4, replace=False))
        a, b, c, d = cuts.tolist()

        x = np.concatenate([
            x[:a],
            x[c:d],
            x[b:c],
            x[a:b],
            x[d:],
        ])

    return x


@torch.no_grad()
def ils(orders, raw, P, rows, device, passes, seed, strength=2, max_pass=60):
    B, N = orders.shape
    rng = np.random.default_rng(seed + 7000 + passes)

    best = orders.clone()
    inst = torch.arange(B, device=device)
    bestc = dp_cost64(best, inst, raw, rows)

    for _ in range(passes):
        pert = torch.stack([
            torch.tensor(
                random_perturb(
                    best[b].detach().cpu().numpy(),
                    rng,
                    strength=strength,
                ),
                dtype=torch.long,
            )
            for b in range(B)
        ]).to(device)

        loc, c = local_search(
            pert,
            raw,
            P,
            rows=rows,
            device=device,
            max_pass=max_pass,
        )

        take = c < bestc - TIE
        best[take] = loc[take]
        bestc[take] = c[take]

    return best, bestc


# ---------------------------------------------------------------------------


def expand_raw(raw, B, K):
    """repeat every per-instance tensor K times, so row r of a [B*K, N] order
    block is scored against instance r // K"""
    return {k: (v.repeat_interleave(K, 0)
                if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == B else v)
            for k, v in raw.items()}


@torch.no_grad()
def construct(model, raw, look_w, start_mode, rep, device, chunk):
    """-> costs [B,S], orders [B,S,N] for the whole 160-rollout budget"""
    B = raw["T"].shape[0]
    cs, os_ = [], []
    for s in range(0, B, chunk):
        sl = slice(s, min(s + chunk, B))
        part = {k: (v[sl] if torch.is_tensor(v) and v.ndim > 0
                    and v.shape[0] == B else v) for k, v in raw.items()}
        r = rollout_v4(model, part, starts=0, rep=rep, d4_group=True,
                       greedy=True, temperature=1.0, start_mode=start_mode,
                       want_labels=False, force_starts=False, dec=0,
                       look_w=look_w)
        cs.append(r["cost"].detach())
        os_.append(r["orders"].detach())
    return torch.cat(cs, 0), torch.cat(os_, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test_n20_v3.pt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--rep", type=int, default=8, help="D4 framings; 8 = aug8")
    ap.add_argument("--topk_ls", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--ils_passes", nargs="+", type=int, default=[10])
    ap.add_argument("--beam", type=int, default=0,
                    help=">0: also run a DP-bound-guided beam row")
    ap.add_argument("--k_expand", type=int, default=0)
    ap.add_argument("--w_dp", type=float, default=1.0,
                    help="beam score weight on the DP lower bound; 0 = pure "
                         "NCO beam (the search ablation)")
    ap.add_argument("--w_lp", type=float, default=0.1)
    ap.add_argument("--strata", action="store_true",
                    help="also write <out>_strata.csv: means by v_drone, e, "
                         "n_inelig band and depot mode")
    ap.add_argument("--hga_csv", default="",
                    help="optional per-instance reference csv with columns "
                         "inst,<ref_col> for per-stratum gaps")
    ap.add_argument("--hga_col", default="hga_best")
    ap.add_argument("--ils_strength", type=int, default=2)
    ap.add_argument("--ls_moves", nargs="+", default=["2opt", "oropt1", "oropt2"])
    ap.add_argument("--ls_max_pass", type=int, default=60)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--ref_e", type=float, default=20.0)
    ap.add_argument("--ref_mean", type=float, default=83.3175)
    ap.add_argument("--out", default="eval/n20_v4_post.csv")
    ap.add_argument("--out_orders", default="eval/n20_v4_post_orders.csv")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    raw, meta = load_data(args.data, device)
    check_dims(raw)

    B, V, _ = raw["T"].shape
    N = V - 1
    depot = int(raw["depot"])
    inst = torch.arange(B, device=device)
    sub = (raw["e"].reshape(-1).float().cpu().numpy() == args.ref_e)

    model, ck, cargs, ver, inflated = build_model(args.ckpt, device)
    look_w = int(cargs.get("look_w", 0))
    start_mode = str(cargs.get("start_mode", "pomo"))
    tag = f"{args.tag}(v3inf)" if inflated else args.tag

    print("=" * 84)
    print(f"data {args.data}   n={N}  inst={B}   ckpt {args.ckpt} "
          f"(ver {ver}, epoch {ck.get('epoch')})")
    print(f"budget {args.rep * N} rollouts   subset e=={args.ref_e} "
          f"-> {int(sub.sum())} inst   HGA ref {args.ref_mean}")
    print("=" * 84)

    t0 = time.perf_counter()
    C, O = construct(model, raw, look_w, start_mode, args.rep, device, args.chunk)
    t_con = time.perf_counter() - t0
    S = C.shape[1]
    print(f"constructed {S} rollouts/instance in {t_con:.1f}s")

    P = build_moves(N, tuple(args.ls_moves))
    print(f"LS moves: {P.shape[0]} ({', '.join(args.ls_moves)})")

    srt = C.argsort(1)
    best_o = O[inst, srt[:, 0]]

    rows_, orec = [], []

    def add(name, cost, order, sec):
        rc = dp_cost64(order, inst, raw, args.rows).cpu().numpy()
        d = float(np.abs(rc - cost).max())
        m_all, m_sub = float(rc.mean()), float(rc[sub].mean())
        gap = 100 * (m_sub - args.ref_mean) / args.ref_mean
        rows_.append(dict(strategy=name, mean_all=m_all, mean_sub=m_sub,
                          gap_vs_hga_pct=gap, s_per_inst=sec / B,
                          rescore_max_abs=d))
        print(f"  {name:20s} all {m_all:8.4f}   e{args.ref_e:.0f} {m_sub:8.4f}"
              f"   gap {gap:+6.3f}%   {sec / B:7.4f} s/inst   |Δ| {d:.1e}")
        for i in range(B):
            orec.append({"inst": i, "strategy": name, "cost": float(rc[i]),
                         "order": " ".join(str(int(x)) for x in order[i].cpu())})

    add(f"{tag}|construct",
        C.min(1).values.double().cpu().numpy(), best_o, t_con)

    if args.k_expand and args.k_expand > 0:
        t1 = time.perf_counter()
        bc = torch.zeros(B, dtype=torch.float64, device=device)
        bo = torch.zeros(B, N, dtype=torch.long, device=device)
        for s in range(0, B, args.chunk):
            sl = slice(s, min(s + args.chunk, B))
            part = {k: (v[sl] if torch.is_tensor(v) and v.ndim > 0
                        and v.shape[0] == B else v) for k, v in raw.items()}
            c_, o_ = beam_decode(model, part, beam=args.beam,
                                 k_expand=args.k_expand, w_dp=args.w_dp,
                                 w_lp=args.w_lp, d4_mode=0, dec=0,
                                 start_mode=start_mode, look_w=look_w)
            bc[sl] = c_.double()
            bo[sl] = o_
        wtag = (f"{tag}|beam{args.beam or N}x{args.k_expand}"
                f"(wdp={args.w_dp:g},wlp={args.w_lp:g})")
        add(wtag, bc.cpu().numpy(), bo, time.perf_counter() - t1)

    for K in args.topk_ls:
        K = min(K, S)
        t1 = time.perf_counter()
        flat = O.gather(1, srt[:, :K, None].expand(B, K, N)).reshape(B * K, N)
        rrep = expand_raw(raw, B, K)
        o_flat, c_flat = local_search(flat, rrep, P, args.rows, device,
                                      max_pass=args.ls_max_pass)
        cmat, omat = c_flat.reshape(B, K), o_flat.reshape(B, K, N)
        bc, bi = cmat.min(1)
        add(f"{tag}|top{K}+LS", bc.cpu().numpy(), omat[inst, bi],
            t_con + time.perf_counter() - t1)

    for p in args.ils_passes:
        t1 = time.perf_counter()
        oi, ci = ils(best_o.clone(), raw, P, args.rows, device, passes=p,
                     seed=args.seed, strength=args.ils_strength,
                     max_pass=args.ls_max_pass)
        add(f"{tag}|ILS{p}", ci.cpu().numpy(), oi,
            t_con + time.perf_counter() - t1)

    df = pd.DataFrame(rows_)
    df.to_csv(args.out, index=False)
    pd.DataFrame(orec).to_csv(args.out_orders, index=False)

    print()
    print(df.to_string(index=False))
    if float(df.rescore_max_abs.max()) > 1e-4:
        print("\n!! a strategy's reported cost disagrees with the DP rescore")
    print()
    print(f"written: {args.out}")
    print(f"written: {args.out_orders}")

    if args.strata:
        vdr = (raw["feat"][:, 0, 4] * 25.0).round().long().cpu().numpy()
        ee = raw["e"].reshape(-1).float().cpu().numpy().astype(int)
        ine = (~raw["elig"][:, :N]).sum(1).cpu().numpy()
        dmode = classify_depot(raw).cpu().numpy()
        dname = np.array(["center", "centroid", "edge", "corner", "?"])
        instd = pd.DataFrame({"inst": np.arange(B), "v_drone": vdr, "e": ee,
                              "n_inelig": ine,
                              "depot": dname[np.where(dmode >= 0, dmode, 4)]})
        oc = pd.DataFrame(orec).merge(instd, on="inst")
        if args.hga_csv:
            ref = pd.read_csv(args.hga_csv)[["inst", args.hga_col]]
            oc = oc.merge(ref, on="inst", how="left")
            oc["gap_pct"] = 100 * (oc["cost"] - oc[args.hga_col]) \
                / oc[args.hga_col]
        srows = []
        for axis in ("v_drone", "e", "n_inelig", "depot"):
            g = oc.groupby(["strategy", axis])
            agg = g["cost"].agg(["mean", "count"]).reset_index()
            agg = agg.rename(columns={axis: "level", "mean": "mean_cost",
                                      "count": "n_inst"})
            agg.insert(1, "axis", axis)
            if "gap_pct" in oc:
                agg["mean_gap_pct"] = g["gap_pct"].mean().values
            srows.append(agg)
        sdf = pd.concat(srows, ignore_index=True)
        spath = os.path.splitext(args.out)[0] + "_strata.csv"
        sdf.to_csv(spath, index=False)
        print(f"written: {spath}")
        with pd.option_context("display.width", 120):
            print(sdf.to_string(index=False))

    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"data": args.data, "ckpt": args.ckpt, "ver": ver,
                   "epoch": ck.get("epoch"), "n": N, "num": B,
                   "budget": args.rep * N, "ls_moves": args.ls_moves,
                   "ref_e": args.ref_e, "ref_mean": args.ref_mean,
                   "meta": str(meta), "summary": rows_}, f, indent=2)


if __name__ == "__main__":
    main()