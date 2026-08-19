r"""
Exact optimum on the REAL Murray Set M instances, under all three candidate
cost conventions.  This is the experiment that decides which convention the
published Table 12 numbers were produced under.

  python run_setM_exact.py --root path\to\FSTSP_10_customer_problems ^
      --e 20 --sL 1 --sR 1 --out setM_e20_exact.csv

  python run_setM_exact.py ... --ref table12_reference.csv

Instance folder layout (as shipped in TSPDroneHGATAC.jl/test):
  tau.csv        (n+2) x (n+2) truck times
  tauprime.csv   (n+2) x (n+2) drone times
  nodes.csv      nodeID, x, y, flag        <- 4 columns, not Murray's original 5
Row/col 0 is the departure depot, 1..n the customers, n+1 the return depot.

THE FLAG COLUMN IS AMBIGUOUS.  read_files.jl::read_Murray does

    for i = 0:10; if d[i+1, 4] == 1; push!(dEligible, i); end; end

i.e. it treats flag==1 as ELIGIBLE, which leaves only 1-2 flyable customers per
instance.  But solve_tspd_by_HGA_TAC takes drone_ineligible_nodes, so a
complement is taken somewhere, and that is exactly where a sign flip hides.
--flag_means decides it; run both and let the published optima pick the winner.

  --flag_means eligible     flag==1 customers are the only ones that may fly
  --flag_means ineligible   flag==1 customers may NOT fly, everyone else may
  --flag_means all          ignore the flag, everyone may fly (the null model)

Endurance is not stored in the folder; pass it with --e.

If the exact optimum under a convention ever comes out BELOW the best published
heuristic value for an instance, that convention is too permissive.  If it sits
systematically above, it is too expensive.  The convention that matches is the
one to adopt everywhere -- training reward, evaluation, and the RL environment.
"""

import argparse
import csv
import os
import time

import numpy as np

from fstsp.exact import exact_dp

VARIANTS = ("current", "paper", "mip")


def read_matrix(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and any(x.strip() for x in r)]
    return np.array([[float(x) for x in r] for r in rows], dtype=np.float64)


def short_name(folder):
    """20140810T123437v1 -> 37v1, matching the published tables."""
    b = os.path.basename(folder.rstrip("/\\"))
    if "v" in b:
        stem, v = b.rsplit("v", 1)
        return f"{stem[-2:]}v{v}"
    return b


def read_instance(folder, flag_means, time_scale, inelig_override=None):
    tau = read_matrix(os.path.join(folder, "tau.csv")) * time_scale
    taup = read_matrix(os.path.join(folder, "tauprime.csv")) * time_scale
    m = tau.shape[0]
    assert tau.shape == taup.shape and m == tau.shape[1], f"bad shape in {folder}"
    n = m - 2

    # (n+2) depot-duplicated layout -> (n+1) with customers 0..n-1, depot n
    T = np.zeros((n + 1, n + 1))
    D = np.zeros((n + 1, n + 1))
    T[:n, :n] = tau[1:n + 1, 1:n + 1]
    D[:n, :n] = taup[1:n + 1, 1:n + 1]
    T[n, :n] = tau[0, 1:n + 1]          # leaving the depot
    D[n, :n] = taup[0, 1:n + 1]
    T[:n, n] = tau[1:n + 1, n + 1]      # returning to the depot
    D[:n, n] = taup[1:n + 1, n + 1]

    elig = np.ones(n + 1, dtype=bool)
    elig[n] = False
    if inelig_override is not None:
        for c in inelig_override:
            elig[c] = False
        return T, D, elig, n, None

    rows = []
    with open(os.path.join(folder, "nodes.csv")) as f:
        for r in csv.reader(f):
            if not r or not r[0].strip() or r[0].strip().startswith(("%", "#")):
                continue
            rows.append([x.strip() for x in r if x.strip() != ""])
    # rows[0] is the departure depot, rows[1..n] the customers, rows[n+1] the
    # return depot.  Python customer c is Murray node id c+1 is rows[c+1].
    flags = [float(rows[c + 1][3]) for c in range(n)]

    if flag_means == "eligible":
        for c, fl in enumerate(flags):
            if fl != 1:
                elig[c] = False
    elif flag_means == "ineligible":
        for c, fl in enumerate(flags):
            if fl == 1:
                elig[c] = False
    elif flag_means != "all":
        raise ValueError(flag_means)
    return T, D, elig, n, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory of instance folders, or one instance folder")
    ap.add_argument("--e", type=float, required=True)
    ap.add_argument("--sL", type=float, default=1.0)
    ap.add_argument("--sR", type=float, default=1.0)
    ap.add_argument("--flag_means", default="eligible",
                    choices=["eligible", "ineligible", "all"],
                    help="what nodes.csv column 4 == 1 means")
    ap.add_argument("--time_scale", type=float, default=1.0,
                    help="multiply the CSV values by this to get minutes")
    ap.add_argument("--inelig", default=None,
                    help='explicit 0-based ineligible customers, e.g. "2 7"')
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--ref", default=None,
                    help="table12_reference.csv to join against")
    ap.add_argument("--opt", default=None,
                    help="two-column csv of published proven optima: "
                         "short instance name (e.g. 37v1) and value")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="setM_exact.csv")
    args = ap.parse_args()

    inelig = ([int(x) for x in args.inelig.split()]
              if args.inelig else None)

    if os.path.exists(os.path.join(args.root, "tau.csv")):
        folders = [args.root]
    else:
        folders = sorted(
            os.path.join(args.root, d) for d in os.listdir(args.root)
            if os.path.isdir(os.path.join(args.root, d))
            and os.path.exists(os.path.join(args.root, d, "tau.csv")))
    if args.limit:
        folders = folders[:args.limit]
    if not folders:
        raise SystemExit(f"no instance folders with tau.csv under {args.root}")

    published = {}
    if args.opt:
        with open(args.opt) as f:
            for r in csv.reader(f):
                if len(r) < 2:
                    continue
                try:
                    published[r[0].strip()] = float(r[1])
                except ValueError:
                    continue

    print(f"{len(folders)} instances   e={args.e}  sL={args.sL}  sR={args.sR}  "
          f"flag_means={args.flag_means}  time_scale={args.time_scale}")
    print(f"{'inst':<8}{'n':>3}{'inelig':>8}" +
          "".join(f"{v:>12}" for v in args.variants) +
          f"{'sec':>6}  {'pub_opt':>9}  verdict")

    rows = []
    for folder in folders:
        T, D, elig, n, w = read_instance(folder, args.flag_means,
                                         args.time_scale, inelig)
        ni = int((~elig[:n]).sum())
        t0 = time.time()
        vals = {v: exact_dp(T, D, elig, args.e, n, sL=args.sL, sR=args.sR,
                            depot=n, variant=v) for v in args.variants}
        name = os.path.basename(folder.rstrip("/\\"))
        sname = short_name(folder)
        po = published.get(sname)
        tail = ""
        if po is not None:
            marks = []
            for v in args.variants:
                d = vals[v] - po
                marks.append("=" if abs(d) < 5e-3 else ("BELOW" if d < 0 else "above"))
            tail = f"{po:>9.2f}  " + " ".join(f"{v}:{m}" for v, m in
                                             zip(args.variants, marks))
        print(f"{sname:<8}{n:>3}{ni:>8}" +
              "".join(f"{vals[v]:>12.4f}" for v in args.variants) +
              f"{time.time()-t0:>6.1f}  {tail}")
        rows.append(dict(instance=name, short=sname, n=n, e=args.e, n_inelig=ni,
                         published_opt=po,
                         **{f"exact_{v}": round(vals[v], 6)
                            for v in args.variants}))

    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwritten: {args.out}")

    for v in args.variants:
        a = np.array([r[f"exact_{v}"] for r in rows])
        print(f"  mean exact ({v:8s}) = {a.mean():.4f}")

    if published:
        m = [r for r in rows if r["published_opt"] is not None]
        print(f"\nagainst {len(m)} published optima  (flag_means={args.flag_means})")
        for v in args.variants:
            d = np.array([r[f"exact_{v}"] - r["published_opt"] for r in m])
            eq = int((np.abs(d) < 5e-3).sum())
            lo = int((d < -5e-3).sum())
            hi = int((d > 5e-3).sum())
            print(f"  {v:8s}  equal {eq:>3}/{len(m)}   below {lo:>3}   "
                  f"above {hi:>3}   mean diff {d.mean():+7.4f}")
        print("  'below' is impossible for a correct model: it means the search")
        print("  space is too permissive.  'above' means it is too restrictive.")
        print("  The right settings give equal on every instance.")

    if args.ref:
        try:
            import pandas as pd
        except ImportError:
            return
        ref = pd.read_csv(args.ref)
        got = pd.DataFrame(rows)
        num = [c for c in ref.columns
               if ref[c].dtype.kind in "fi" and c.lower() not in ("e", "n")]
        print(f"\njoining {args.ref} on instance name; numeric columns: {num}")
        objcols = [c for c in ref.columns if ref[c].dtype == object]
        m = None
        for key in objcols or [ref.columns[0]]:
            for mine in ("instance", "short"):
                cand = got.merge(ref, left_on=mine, right_on=key, how="inner")
                if len(cand) and (m is None or len(cand) > len(m)):
                    m = cand
                    print(f"  joined on got.{mine} == ref.{key}")
        if m is None:
            m = got.head(0)
        if not len(m):
            print("  no rows matched -- check the instance-name column")
            return
        print(f"  matched {len(m)} rows")
        for v in args.variants:
            for c in num:
                d = m[f"exact_{v}"] - m[c]
                print(f"  exact_{v:8s} - {c:<16} mean {d.mean():+8.4f}  "
                      f"min {d.min():+8.4f}  max {d.max():+8.4f}  "
                      f"exact_below {int((d < -1e-6).sum())}/{len(m)}")


if __name__ == "__main__":
    main()
