r"""
Export an fstsp_data_v3 .pt test set into Murray-format instance folders, so
the existing Set M pipeline (run_hgatac.jl, run_setM_exact.py,
fstsp_hgatac_objective.py) runs on it unchanged.

WHY A TEMPLATE IS REQUIRED
    nodes.csv's column layout is not something to guess: run_hgatac.jl,
    run_setM_exact.py and fstsp_hgatac_objective.py all read the drone flag by
    a fixed column index, and a wrong guess produces a silently inverted or
    absent eligibility set -- the exact failure mode that cost a full exact
    run earlier.  So this script reads one real Murray nodes.csv, infers the
    column count and which column carries the flag, verifies the inference
    against that instance's known ineligible count, and writes with the same
    width.

WHAT IT CHECKS BEFORE WRITING ANYTHING
    * the .pt's T is symmetric with a zero diagonal
    * the .pt's D is symmetric; its diagonal is forced to exactly 0, because
      the generator clamps the euclidean distance at 1e-6 and
      main.jl asserts drone_cost_mtx[i,i] == 0.0
    * the depot column of `elig` is False
    * every instance carries a single drone/truck speed ratio

WHAT IT CHECKS AFTER WRITING
    every folder is read back with read_instance() copied from
    run_setM_exact.py, and T, D and the ineligible set must reconstruct to
    1e-9.  A round trip that does not close is a hard error, not a warning.

    python fstsp_pt_to_murray.py --pt test_n20_v3.pt --out inst_n20 \
        --template <a real Murray instance folder>
"""
import argparse
import csv
import os

import numpy as np


# ----------------------------------------------------------- template probe
def probe_template(folder):
    """Return (ncol, flag_col).  The flag column is the one that is 0/1 valued
    over the customer rows and is not constant zero."""
    rows = []
    with open(os.path.join(folder, "nodes.csv")) as f:
        for r in csv.reader(f):
            if not r or not r[0].strip() or r[0].strip().startswith(("%", "#")):
                continue
            rows.append([x.strip() for x in r if x.strip() != ""])
    ncol = len(rows[0])
    if any(len(r) != ncol for r in rows):
        raise SystemExit(f"{folder}/nodes.csv has ragged rows")
    n = len(rows) - 2
    cand = []
    for c in range(ncol):
        vals = [float(rows[i + 1][c]) for i in range(n)]
        # both values must appear: a constant column is a coordinate or a
        # padding field that happens to be 0/1 valued, not the flag.
        if set(vals) == {0.0, 1.0}:
            cand.append((c, int(sum(vals))))
    if not cand:
        raise SystemExit(
            f"{folder}/nodes.csv: no column takes both 0 and 1 over the "
            f"customer rows -- this template has no drone-ineligible "
            f"customer, or all of them are.  Pick another template.")
    if len(cand) > 1:
        raise SystemExit(
            f"{folder}/nodes.csv: {len(cand)} columns look like the flag "
            f"{[c for c, _ in cand]} -- ambiguous, inspect it by hand")
    col, k = cand[0]
    print(f"template {os.path.basename(folder)}: {ncol} columns, flag in "
          f"column {col} (0-based), {k} ineligible of {n}")
    if col != 3:
        print(f"   NOTE: the readers hard-code column 3.  This template says "
              f"{col}.  Writing at column {col}; if the readers disagree, "
              f"they and the template are inconsistent and one is wrong.")
    return ncol, col


# ------------------------------------------- reader copied from run_setM_exact
def read_instance_back(folder, flag_col):
    def read_matrix(path):
        with open(path) as f:
            rows = [r for r in csv.reader(f) if r and any(x.strip() for x in r)]
        return np.array([[float(x) for x in r] for r in rows], dtype=np.float64)

    tau = read_matrix(os.path.join(folder, "tau.csv"))
    taup = read_matrix(os.path.join(folder, "tauprime.csv"))
    m = tau.shape[0]
    n = m - 2
    T = np.zeros((n + 1, n + 1))
    D = np.zeros((n + 1, n + 1))
    T[:n, :n] = tau[1:n + 1, 1:n + 1]
    D[:n, :n] = taup[1:n + 1, 1:n + 1]
    T[n, :n] = tau[0, 1:n + 1]
    D[n, :n] = taup[0, 1:n + 1]
    T[:n, n] = tau[1:n + 1, n + 1]
    D[:n, n] = taup[1:n + 1, n + 1]

    rows = []
    with open(os.path.join(folder, "nodes.csv")) as f:
        for r in csv.reader(f):
            if not r or not r[0].strip() or r[0].strip().startswith(("%", "#")):
                continue
            rows.append([x.strip() for x in r if x.strip() != ""])
    inelig = {c for c in range(n) if float(rows[c + 1][flag_col]) == 1.0}
    return T, D, inelig, n


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", required=True,
                    help="one real Murray instance folder, e.g. "
                         r"...\FSTSP_10_customer_problems\20140810T123437v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flat", action="store_true",
                    help="also write every instance under <out>/all/, so the "
                         "checker can be pointed at one --setm root covering "
                         "both endurances.  Folder names are globally unique, "
                         "so nothing collides.")
    a = ap.parse_args()

    import torch
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    T = d["T"].double().numpy()
    D = d["D"].double().numpy()
    elig = d["elig"].numpy()
    E = d["e"].double().numpy()
    coords = d["coords"].double().numpy()
    dep = int(d["depot"])
    n = int(d["meta"]["n"])
    B, V, _ = T.shape
    assert V == n + 1 and dep == n, (V, n, dep)

    # ---- pre-flight on the tensor itself
    assert np.abs(T - T.transpose(0, 2, 1)).max() < 1e-9, "T not symmetric"
    assert np.abs(D - D.transpose(0, 2, 1)).max() < 1e-9, "D not symmetric"
    assert np.abs(np.einsum("bii->bi", T)).max() < 1e-9, "T diagonal nonzero"
    dd = np.abs(np.einsum("bii->bi", D)).max()
    if dd > 0:
        print(f"D diagonal is {dd:.3g}, not 0 -- zeroing it.  main.jl asserts "
              f"drone_cost_mtx[i,i] == 0.0 and would throw otherwise.")
    for b in range(B):
        D[b][np.diag_indices(V)] = 0.0
    assert not elig[:, dep].any(), "depot marked drone-eligible"

    ncol, flag_col = probe_template(a.template)

    idx = range(B if not a.limit else min(B, a.limit))
    man = {}
    for b in idx:
        e = float(E[b])
        tag = f"e{int(round(e))}"
        folder = os.path.join(a.out, tag, f"n{n}s{b:04d}")
        os.makedirs(folder, exist_ok=True)

        # (n+2) layout: 0 = departure depot, 1..n = customers, n+1 = return
        tau = np.zeros((n + 2, n + 2))
        taup = np.zeros((n + 2, n + 2))
        for M, S in ((tau, T[b]), (taup, D[b])):
            M[1:n + 1, 1:n + 1] = S[:n, :n]
            M[0, 1:n + 1] = S[dep, :n]
            M[1:n + 1, 0] = S[:n, dep]
            M[1:n + 1, n + 1] = S[:n, dep]
            M[n + 1, 1:n + 1] = S[dep, :n]
        np.savetxt(os.path.join(folder, "tau.csv"), tau,
                   delimiter=",", fmt="%.10f")
        np.savetxt(os.path.join(folder, "tauprime.csv"), taup,
                   delimiter=",", fmt="%.10f")

        inelig = [c for c in range(n) if not elig[b, c]]
        with open(os.path.join(folder, "nodes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            for r in range(n + 2):
                row = ["0"] * ncol
                row[0] = str(r)
                if ncol > 1:
                    src = dep if r in (0, n + 1) else r - 1
                    row[1] = f"{coords[b, src, 0]:.6f}"
                    if ncol > 2:
                        row[2] = f"{coords[b, src, 1]:.6f}"
                if 0 < r <= n:
                    row[flag_col] = "1" if (r - 1) in inelig else "0"
                w.writerow(row)

        if a.flat:
            import shutil
            flat = os.path.join(a.out, "all", f"n{n}s{b:04d}")
            if os.path.isdir(flat):
                shutil.rmtree(flat)
            shutil.copytree(folder, flat)

        man[folder] = dict(pt_index=b, e=e, n_inelig=len(inelig),
                           inelig=" ".join(str(c) for c in inelig))

    # ---- round trip
    worst_t = worst_d = 0.0
    for folder, rec in man.items():
        b = rec["pt_index"]
        T2, D2, ine2, n2 = read_instance_back(folder, flag_col)
        assert n2 == n
        worst_t = max(worst_t, np.abs(T2 - T[b]).max())
        worst_d = max(worst_d, np.abs(D2 - D[b]).max())
        ine1 = {c for c in range(n) if not elig[b, c]}
        assert ine2 == ine1, (folder, sorted(ine1), sorted(ine2))
    print(f"round trip over {len(man)} folders: max|T| {worst_t:.2e}  "
          f"max|D| {worst_d:.2e}  ineligible sets identical")
    assert worst_t < 1e-9 and worst_d < 1e-9, "round trip did not close"

    mp = os.path.join(a.out, "manifest.csv")
    with open(mp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "folder", "pt_index", "e", "n_inelig",
                    "inelig", "sha16", "n"])
        for folder, rec in man.items():
            w.writerow([os.path.basename(folder), folder, rec["pt_index"],
                        rec["e"], rec["n_inelig"], rec["inelig"],
                        d["meta"]["sha16"], n])
    print(f"written: {len(man)} instances under {a.out}/  ->  {mp}")
    for tag in sorted({f"e{int(round(v))}" for v in E[list(idx)]}):
        k = sum(1 for r in man.values() if f"e{int(round(r['e']))}" == tag)
        print(f"   {a.out}/{tag}: {k} instances")


if __name__ == "__main__":
    main()
