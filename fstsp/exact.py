"""fstsp.exact -- the three cost conventions as fixed-order split DPs +
the exact subset DP.  Merge of fstsp_cost.py + fstsp_exact_variants.py,
verbatim.  The PIPELINE convention (SplitDPTorch, training reward, every
proven Set M number) is variant=\"current\"."""

import itertools

r"""
Three candidate FSTSP cost conventions, as fixed-order split DPs.

They differ only in where the launch service time sL is charged.  Everything
else -- Manhattan truck / Euclidean drone matrices, one customer per sortie,
sR at every recovery, endurance covering the truck leg -- is shared.

  "current"   what fstsp_split.split_dp_murray does today
                truck op : tau_tr
                drone op : sL*[i is not the initial depot] + max(tau_tr, tau_dr) + sR
                endurance: max(tau_tr, tau_dr) + sR <= e

  "paper"     Mahmoudinazlou & Kwon (2024) eq. for C_LL, and the recursion in
              TSPDroneHGATAC.jl/src/dynamic_programming.jl
                truck op : tau_tr                       (no service time)
                drone op : max(tau_tr + sR + sigma_k*sL, tau_dr + sR)
                endurance: tau_tr + sR + sigma_k*sL <= e AND tau_dr + sR <= e

  "mip"       Murray & Chu (2015) constraint (16) read literally: the truck's
              arrival at k carries sigma_k*sL however it got there
                truck op : tau_tr + sigma_k*sL
                drone op : as "paper"
                endurance: as "paper"

sigma_k = 1 iff a drone launch happens at node k.  It is 0 at the final depot.
There is no sL for the first launch out of the depot in any variant, because
Murray fixes t_0 = 0.

"paper" and "mip" are cheaper than or equal to "current": when the drone leg
dominates, the next launch's sL is absorbed into the truck's waiting time
instead of being paid separately.

Signature matches fstsp_split.split_dp_murray so it is a drop-in.
"""

import numpy as np

INF = 1e100
VARIANTS = ("current", "paper", "mip")


def split_dp(T, D, elig, e, order, sL=1.0, sR=1.0, depot=None,
             variant="paper", want_ops=False):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")

    order = [int(x) for x in order]
    N = len(order)
    if depot is None:
        depot = N
    route = [depot] + order + [depot]
    L = len(route)

    pref = np.zeros(L)
    for p in range(L - 1):
        pref[p + 1] = pref[p] + float(T[route[p], route[p + 1]])

    # dp[p][s]: best prefix cost ending at route position p, where s == 1 means
    # a drone launch happens at p.  The sL for that launch, when the convention
    # charges it, is already included -- it belongs to the operation ending at p.
    dp = np.full((L, 2), INF)
    prev = np.full((L, 2), -1, dtype=np.int32)
    prev_s = np.full((L, 2), -1, dtype=np.int32)
    used = np.full((L, 2), -1, dtype=np.int32)

    dp[0, 0] = dp[0, 1] = 0.0

    for k in range(1, L):
        for s in (0, 1):
            if k == L - 1 and s == 1:
                continue                       # nothing launches at the end
            slk = float(sL) if s == 1 else 0.0

            for i in range(0, k):
                # ---- truck-only operation i -> k --------------------------
                full = pref[k] - pref[i]
                if variant == "current":
                    cand = dp[i, 0] + full
                elif variant == "paper":
                    cand = dp[i, 0] + full
                else:                          # mip
                    cand = dp[i, 0] + full + slk
                if cand < dp[k, s] - 1e-12:
                    dp[k, s], prev[k, s], prev_s[k, s], used[k, s] = cand, i, 0, -1

                # ---- drone operation i -> k ------------------------------
                if k - i < 2:
                    continue
                inode, knode = route[i], route[k]
                if variant == "current":
                    launch_s = 0.0 if (i == 0 and inode == depot) else float(sL)

                for jpos in range(i + 1, k):
                    jnode = route[jpos]
                    if not bool(elig[jnode]):
                        continue
                    tr = (full
                          - float(T[route[jpos - 1], jnode])
                          - float(T[jnode, route[jpos + 1]])
                          + float(T[route[jpos - 1], route[jpos + 1]]))
                    dr = float(D[inode, jnode] + D[jnode, knode])

                    if variant == "current":
                        if max(tr, dr) + sR > e + 1e-9:
                            continue
                        op = launch_s + max(tr, dr) + sR
                        base = dp[i, 1]
                    else:
                        if tr + sR + slk > e + 1e-9 or dr + sR > e + 1e-9:
                            continue
                        op = max(tr + sR + slk, dr + sR)
                        base = dp[i, 1]

                    cand = base + op
                    if cand < dp[k, s] - 1e-12:
                        dp[k, s] = cand
                        prev[k, s], prev_s[k, s], used[k, s] = i, 1, jpos

    best = float(dp[L - 1, 0])
    if not want_ops:
        return best

    ops, cur, s = [], L - 1, 0
    while cur > 0:
        p, ps, j = int(prev[cur, s]), int(prev_s[cur, s]), int(used[cur, s])
        ops.append((p, cur, j))
        cur, s = p, ps
    ops.reverse()
    return best, ops


def split_dp_current(T, D, elig, e, order, **kw):
    return split_dp(T, D, elig, e, order, variant="current", **kw)


def split_dp_paper(T, D, elig, e, order, **kw):
    return split_dp(T, D, elig, e, order, variant="paper", **kw)


def split_dp_mip(T, D, elig, e, order, **kw):
    return split_dp(T, D, elig, e, order, variant="mip", **kw)


# ===========================================================================
# subset-DP exact solver (was fstsp_exact_variants.py)
# ===========================================================================
r"""
Exact FSTSP optimum under any of the three cost conventions in fstsp_cost.py.

The subset DP carries one extra bit of state: sigma, whether a drone launch
happens at the current node.  The "paper" and "mip" conventions charge the
launch service sL at the RENDEZVOUS node of the previous operation, inside the
max, so the operation cost depends on what the NEXT operation does.  The bit
makes that Markovian.

  F[mask][i][sigma] = best cost with `mask` served, truck at node i, and the
                      operation starting at i being a drone sortie iff sigma=1.
                      Any sL owed for that launch is already included.

Complexity O(3^n * n^2 * 2).  n=10 is a few seconds per instance.
"""




INF = 1e100


def _held_karp_paths(T, N):
    NP1, full = N + 1, 1 << N
    hk = np.full((NP1, full, NP1), INF)
    for i in range(NP1):
        hi = hk[i]
        hi[0, :] = T[i, :]
        if i < N:
            hi[0, i] = INF
        for mask in range(1, full):
            if i < N and (mask >> i) & 1:
                continue
            ms = [m for m in range(N) if (mask >> m) & 1]
            base = np.array([hi[mask ^ (1 << m), m] for m in ms])
            row = (base[:, None] + T[ms, :]).min(axis=0)
            row[ms] = INF
            hi[mask] = row
    return hk


def _op_tables(T, D, elig, e, sL, sR, N, hk, variant):
    """op[sigma][i][M][k]: best sortie cost, INF if infeasible."""
    NP1, full = N + 1, 1 << N
    op = np.full((2, NP1, full, NP1), INF)
    for sg in (0, 1):
        slk = sL if sg == 1 else 0.0
        for i in range(NP1):
            for M in range(1, full):
                if i < N and (M >> i) & 1:
                    continue
                best = None
                for j in range(N):
                    if not ((M >> j) & 1) or not elig[j]:
                        continue
                    tr = hk[i, M ^ (1 << j)]
                    dr = D[i, j] + D[j, :]
                    if variant == "current":
                        m = np.maximum(tr, dr)
                        ok = (m + sR <= e + 1e-9) & (tr < INF)
                        cost = np.where(ok, m + sR, INF)
                    else:
                        ok = ((tr + sR + slk <= e + 1e-9)
                              & (dr + sR <= e + 1e-9) & (tr < INF))
                        cost = np.where(ok, np.maximum(tr + sR + slk, dr + sR),
                                        INF)
                    best = cost if best is None else np.minimum(best, cost)
                if best is not None:
                    best[[m for m in range(N) if (M >> m) & 1]] = INF
                    op[sg, i, M] = best
    return op


def exact_dp(T, D, elig, e, N, sL=1.0, sR=1.0, depot=None, variant="paper",
             want_order=False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if depot is None:
        depot = N
    T = np.asarray(T, float)
    D = np.asarray(D, float)
    elig = np.asarray(elig, bool)
    NP1, full, FULL = N + 1, 1 << N, (1 << N) - 1

    hk = _held_karp_paths(T, N)
    op = _op_tables(T, D, elig, e, sL, sR, N, hk, variant)

    F = np.full((full, NP1, 2), INF)
    F[0, depot, 0] = F[0, depot, 1] = 0.0
    par = {}

    for mask in range(full):
        comp = FULL ^ mask
        cand_i = [depot] if mask == 0 else [m for m in range(N) if (mask >> m) & 1]
        for i in cand_i:
            # ---- sigma = 0 : truck moves alone -------------------------
            base = F[mask, i, 0]
            if base < INF:
                for j in range(N):
                    if (mask >> j) & 1:
                        continue
                    nm = mask | (1 << j)
                    for sg in (0, 1):
                        extra = sL if (variant == "mip" and sg == 1) else 0.0
                        v = base + T[i, j] + extra
                        if v < F[nm, j, sg] - 1e-12:
                            F[nm, j, sg] = v
                            if want_order:
                                par[(nm, j, sg)] = (mask, i, 0, 0, "truck")
            # ---- sigma = 1 : a sortie launches here --------------------
            base = F[mask, i, 1]
            if base < INF and comp:
                # "current" charges sL at the launch node, outside the max, and
                # waives it only for the very first launch out of the depot.
                if variant == "current" and not (mask == 0 and i == depot):
                    base = base + sL
                sub = comp
                while sub:
                    nm_base = mask | sub
                    for sg in (0, 1):
                        row = op[sg, i, sub]
                        for k in range(N):
                            if (nm_base >> k) & 1 or row[k] >= INF:
                                continue
                            nm = nm_base | (1 << k)
                            v = base + row[k]
                            if v < F[nm, k, sg] - 1e-12:
                                F[nm, k, sg] = v
                                if want_order:
                                    par[(nm, k, sg)] = (mask, i, 1, sub, "drone")
                    sub = (sub - 1) & comp

    best, bstate = INF, None
    for i in range(NP1):
        if F[FULL, i, 0] < INF:
            v = F[FULL, i, 0] + T[i, depot]
            if v < best:
                best, bstate = v, (FULL, i, 0, 0, "truck_home")
    for mask in range(full):
        comp = FULL ^ mask
        if not comp:
            continue
        cand_i = [depot] if mask == 0 else [m for m in range(N) if (mask >> m) & 1]
        for i in cand_i:
            if F[mask, i, 1] >= INF:
                continue
            c = op[0, i, comp, depot]
            if c >= INF:
                continue
            v = F[mask, i, 1] + c
            if variant == "current" and not (mask == 0 and i == depot):
                v += sL
            if v < best:
                best, bstate = v, (mask, i, 1, comp, "drone_home")

    if not want_order:
        return float(best)
    return float(best), _reconstruct(par, bstate, T, D, elig, e, N, depot, hk)


def _reconstruct(par, bstate, T, D, elig, e, N, depot, hk):
    segs = []
    mask, i, sg, M, kind = bstate
    if kind == "drone_home":
        segs.append((i, M, depot))
    cur = (mask, i, sg)
    while cur[:2] != (0, depot):
        pmask, pi, psg, pM, pkind = par[cur]
        segs.append((pi, pM if pkind == "drone" else 0, cur[1]))
        cur = (pmask, pi, psg)
    segs.reverse()

    order = []
    for (a, M, b) in segs:
        if M:
            order.extend(_inner(T, D, elig, a, M, b, N, hk))
        if b != depot:
            order.append(b)
    return order


def _inner(T, D, elig, i, M, k, N, hk):
    best, bj, bR = INF, None, 0
    for j in range(N):
        if not ((M >> j) & 1) or not elig[j]:
            continue
        R = M ^ (1 << j)
        tr = hk[i, R, k]
        if tr >= INF:
            continue
        m = max(tr, D[i, j] + D[j, k])
        if m < best:
            best, bj, bR = m, j, R
    if bj is None:
        return []
    path, cm, ce = [], bR, k
    while cm:
        for m in range(N):
            if ((cm >> m) & 1) and abs(hk[i, cm ^ (1 << m), m] + T[m, ce]
                                       - hk[i, cm, ce]) < 1e-9:
                path.append(m)
                cm ^= (1 << m)
                ce = m
                break
        else:
            break
    path.reverse()
    return [bj] + path


def exact_perm(T, D, elig, e, N, sL=1.0, sR=1.0, depot=None, variant="paper"):
    if depot is None:
        depot = N
    best, bp = INF, None
    for p in itertools.permutations(range(N)):
        c = split_dp(T, D, elig, e, p, sL=sL, sR=sR, depot=depot, variant=variant)
        if c < best:
            best, bp = c, p
    return float(best), bp
