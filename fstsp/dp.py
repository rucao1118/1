"""fstsp.dp -- SplitDPTorch (reward engine) + SplitDPTorchV4 (lookahead,
beam row ops).  Merge of fstsp_dp_torch.py + fstsp_dp_v4.py, verbatim."""

"""
Vectorized incremental split-DP (strict Murray convention), on the GPU.

This replaces the ~2560 pure-python calls to split_dp_murray that used to sit
in the reward path.  Three things come out of it that the python version could
not give:

  1. speed.  The whole batch x group is one tensor program; the triple loop
     over (i, jpos) becomes a [M, t, t] reduce per appended node.  n=20 has an
     O(L^3) DP that is ~8x the n=10 work -- unaffordable in python, free here.

  2. dp[t] at decision time.  Verified identity (see verify_dp.py):

         dp[t] == optimal split cost of the OPEN route [depot, r_1..r_t]

     i.e. dp[t] is a function of the PREFIX only.  So C - dp[t] is an exact,
     unbiased, strictly-lower-variance per-step return: subtracting any
     function of the state leaves E[grad] untouched.  That is reward-to-go,
     and it exists here only because the reward is a DP over the prefix.

  3. drone labels, by backtracking prev/used, without a second pass.

Conventions (strict Murray & Chu 2015; verify_dp.py checks them against an
independent reference implementation):
    customers 0..N-1, depot = N, route = [depot] + order + [depot]
    dp[k] = min over i<k of dp[i] + op(i,k)
    op(i,k) = min( pref[k]-pref[i],
                   min over jpos in (i,k) with elig[route[jpos]] and
                       span + sR <= e   of   launch(i) + span + sR )
      span       = max(truck_path, drone_path)
      truck_path = (pref[k]-pref[i]) - skip[jpos]
      skip[p]    = T[r_{p-1}][r_p] + T[r_p][r_{p+1}] - T[r_{p-1}][r_{p+1}]
      drone_path = D[r_i][r_jpos] + D[r_jpos][r_k]
      launch(i)  = 0 if i == 0 (depot launch) else sL

Note skip[p] needs r_{p+1}, so it only becomes known when position p+1 is
appended -- which is exactly when it is first needed, since an operation
ending at t can fly jpos <= t-1.  That is what makes the DP incremental.
"""

import torch

INF = 1e30
TOL = 1e-9
DP_DIM = 6          # must match fstsp_model_v3.DP_DIM
STATE_CAP = 5.0


class SplitDPTorch:
    """
    Incremental, batched.  Start with the depot at position 0, append one node
    at a time, read self.dp[:, t] at any point.

        eng = SplitDPTorch(T, D, elig, e, sL, sR, depot, Lmax=N+2, typ=typ)
        for a in actions:      # N customers
            eng.append(a)
        eng.append(depot_col)
        cost = eng.dp[:, N+1]
    """

    def __init__(self, T, D, elig, e, sL, sR, depot, Lmax, typ=None, tol=TOL):
        M, V, _ = T.shape
        dev, dt = T.device, T.dtype
        self.M, self.V, self.Lmax = M, V, int(Lmax)
        self.sL, self.sR = float(sL), float(sR)
        self.depot, self.tol = int(depot), float(tol)
        self.dev, self.dt = dev, dt

        self.Tf = T.reshape(M, V * V)
        self.Df = D.reshape(M, V * V)
        self.elig = elig
        self.e = e.reshape(M)
        self.typ = torch.ones(M, device=dev, dtype=dt) if typ is None \
            else typ.reshape(M).to(dt)

        self.node = torch.full((M, self.Lmax), int(depot), dtype=torch.long, device=dev)
        self.dp = torch.full((M, self.Lmax), INF, device=dev, dtype=dt)
        self.dp[:, 0] = 0.0
        self.pref = torch.zeros(M, self.Lmax, device=dev, dtype=dt)
        self.prev = torch.full((M, self.Lmax), -1, dtype=torch.long, device=dev)
        self.used = torch.full((M, self.Lmax), -1, dtype=torch.long, device=dev)
        self.skip = torch.zeros(M, self.Lmax, device=dev, dtype=dt)
        self.t = 0

    # -- helpers ------------------------------------------------------------
    def _tg(self, i, j):
        """T[m, i[m], j[m]] -> [M]"""
        return self.Tf.gather(1, (i * self.V + j).reshape(self.M, 1)).reshape(self.M)

    # -- main ---------------------------------------------------------------
    @torch.no_grad()
    def append(self, a):
        M, V = self.M, self.V
        a = a.reshape(M).long()
        t = self.t + 1
        assert t < self.Lmax, "SplitDPTorch: Lmax too small"
        self.node[:, t] = a

        n1 = self.node[:, t - 1]
        self.pref[:, t] = self.pref[:, t - 1] + self._tg(n1, a)

        if t >= 2:
            n2 = self.node[:, t - 2]
            self.skip[:, t - 1] = self._tg(n2, n1) + self._tg(n1, a) - self._tg(n2, a)

        # ---- op(i, t) for every i < t ------------------------------------
        ft = self.pref[:, t].unsqueeze(1) - self.pref[:, :t]           # [M,t]
        best = ft.clone()
        bestj = torch.full((M, t), -1, dtype=torch.long, device=self.dev)

        if t >= 2:
            J = t - 1
            sk = self.skip[:, 1:t]                                     # [M,J]
            jn = self.node[:, 1:t]                                     # [M,J]
            ni = self.node[:, :t]                                      # [M,t]

            truck = ft.unsqueeze(2) - sk.unsqueeze(1)                  # [M,t,J]
            idx1 = (ni.unsqueeze(2) * V + jn.unsqueeze(1)).reshape(M, t * J)
            d1 = self.Df.gather(1, idx1).reshape(M, t, J)
            d2 = self.Df.gather(1, jn * V + a.unsqueeze(1))            # [M,J]
            drone = d1 + d2.unsqueeze(1)

            span = torch.maximum(truck, drone)
            okj = self.elig.gather(1, jn)                              # [M,J] bool
            ok = okj.unsqueeze(1) & (span + self.sR
                                     <= self.e.reshape(M, 1, 1) + self.tol)
            ai = torch.arange(t, device=self.dev).reshape(1, t, 1)
            ap = torch.arange(J, device=self.dev).reshape(1, 1, J)
            ok = ok & (ap >= ai)                                       # jpos > i

            launch = torch.where(torch.arange(t, device=self.dev) == 0,
                                 torch.zeros((), device=self.dev, dtype=self.dt),
                                 torch.full((), self.sL, device=self.dev, dtype=self.dt))
            op = launch.reshape(1, t, 1) + span + self.sR
            op = torch.where(ok, op, torch.full_like(op, INF))

            mj, aj = op.min(dim=2)                                     # [M,t]
            better = mj < best - 1e-12
            best = torch.where(better, mj, best)
            bestj = torch.where(better, aj + 1, bestj)

        cand = self.dp[:, :t] + best
        mv, mi = cand.min(dim=1)
        self.dp[:, t] = mv
        self.prev[:, t] = mi
        self.used[:, t] = bestj.gather(1, mi.unsqueeze(1)).reshape(M)
        self.t = t
        return self

    # -- readouts -----------------------------------------------------------
    @torch.no_grad()
    def state_feats(self):
        """
        [M, DP_DIM] describing the prefix that has already been committed:
          0  dp[t]                 cost of the best split of the prefix
          1  dp[t] - dp[t-1]       marginal cost of the node just placed
          2  pref[t] - dp[t]       cumulative saving the drone has bought
          3  1[the op that closed at t flew somebody]
          4  (t - prev[t]) / L     how long that operation was
          5  t / L                 progress
        0..2 are divided by the instance time scale and clipped.
        """
        M, t = self.M, self.t
        L = float(max(self.Lmax - 1, 1))
        typ = self.typ.clamp_min(1e-6)
        dpt = self.dp[:, t]
        prv = self.prev[:, t]

        if t >= 1:
            marg = dpt - self.dp[:, t - 1]
            flew = (self.used[:, t] >= 0).to(self.dt)
            oplen = (t - prv).clamp_min(0).to(self.dt) / L
        else:
            z = torch.zeros(M, device=self.dev, dtype=self.dt)
            marg, flew, oplen = z, z, z

        sav = self.pref[:, t] - dpt
        c = STATE_CAP
        return torch.stack([
            (dpt / typ).clamp(-c, c),
            (marg / typ).clamp(-c, c),
            (sav / typ).clamp(-c, c),
            flew,
            oplen,
            torch.full((M,), t / L, device=self.dev, dtype=self.dt),
        ], dim=-1)

    @torch.no_grad()
    def backtrack_labels(self, N):
        """[M, N] float, 1 where customer c is served by the drone."""
        M = self.M
        lab = torch.zeros(M, N, device=self.dev, dtype=self.dt)
        cur = torch.full((M,), self.t, dtype=torch.long, device=self.dev)
        alive = torch.ones(M, dtype=torch.bool, device=self.dev)
        for _ in range(self.Lmax):
            p = self.prev.gather(1, cur.unsqueeze(1)).reshape(M)
            j = self.used.gather(1, cur.unsqueeze(1)).reshape(M)
            nj = self.node.gather(1, j.clamp_min(0).unsqueeze(1)).reshape(M)
            hit = alive & (j >= 0) & (nj < N)
            lab.scatter_add_(1, nj.clamp(0, N - 1).unsqueeze(1),
                             hit.to(self.dt).unsqueeze(1))
            cur = torch.where(alive & (cur > 0), p, cur)
            alive = alive & (cur > 0)
            if not bool(alive.any()):
                break
        return lab.clamp(max=1.0)

    @torch.no_grad()
    def sortie_count(self, N):
        # used[k] is only meaningful for positions on the backtracked optimal
        # path; every other entry is stale.  Each sortie flies exactly one
        # customer, so the label sum is the count.
        return self.backtrack_labels(N).sum(1)


# ==========================================================================
# v4 read-only additions (was fstsp_dp_v4.py)
# ==========================================================================
"""
v4 split-DP.  ONE addition to the v3 engine, and it is READ-ONLY.

fstsp_dp_torch.SplitDPTorch is the reward.  It has been checked against an
independent triple-loop reference (verify_dp.py checks 1-3) and it produced
every proven-optimal number in this project.  So v4 does not touch it: it
subclasses it and adds two methods that only READ the committed state.

    lookahead(window)   exact dp[t+1] for EVERY candidate next node c
    select_rows(idx)    a new engine holding a gathered subset/superset of rows

WHY lookahead MATTERS

    The v3 decoder saw a hand-rolled one-step approximation of the DP
    (candidate_feats channels 5-7): "if I fly the node I am standing on and
    make c the rendezvous, what does that one operation cost".  That is one
    term of one row of the DP.  The real question the decoder is being asked
    is

        if I append c, what does dp[t+1] become

    and that is computable in closed form from state the engine already holds,
    because dp[t+1] = min_{i<=t} dp[i] + op(i, t+1) and everything on the
    right except the two terms involving c is already committed.  Cost is
    O(W^2 V) per step instead of the O(t^2 V) full version, where W caps how
    far back the last operation may have launched.  W = t+1 is exact; the
    default W = 4 is exact for every operation that spans at most 4 truck
    positions, which at Set M endurance is essentially all of them (an
    operation that spans more has span >= 4 truck legs and fails
    span + sR <= e long before it is optimal).

    dp[t] is NON-DECREASING in t, because op(i,k) >= 0 always.  So dp[t] is a
    valid LOWER BOUND on the cost of every completion of the prefix.  That is
    what makes it legitimate to rank partial solutions by it -- see
    fstsp_rollout_v4.beam_decode.

WHY select_rows MATTERS
    Beam search reorders, duplicates and drops partial solutions every step.
    All the mutable state is [M, Lmax], so a row gather is the whole job, and
    doing it this way means the beam re-uses the SAME append() that the reward
    uses.  No second implementation of the DP, nothing to keep in sync.
"""



LOOK_CAP = 3.0          # marginal-cost channels are clipped to +-LOOK_CAP


class SplitDPTorchV4(SplitDPTorch):

    # ------------------------------------------------------------------ rows
    @torch.no_grad()
    def select_rows(self, idx):
        """
        New engine whose rows are self's rows at `idx` ([M2] long).  idx may
        repeat rows (beam expansion) or drop them (beam pruning).  Everything
        is copied, so the parent engine is untouched.
        """
        idx = idx.reshape(-1).long()
        M2 = idx.shape[0]
        out = object.__new__(type(self))
        out.M, out.V, out.Lmax = M2, self.V, self.Lmax
        out.sL, out.sR = self.sL, self.sR
        out.depot, out.tol = self.depot, self.tol
        out.dev, out.dt = self.dev, self.dt
        out.t = self.t

        out.Tf = self.Tf[idx]
        out.Df = self.Df[idx]
        out.elig = self.elig[idx]
        out.e = self.e[idx]
        out.typ = self.typ[idx]

        out.node = self.node[idx].clone()
        out.dp = self.dp[idx].clone()
        out.pref = self.pref[idx].clone()
        out.prev = self.prev[idx].clone()
        out.used = self.used[idx].clone()
        out.skip = self.skip[idx].clone()
        return out

    # ------------------------------------------------------------- lookahead
    @torch.no_grad()
    def lookahead(self, window=0):
        """
        Returns (dpn, flew):
            dpn  [M, V] float   dp[t+1] if node c were appended next
            flew [M, V] float   1 where the operation that would close at c
                                flies somebody (i.e. c is a rendezvous)

        window <= 0 means exact (no truncation).  Entries for nodes that are
        already on the route are meaningless; the caller masks them.
        """
        M, V, t = self.M, self.V, self.t
        dev, dt = self.dev, self.dt
        ar = torch.arange(V, device=dev)

        W = int(window) if window and window > 0 else (t + 1)

        # ---- launch positions i we are willing to consider -----------------
        # the last W committed positions, and always position 0, because a
        # depot launch is free of sL and is never dominated for that reason.
        ipos = list(range(max(0, t - W + 1), t + 1))
        if ipos[0] != 0:
            ipos = [0] + ipos
        ip = torch.tensor(ipos, device=dev, dtype=torch.long)
        Wi = ip.shape[0]

        rt = self.node[:, t]                                        # [M]
        Trow = self.Tf.gather(1, rt.view(M, 1) * V + ar.view(1, V))  # [M,V] T[r_t,c]
        pref_new = self.pref[:, t].view(M, 1) + Trow                 # [M,V]

        ft = pref_new.unsqueeze(1) - self.pref[:, ip].unsqueeze(2)   # [M,Wi,V]
        best = ft.clone()
        flew = torch.zeros_like(best, dtype=torch.bool)

        # ---- fly branch ---------------------------------------------------
        jl = list(range(max(1, t - W + 1), t + 1))                   # jpos <= t
        if jl:
            jp = torch.tensor(jl, device=dev, dtype=torch.long)
            Wj = jp.shape[0]
            jn = self.node[:, jp]                                    # [M,Wj]
            inode = self.node[:, ip]                                 # [M,Wi]

            # skip'[jpos].  Known for jpos < t; for jpos == t it only exists
            # once c is chosen, which is exactly the term c enters through.
            sk = self.skip[:, jp].unsqueeze(2).expand(M, Wj, V).clone()
            rtm1 = self.node[:, t - 1]
            a1 = self.Tf.gather(1, (rtm1 * V + rt).view(M, 1))        # T[r_{t-1},r_t]
            a3 = self.Tf.gather(1, rtm1.view(M, 1) * V + ar.view(1, V))
            sk[:, jl.index(t), :] = a1 + Trow - a3

            truck = ft.unsqueeze(2) - sk.unsqueeze(1)                 # [M,Wi,Wj,V]

            d1 = self.Df.gather(
                1, (inode.unsqueeze(2) * V + jn.unsqueeze(1)).reshape(M, Wi * Wj)
            ).reshape(M, Wi, Wj)                                      # D[r_i,r_j]
            d2 = self.Df.gather(
                1, (jn.view(M, Wj, 1) * V + ar.view(1, 1, V)).reshape(M, Wj * V)
            ).reshape(M, Wj, V)                                       # D[r_j,c]
            drone = d1.unsqueeze(3) + d2.unsqueeze(1)                 # [M,Wi,Wj,V]

            span = torch.maximum(truck, drone)
            okj = self.elig.gather(1, jn)                             # [M,Wj]
            ok = okj.view(M, 1, Wj, 1) & \
                (span + self.sR <= self.e.view(M, 1, 1, 1) + self.tol)
            ok = ok & (jp.view(1, 1, Wj, 1) > ip.view(1, Wi, 1, 1))   # jpos > i

            launch = torch.where(
                ip == 0,
                torch.zeros((), device=dev, dtype=dt),
                torch.full((), self.sL, device=dev, dtype=dt))
            op = launch.view(1, Wi, 1, 1) + span + self.sR
            op = torch.where(ok, op, torch.full_like(op, INF))

            fly = op.min(dim=2).values                                # [M,Wi,V]
            better = fly < best - 1e-12
            best = torch.where(better, fly, best)
            flew = better

        cand = self.dp[:, ip].unsqueeze(2) + best                     # [M,Wi,V]
        dpn, ai = cand.min(dim=1)                                     # [M,V]
        fl = flew.gather(1, ai.unsqueeze(1)).squeeze(1).to(dt)
        return dpn, fl

    # ----------------------------------------------------------- feature use
    @torch.no_grad()
    def lookahead_feats(self, selectable, window=0, cap=LOOK_CAP):
        """
        [M, V, 3] ready to concatenate onto the candidate features:
          0  -(dp[t+1](c) - dp[t]) / typ        exact marginal cost of c
          1   1[the closing operation flies]
          2  -(dp[t+1](c) - min over legal c) / typ    scale-free regret

        Channel 2 is not a monotone re-encoding of channel 0 as far as the
        per-candidate MLP is concerned: the MLP sees one candidate at a time
        and cannot compute a min across candidates, so handing it the gap to
        the best legal option is genuinely new information.
        """
        dpn, fl = self.lookahead(window=window)
        typ = self.typ.clamp_min(1e-6).unsqueeze(1)
        marg = (dpn - self.dp[:, self.t].unsqueeze(1)) / typ
        big = torch.full_like(dpn, float("inf"))
        legal_min = torch.where(selectable, dpn, big).min(1, keepdim=True).values
        regret = (dpn - legal_min) / typ
        return torch.stack([
            (-marg).clamp(-cap, cap),
            fl,
            (-regret).clamp(-cap, cap),
        ], dim=-1)
