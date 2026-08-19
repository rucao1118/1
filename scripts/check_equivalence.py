#!/usr/bin/env python3
"""
THE GATE.  Run this ONCE on the machine that has BOTH the old flat files and
the new fstsp/ package, BEFORE switching anything over:

    cd <repo root containing the old fstsp_*.py files>
    python <new>/scripts/check_equivalence.py --new_dir <new> \
        --ckpt runs/n50_lead_inf/best.pt --device cuda

It asserts BIT equality (torch.equal / max|diff| == 0) between the old
monkey-patched stack and the new package on:

  A  data        generate_batch(center) == generate_batch_v4;
                 generate_batch(murray3/mix4) == generate_batch_depot
  B  advantage   fstsp.losses.advantage == leader(advfix(train_v4.advantage))
                 over rep in {1,2} x adv_scale {pair,group} x alpha {0,10,inf}
  C  rollout     greedy + sampled orders/costs/logp identical under one seed
  D  train_step  one full loss identical under one seed
  E  checkpoint  the live best.pt loads strict into the new model
  F  construct   fstsp.evaluate construct == fstsp_post_v4.construct

Any FAIL means DO NOT switch; send me the section name and the printed diff.
"""

import argparse
import math
import os
import sys

import torch


def eq(name, a, b):
    if torch.is_tensor(a):
        same = torch.equal(a, b)
        d = float((a.float() - b.float()).abs().max()) if a.shape == b.shape \
            else float("nan")
    else:
        same, d = a == b, abs(float(a) - float(b))
    print(f"  {'PASS' if same else 'FAIL':4s}  {name}   max|diff|={d:.3e}")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_dir", default=".",
                    help="directory that CONTAINS the fstsp/ package")
    ap.add_argument("--old_dir", default=".",
                    help="directory with the old flat fstsp_*.py files")
    ap.add_argument("--ckpt", default="", help="a v4 best.pt (e.g. the live "
                    "leader best) for sections C-F")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.old_dir))
    sys.path.insert(0, os.path.abspath(args.new_dir))
    dev = args.device
    fails = 0

    # ---- A: data -----------------------------------------------------------
    print("\nA) data streams")
    import fstsp_data_v4 as OD4
    import fstsp_data_depot as ODD
    from fstsp.data import generate_batch as ng

    for n, B, seed, em in ((20, 32, 12345, "strat"), (50, 16, 777, "rate_cap"),
                           (10, 24, 31, "strat")):
        a = OD4.generate_batch_v4(B, n_customer=n, device="cpu", seed=seed,
                                  balanced=True, elig_mode=em)
        b = ng(B, n_customer=n, device="cpu", seed=seed, balanced=True,
               depot_mode="center", elig_mode=em)
        b.pop("depot_mode_idx", None)
        for k in ("coords", "feat", "edge_feat", "T", "D", "elig", "e"):
            fails += not eq(f"center n={n} seed={seed} {k}", a[k], b[k])
        for m in ("murray3", "mix4"):
            a2 = ODD.generate_batch_depot(B, n_customer=n, device="cpu",
                                          seed=seed, balanced=True,
                                          depot_mode=m, elig_mode=em)
            b2 = ng(B, n_customer=n, device="cpu", seed=seed, balanced=True,
                    depot_mode=m, elig_mode=em)
            for k in ("coords", "T", "elig", "depot_mode_idx"):
                fails += not eq(f"{m} n={n} {k}", a2[k], b2[k])

    # ---- B: advantage chain -------------------------------------------------
    print("\nB) advantage  (leader o advfix o base)")
    import fstsp_train_v4 as TV4
    import fstsp_leader as LEAD          # importing installs advfix underneath
    LEAD.install()                       # leader on top -> the full live chain
    from fstsp.losses import advantage as new_adv

    class A:
        rtg, adv_norm = 0, 1
        adv_scale, leader_alpha = "group", 0.0

    torch.manual_seed(0)
    for (K, rep) in ((50, 1), (20, 1), (25, 2)):
        S = K * rep
        C = 140.0 + 12.0 * torch.randn(64, S)
        dp = torch.zeros(64, S, K)
        for scale in ("pair", "group"):
            for al in (0.0, 10.0, float("inf")):
                a = A()
                a.adv_scale, a.leader_alpha = scale, al
                o = TV4.advantage(C, dp, K, rep, a)
                nw = new_adv(C, dp, K, rep, a)
                tag = f"K={K} rep={rep} scale={scale} alpha={al}"
                fails += not eq(tag, o, nw)

    if not args.ckpt:
        print("\n(no --ckpt: sections C-F skipped; pass the live best.pt to "
              "run them)")
        _verdict(fails)
        return

    # ---- C-F need a checkpoint ---------------------------------------------
    print("\nC) rollout")
    import fstsp_rollout_v4 as OR
    import fstsp_model_v4 as OM
    import fstsp_post_v4 as OP
    from fstsp.rollout import rollout_v4 as nroll
    from fstsp.evaluate import build_model, load_data  # noqa: F401
    from fstsp.evaluate import construct as ncon

    model_new, ck, cargs, ver, infl = build_model(args.ckpt, dev)
    look_w = int(cargs.get("look_w", 0))
    model_old = OM.FSTSPv4(in_dim=12, edge_dim=5, dim=int(cargs.get("dim", 128)),
                           heads=int(cargs.get("heads", 8)),
                           layers=int(cargs.get("layers", 4)),
                           ff=int(cargs.get("ff", 512)),
                           use_edge_bias=bool(cargs.get("use_edge_bias", 1)),
                           use_cand_bias=bool(cargs.get("use_cand_bias", 1)),
                           use_dp_state=bool(cargs.get("use_dp_state", 1)),
                           n_dec=int(cargs.get("n_dec", 1)),
                           prenorm=bool(cargs.get("prenorm", 0))).to(dev)
    model_old.load_state_dict(ck["model"])
    model_old.eval()
    model_new.eval()

    n = int(cargs.get("n", 50))
    batch = ng(8, n_customer=n, device=dev, seed=424242, balanced=True,
               depot_mode=str(cargs.get("depot_mode", "center")) or "center",
               elig_mode="rate_cap")
    batch.pop("depot_mode_idx", None)

    with torch.no_grad():
        ro = OR.rollout_v4(model_old, batch, greedy=True, look_w=look_w,
                           want_labels=False)
        rn = nroll(model_new, batch, greedy=True, look_w=look_w,
                   want_labels=False)
    fails += not eq("greedy orders", ro["orders"], rn["orders"])
    fails += not eq("greedy cost", ro["cost"], rn["cost"])

    torch.manual_seed(9)
    with torch.no_grad():
        rs_o = OR.rollout_v4(model_old, batch, greedy=False, look_w=look_w,
                             want_labels=True)
    torch.manual_seed(9)
    with torch.no_grad():
        rs_n = nroll(model_new, batch, greedy=False, look_w=look_w,
                     want_labels=True)
    fails += not eq("sampled orders", rs_o["orders"], rs_n["orders"])
    fails += not eq("sampled logp", rs_o["logp_steps"], rs_n["logp_steps"])
    fails += not eq("sampled labels", rs_o["labels"], rs_n["labels"])

    # ---- D: one full train_step --------------------------------------------
    print("\nD) train_step loss (leader chain live on the old side)")
    from fstsp.train import train_step as nstep
    import argparse as _ap
    afields = dict(vars(_ap.Namespace(**TV4.CFG)))
    afields.update(cargs)
    afields.update(dict(n=n, sil=0, n_dec=int(cargs.get("n_dec", 1))))
    for al in (0.0, float("inf")):
        afields["leader_alpha"] = al
        a_ns = _ap.Namespace(**afields)
        torch.manual_seed(5)
        lo, _ = TV4.train_step(model_old, batch, a_ns, ep=3)
        torch.manual_seed(5)
        ln, _ = nstep(model_new, batch, a_ns, ep=3)
        fails += not eq(f"loss alpha={al}", lo.detach(), ln.detach())

    # ---- E: strict checkpoint load -----------------------------------------
    print("\nE) checkpoint")
    print(f"  ver={ver} inflated={infl} (must be ver=4, inflated=False)")
    fails += not eq("ver==4", float(ver), 4.0)
    fails += not eq("not inflated", float(0 if not infl else 1), 0.0)

    # ---- F: construct parity ------------------------------------------------
    print("\nF) evaluate.construct vs fstsp_post_v4.construct (rep=2)")
    with torch.no_grad():
        Co, Oo = OP.construct(model_old, batch, look_w,
                              str(cargs.get("start_mode", "pomo")), 2, dev, 8)
        Cn, On = ncon(model_new, batch, look_w,
                      str(cargs.get("start_mode", "pomo")), 2, dev, 8)
    fails += not eq("construct costs", Co, Cn)
    fails += not eq("construct orders", Oo, On)

    _verdict(fails)


def _verdict(fails):
    print("\n" + "=" * 60)
    if fails:
        print(f"{fails} FAILURES -- do NOT switch to the new package yet")
        sys.exit(1)
    print("ALL EQUAL.  The new package is the old pipeline, bit for bit.")


if __name__ == "__main__":
    main()
