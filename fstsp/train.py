"""
fstsp.train -- the ONE trainer.

    python -m fstsp.train --preset n50_mix4 --phase depot  --out runs/n50/depot
    python -m fstsp.train --preset n50_mix4 --phase leader \
        --init_from runs/n50/depot/best.pt --epochs 24000 --seed 61003
    python -m fstsp.train --preset n50_mix4 --phase tail \
        --init_from runs/n50/leader/best.pt --epochs 6000 --seed 61005

Replaces the fstsp_leader.py -> fstsp_advfix.py -> fstsp_train_depot.py ->
fstsp_train_v4.py wrapper chain.  Nothing is monkey-patched:

  * data goes through fstsp.data.generate_batch with depot_mode passed
    EXPLICITLY, for training AND for the validation set (val_depot_mode),
    which is what the old launcher's one-name patch achieved;
  * the advantage is fstsp.losses.advantage, the flattened
    leader(advfix(base)) dispatch;
  * the depot guards, the runtime probe, and the n=50 preflight ride along
    as flags (--probe_only / --preflight_only) instead of separate scripts.

The training loop, checkpoint format ({model,opt,args,epoch,metric,best,ver}),
log.jsonl rows and model-selection rule are fstsp_train_v4's, verbatim.
Old checkpoints load unchanged; new checkpoints load in the old scripts.

Launcher-level flags (not part of the config, not saved):
  --probe_only        classify what gen()/make_val() actually produce, exit
  --preflight_only    time 3 real steps, project the slot, exit
  --preflight_steps N
  --max_hours H       refuse to launch if the projection exceeds H
  --force             override the init_from depot-compatibility guard
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

from .config import CFG, PRESETS, resolve, group_size
from .data import (FEAT_DIM, EDGE_DIM, generate_batch, classify_depot,
                   ALL_MODES, SINGLE_MODES)
from .model import FSTSPv4, inflate_v3_state_dict
from .rollout import rollout_v4, solve_budget_v4
from .losses import advantage, masked_bce, sil_term

# sizing reference used by preflight (the n=20 v4 run)
N20_BATCH, N20_BEST_EPOCH = 64, 54750
N20_INSTANCES = N20_BATCH * N20_BEST_EPOCH


# ---------------------------------------------------------------------- data
def gen(a, bs, n, device, seed=None):
    mx = int(a.max_inelig) or None
    b = generate_batch(bs, n_customer=n, profile=a.profile, device=device,
                       seed=seed, balanced=True, depot_mode=a.depot_mode,
                       elig_mode=a.elig_mode, inelig_rate=a.inelig_rate,
                       min_inelig=a.min_inelig, max_inelig=mx)
    b.pop("depot_mode_idx", None)      # keep the batch dict byte-identical in
    return b                           # shape to what v4 hands downstream


def make_val(a, device):
    """
    Fixed seed on CPU so every machine sees the same set, drawn from
    val_elig_mode and val_depot_mode -- the distribution the paper reports
    on.  Selecting best.pt on the training distribution when you report on
    another one is how a model quietly gets tuned to a tail you said you
    were not reporting.
    """
    if a.val_data:
        b = torch.load(a.val_data, map_location="cpu", weights_only=False)
        b = {k: v for k, v in b.items() if k != "meta"}
    else:
        vmode = a.val_depot_mode or a.depot_mode
        parts, done = [], 0
        while done < a.val_size:
            bs = min(64, a.val_size - done)
            parts.append(generate_batch(
                bs, n_customer=a.val_n, profile=a.profile, device="cpu",
                seed=a.val_seed + done, balanced=True, depot_mode=vmode,
                elig_mode=a.val_elig_mode, inelig_rate=a.inelig_rate,
                min_inelig=a.min_inelig,
                max_inelig=(int(a.max_inelig) or None)))
            done += bs
        keys = ["coords", "feat", "edge_feat", "T", "D", "elig", "e"]
        b = {k: torch.cat([p[k] for p in parts], 0) for k in keys}
        b.update({k: parts[0][k] for k in ("sL", "sR", "depot", "profile")})
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


# ------------------------------------------------------------------- a step
def train_step(model, batch, a, ep=0):
    """One optimiser step's worth of loss.  Returns (loss, log row)."""
    rs = [rollout_v4(model, batch, starts=a.starts, rep=a.rep,
                     d4_group=bool(a.d4_group), greedy=False,
                     temperature=1.0, start_mode=a.start_mode,
                     want_labels=True, dec=d, look_w=a.look_w)
          for d in range(a.n_dec)]

    B, S, N = rs[0]["orders"].shape
    Kk, rep = rs[0]["K"], rs[0]["rep"]

    rl_terms = []
    for r in rs:
        adv = advantage(r["cost"], r["dp_at"], Kk, rep, a)
        rl_terms.append((adv.detach().to(r["logp_steps"].dtype)
                         * r["logp_steps"]).sum(-1))

    if a.pop_loss:
        dbest = torch.stack([r["cost"].min(1).values for r in rs], 1)
        srt, order = dbest.sort(1)
        win = order[:, 0]
        margin = (srt[:, 1] - srt[:, 0]).clamp_min(0.0)
        scale = margin / margin.mean().clamp_min(1e-6)
        rl = 0.0
        for d, r in enumerate(rs):
            is_win = (win == d).to(r["cost"].dtype)
            wbest = r["cost"].argmin(1)
            lp_best = r["logp_steps"].sum(-1).gather(1, wbest[:, None]).squeeze(1)
            rl = rl - (is_win * scale * lp_best).mean()
            rl = rl + a.pop_mix * (is_win.logical_not().to(rl_terms[d].dtype)[:, None]
                                   * rl_terms[d]).mean()
        rl_loss = rl
    else:
        rl_loss = torch.stack(rl_terms, 0).mean()

    target = torch.stack([r["labels"].mean(1) for r in rs], 0).mean(0)
    mask = batch["elig"][:, :N].to(target.dtype) if a.aux_mask \
        else torch.ones_like(target)
    dl = model.drone_logits(rs[0]["node_emb0"])[:, :N]
    aux_loss = masked_bce(dl, target, mask)
    ent = torch.stack([r["entropy"].mean() for r in rs]).mean()

    loss = rl_loss + a.aux_coef * aux_loss - a.entropy_coef * ent

    row = {"rl_loss": float(rl_loss.detach()),
           "aux_loss": float(aux_loss.detach()),
           "entropy": float(ent.detach()), "drone_rate": float(target.mean()),
           "train_cost_mean": float(rs[0]["cost"].mean()),
           "train_cost_best": float(rs[0]["cost"].min(1).values.mean()),
           "sil_n": 0, "sil_gain": 0.0}

    if a.sil and ep % max(1, int(a.sil_every)) == 0:
        with torch.no_grad():
            bi = rs[0]["cost"].argmin(1)
            bo = rs[0]["orders"][torch.arange(B, device=rs[0]["orders"].device), bi]
        nll, nimp, gain = sil_term(model, batch, a, bo, N, dec=0)
        if nimp:
            loss = loss + a.sil_coef * nll
            row["sil_n"], row["sil_gain"] = nimp, gain

    if a.n_dec > 1:
        dbest = torch.stack([r["cost"].min(1).values for r in rs], 1)
        row["dec_win_share"] = [float((dbest.argmin(1) == d).float().mean())
                                for d in range(a.n_dec)]
        row["dec_mean_cost"] = [float(x) for x in dbest.mean(0)]
    return loss, row


@torch.no_grad()
def evaluate(model, val, a):
    model.eval()
    B = val["T"].shape[0]
    out = []
    for s in range(0, B, a.eval_chunk):
        sl = slice(s, min(s + a.eval_chunk, B))
        sub = {k: (v[sl] if torch.is_tensor(v) and v.shape[0] == B else v)
               for k, v in val.items()}
        c, _ = solve_budget_v4(model, sub, budget=a.eval_budget,
                               start_mode=a.start_mode, starts=0,
                               force_starts=(a.start_mode == "none"),
                               look_w=a.look_w, beam=a.eval_beam,
                               k_expand=a.eval_kexp)
        out.append(c.float().cpu())
    model.train()
    return float(torch.cat(out).mean())


# ------------------------------------------------------------ depot machinery
def _hist(midx):
    c = torch.bincount(midx.clamp_min(0), minlength=4).tolist()
    bad = int((midx < 0).sum())
    s = "  ".join(f"{m}:{n}" for m, n in zip(SINGLE_MODES, c) if n)
    return s + (f"  UNCLASSIFIED:{bad}" if bad else "")


def probe(a):
    """Runtime proof of what gen() and make_val() actually produce."""
    val_mode = a.val_depot_mode or a.depot_mode
    print("=" * 78)
    print(f"PROBE   train depot_mode={a.depot_mode}   val depot_mode={val_mode}")
    print("=" * 78)

    b = gen(a, 128, a.n, "cpu", seed=1234)
    mt = classify_depot(b)
    print(f"  gen()      128 inst, n={a.n}:  {_hist(mt)}")
    if int((mt < 0).sum()):
        raise SystemExit("!! gen() produced depots the classifier cannot "
                         "identify -- do not launch")

    va = argparse.Namespace(**{**vars(a), "val_size": min(int(a.val_size), 64)})
    val = make_val(va, "cpu")
    mv = classify_depot(val)
    src = f"--val_data {a.val_data}" if a.val_data else "generated"
    print(f"  make_val() {va.val_size} inst ({src}):  {_hist(mv)}")

    mixed_train = a.depot_mode in ("murray3", "mix4")
    if mixed_train and bool((mv == 0).all()):
        raise SystemExit(
            "\n!! training is depot-mixed but the validation set is entirely "
            "centre-depot.\n   best.pt would be selected on a distribution "
            "the run is not training for.\n   Drop --val_data, or point it at "
            "a depot-mixed frozen set.")
    if a.val_data:
        print("   note: --val_data bypasses the generator; the mix above is "
              "whatever is in that file")

    d = int(b["depot"])
    Td = b["T"][:, d, :a.n]
    print("  mean depot->customer truck time, by mode:")
    for i, m in enumerate(SINGLE_MODES):
        sel = mt == i
        if int(sel.sum()):
            print(f"    {m:>9}: {int(sel.sum()):>4} inst   "
                  f"{float(Td[sel].mean()):6.2f} min")
    print()


def depot_guards(a, force):
    """The old fstsp_train_depot launch guards, with the init_from guard made
    checkpoint-aware: an init_from whose ck['args'].depot_mode matches this
    run's is accepted without --force (that is the three-phase case)."""
    if a.init_from:
        src_mode = "center"
        try:
            ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
            src_mode = str((ck.get("args") or {}).get("depot_mode", "center"))
            del ck
        except FileNotFoundError:
            raise SystemExit(f"--init_from {a.init_from}: file not found")
        except Exception as ex:
            print(json.dumps({"event": "warn",
                              "msg": f"could not read init_from args ({ex}); "
                                     f"treating source as depot_mode=center"}))
        if src_mode != a.depot_mode and not force:
            raise SystemExit(
                f"--init_from was trained with depot_mode={src_mode!r}, this "
                f"run uses {a.depot_mode!r}.  Warm-starting across depot "
                f"distributions re-introduces the bias the depot runs exist "
                f"to remove.  Pass --force if you mean it.")

    marker = os.path.join(a.out, "depot_mode.json")
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):
        if not os.path.exists(marker):
            raise SystemExit(
                f"{a.out} holds a run trained WITHOUT a depot marker.  "
                f"Resuming it under depot_mode={a.depot_mode!r} splices two "
                f"distributions into one curve and makes best.pt "
                f"uninterpretable.  Use a fresh --out.")
        old = json.load(open(marker))
        if old.get("depot_mode") != a.depot_mode:
            raise SystemExit(f"{a.out} was trained with depot_mode="
                             f"{old.get('depot_mode')!r}, not {a.depot_mode!r}")
    if a.depot_mode in ("murray3", "mix4", "edge", "corner") and a.rep < 2:
        print("note: rep=1 under non-centre depots uses the baseline SHARED "
              "across POMO starts.  This is the config-matched arm "
              "(n50_mix4 lineage); rep=2 presets exist if you want the "
              "within-start baseline instead.\n")


def write_marker(a):
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "depot_mode.json"), "w") as f:
        json.dump({"depot_mode": a.depot_mode,
                   "val_depot_mode": a.val_depot_mode or a.depot_mode,
                   "phase": a.phase, "preset": a.preset,
                   "launcher": "fstsp.train",
                   "generator": "fstsp.data.generate_batch"}, f, indent=2)


# ---------------------------------------------------------------- preflight
def _build_model(a, device):
    m = FSTSPv4(in_dim=FEAT_DIM, edge_dim=EDGE_DIM, dim=a.dim, heads=a.heads,
                layers=a.layers, ff=a.ff,
                use_edge_bias=bool(a.use_edge_bias),
                use_cand_bias=bool(a.use_cand_bias),
                use_dp_state=bool(a.use_dp_state),
                n_dec=a.n_dec, prenorm=bool(a.prenorm)).to(device)
    info = {}
    if a.init_from:
        ck = torch.load(a.init_from, map_location=device, weights_only=False)
        sd = inflate_v3_state_dict(ck["model"], cand_dim=m.cand_dim,
                                   n_dec=a.n_dec, noise=a.init_pad_noise)
        rep = m.load_state_dict(sd, strict=False)
        if a.n_dec > 1 and int(ck.get("args", {}).get("n_dec", 1) or 1) == 1:
            m.clone_population(noise=a.init_noise)
        info = {"src_ver": int(ck.get("ver", 3)),
                "src_epoch": int(ck.get("epoch", -1)),
                "src_n": int((ck.get("args") or {}).get("n", 0)),
                "src_depot_mode": str((ck.get("args") or {}).get("depot_mode",
                                                                 "center")),
                "missing": len(rep.missing_keys),
                "unexpected": len(rep.unexpected_keys),
                "missing_keys": rep.missing_keys[:8]}
    return m, info


def preflight(a, device, steps=3):
    print("=" * 84)
    print("PREFLIGHT")
    print("=" * 84)
    K, S = group_size(vars(a))
    print(f"n={a.n}  starts K={K}  group S={S}  n_dec={a.n_dec}  "
          f"depot_mode={a.depot_mode}  phase={a.phase or '-'}")
    print(f"roll_per_ep={a.roll_per_ep}  ->  batch={a.batch}  "
          f"(rollouts/epoch {a.rollouts_per_epoch})")
    if a.batch < 32:
        print(f"  !! batch {a.batch} is small; gradient noise will be "
              f"visibly higher than n=20's 64.  Consider --roll_per_ep "
              f"{a.roll_per_ep * 2}.")

    model, info = _build_model(a, device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params {nparam:,}")
    if a.init_from:
        print(f"init_from {a.init_from}: ver {info['src_ver']} epoch "
              f"{info['src_epoch']} trained at n={info['src_n'] or '?'} "
              f"depot_mode={info['src_depot_mode']}")
        print(f"  missing {info['missing']}  unexpected {info['unexpected']}")
        if info["missing"] > 4:
            print(f"  !! {info['missing']} missing keys -- the warm start "
                  f"reset part of the model: {info['missing_keys']}")
            print("  !! do NOT launch; fix init_from first")
            return None, False

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    print(f"\ntiming {steps} full steps (generate + rollout + DP + backward)")
    times = []
    for i in range(steps + 1):                       # first step is warm-up
        t0 = time.time()
        batch = gen(a, a.batch, a.n, device)
        loss, row = train_step(model, batch, a, ep=1)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
            opt.step()
        else:
            print("  !! non-finite loss on step", i)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if i:
            times.append(dt)
        print(f"  step {i}{' (warm-up, ignored)' if i == 0 else ''}: "
              f"{dt:.2f}s   train_cost_best {row['train_cost_best']:.3f}")

    ms = 1000 * sum(times) / len(times)
    print(f"\nms/epoch {ms:.0f}")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        free, total = torch.cuda.mem_get_info()
        total /= 2 ** 30
        print(f"peak GPU mem {peak:.2f} GiB of {total:.0f} GiB")
        print(f"  first evaluation peaks higher than a training step "
              f"({a.eval_chunk} inst x {a.eval_budget} rollouts); if it OOMs, "
              f"lower --eval_chunk, not --roll_per_ep")

    hours = ms / 1000 * a.epochs / 3600
    print(f"\nthroughput {a.batch / (ms / 1000.0):.0f} instances/s")
    print(f"projected {a.epochs} epochs = {hours:.1f} h "
          f"= {a.epochs * a.batch / 1e6:.2f}M instances "
          f"({a.epochs * a.batch / N20_INSTANCES:.2f}x the n=20 run)")
    print(f"\nsizing against the n=20 run ({N20_INSTANCES / 1e6:.2f}M "
          f"instances, batch {N20_BATCH} x epoch {N20_BEST_EPOCH})")
    print(f"  {'multiple':>9}  {'instances':>10}  {'epochs':>8}  {'hours':>7}")
    for mult in (1.0, 1.25, 1.5, 2.0):
        ep = int(round(mult * N20_INSTANCES / a.batch))
        print(f"  {mult:>8.2f}x  {mult * N20_INSTANCES / 1e6:>9.2f}M  "
              f"{ep:>8d}  {ms / 1000 * ep / 3600:>6.1f} h")
    if a.max_minutes:
        done = int(a.max_minutes * 60 * 1000 / ms)
        print(f"--max_minutes {a.max_minutes} -> about {done} epochs "
              f"({100 * done / a.epochs:.0f}% of the schedule)")
    return ms, True


# ---------------------------------------------------------------------- main
def main(argv=None):
    lp = argparse.ArgumentParser(add_help=False)
    lp.add_argument("--probe_only", action="store_true")
    lp.add_argument("--preflight_only", action="store_true")
    lp.add_argument("--preflight_steps", type=int, default=3)
    lp.add_argument("--no_preflight", action="store_true")
    lp.add_argument("--max_hours", type=float, default=0.0)
    lp.add_argument("--force", action="store_true")
    known, rest = lp.parse_known_args(argv if argv is not None
                                      else sys.argv[1:])

    a = resolve(rest)
    if a.depot_mode not in ALL_MODES:
        raise SystemExit(f"depot_mode must be one of {ALL_MODES}, "
                         f"got {a.depot_mode!r}")
    depot_guards(a, known.force)
    probe(a)
    if known.probe_only:
        print("probe only, not launching")
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    if known.preflight_only or (known.max_hours and not known.no_preflight):
        ms, ok = preflight(a, device, steps=known.preflight_steps)
        if ms and known.max_hours:
            h = ms / 1000 * a.epochs / 3600
            if h > known.max_hours and not known.force:
                raise SystemExit(
                    f"\nprojected {h:.1f} h > --max_hours {known.max_hours}. "
                    f"Lower --epochs, set --max_minutes, or pass --force.")
        if not ok and not known.force:
            raise SystemExit("\npreflight failed; pass --force to launch anyway")
        if known.preflight_only:
            print("\npreflight only, not launching")
            return

    write_marker(a)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    if a.tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = FSTSPv4(in_dim=FEAT_DIM, edge_dim=EDGE_DIM, dim=a.dim,
                    heads=a.heads, layers=a.layers, ff=a.ff,
                    use_edge_bias=bool(a.use_edge_bias),
                    use_cand_bias=bool(a.use_cand_bias),
                    use_dp_state=bool(a.use_dp_state),
                    n_dec=a.n_dec, prenorm=bool(a.prenorm)).to(device)

    if a.init_from:
        ck = torch.load(a.init_from, map_location=device, weights_only=False)
        sd = inflate_v3_state_dict(ck["model"], cand_dim=model.cand_dim,
                                   n_dec=a.n_dec, noise=a.init_pad_noise)
        rep = model.load_state_dict(sd, strict=False)
        if a.n_dec > 1 and int(ck.get("args", {}).get("n_dec", 1) or 1) == 1:
            model.clone_population(noise=a.init_noise)
        print(json.dumps({"event": "init_from", "path": a.init_from,
                          "src_ver": int(ck.get("ver", 3)),
                          "epoch": int(ck.get("epoch", -1)),
                          "missing": len(rep.missing_keys),
                          "unexpected": len(rep.unexpected_keys),
                          "cand_pad_noise": a.init_pad_noise,
                          "note": "zero pad => v4 starts as the SAME policy"
                                  if not a.init_pad_noise else "pad perturbed"}),
              flush=True)
        if len(rep.missing_keys) > 4:
            print(json.dumps({"event": "warn", "missing_keys":
                              rep.missing_keys[:12]}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)

    ep0, best, nan_skips = 1, float("inf"), 0
    last_path = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        ep0 = int(ck.get("epoch", 0)) + 1
        best = float(ck.get("best", ck.get("metric", float("inf"))))
        print(json.dumps({"event": "resume", "epoch": ep0, "best": best,
                          "note": "RNG is NOT restored: change --seed, and "
                                  "re-pass every non-default flag "
                                  "(leader_alpha, lr, ...)"}), flush=True)

    val = make_val(a, device)
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    log_path = os.path.join(a.out, "log.jsonl")

    print(json.dumps({"event": "start", "device": device,
                      "params": sum(p.numel() for p in model.parameters()),
                      "cost_mode": "strict_murray (proven on Set M 72/72)",
                      **vars(a)}), flush=True)

    t0 = time.time()
    tmark, epmark = t0, ep0 - 1
    eval_hist = []                       # (epoch, best) at every evaluation

    for ep in range(ep0, a.epochs + 1):
        n_ep = a.n if not a.n_min else random.randint(a.n_min, a.n_max)
        batch = gen(a, a.batch, n_ep, device)

        loss, row = train_step(model, batch, a, ep=ep)

        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            nan_skips += 1
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
        if not torch.isfinite(gn):
            nan_skips += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()

        stop = bool(a.max_minutes) and (time.time() - t0) > a.max_minutes * 60
        plateau = False
        if ep == ep0 or ep % a.eval_every == 0 or stop or ep == a.epochs:
            v = evaluate(model, val, a)
            now = time.time()
            row.update({"epoch": ep + int(a.epoch_offset), "epoch_local": ep,
                        "n": n_ep, "val_cost": v,
                        "grad_norm": float(gn), "nan_skips": nan_skips,
                        "ms_per_epoch": round(1000 * (now - tmark)
                                              / max(ep - epmark, 1), 1),
                        "wall": round(now - t0, 1)})
            tmark, epmark = now, ep
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "args": vars(a), "epoch": ep, "metric": v,
                  "best": min(best, v), "ver": 4}
            torch.save(ck, last_path)
            if v < best:
                best = v
                row["saved_best"] = True
                torch.save({k: x for k, x in ck.items() if k != "opt"},
                           os.path.join(a.out, "best.pt"))
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

            eval_hist.append((ep, best))
            if a.plateau_w and ep - ep0 + 1 >= a.plateau_w:
                ref = [b for (e_, b) in eval_hist if e_ <= ep - a.plateau_w]
                if ref and (min(ref) - best) < float(a.plateau_delta):
                    print(json.dumps({"event": "plateau_stop", "epoch": ep,
                                      "window": a.plateau_w,
                                      "delta": a.plateau_delta,
                                      "best": best}), flush=True)
                    plateau = True

        if stop:
            print(json.dumps({"event": "time_limit", "epoch": ep}), flush=True)
            break
        if plateau:
            break

    print(json.dumps({"event": "done", "best_val_cost": best,
                      "nan_skips": nan_skips,
                      "wall": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
