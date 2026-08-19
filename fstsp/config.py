"""
fstsp.config -- ONE place for every knob.

CFG / resolve() are fstsp_train_v4's, verbatim in semantics, with the keys the
wrapper scripts used to inject (adv_scale, leader_alpha, depot_mode,
val_depot_mode) promoted to first-class citizens.  PRESETS carries every
legacy preset unchanged (so any old command line maps 1:1) plus the canonical
per-size family:

    n10_mix4  n20_mix4  n50_mix4  n100_mix4        (+ n20_mix4_noLA ablation)

n50_mix4 is EXACTLY the live n50 lineage: `--preset n50_depot_rep1
--depot_mode mix4` spelled as one preset.

PHASES encodes the three-segment recipe.  A phase only fills values the
preset/CLI did not set, so `--phase leader --lr 3e-5` still wins.

    depot   main training           (no overrides -- the preset as-is)
    leader  Leader Reward final phase: alpha=inf, lr 5e-5, entropy 0
            (Wang et al., arXiv:2405.13947, their Alg. 2 / final-phase use)
    tail    small-lr finish:        alpha stays inf, lr 1e-5, entropy 0
"""

import argparse
import json
import sys

from .rollout import LOOK_W

# ===========================================================================
PRESET = None

CFG = dict(
    # ---- problem ----------------------------------------------------------
    n=20,
    n_min=0,
    n_max=0,
    profile="setM",
    elig_mode="strat",     # "strat" | "rate_cap" | "rate" | "count"
    inelig_rate=0.15,
    min_inelig=1,
    max_inelig=0,          # 0 -> ceil(0.25 * n)

    # ---- depot (was fstsp_train_depot) ------------------------------------
    depot_mode="center",   # center | centroid | edge | corner | murray3 | mix4
    val_depot_mode="",     # "" -> same as depot_mode

    # ---- rollout group:  S = starts * rep ---------------------------------
    start_mode="pomo",
    starts=0,
    rep=1,
    d4_group=1,
    look_w=LOOK_W,         # 0 = exact, >0 truncated window, <0 DISABLED (noLA)

    # ---- objective --------------------------------------------------------
    rtg=0,                 # tested at n=10 and n=13, both worse.  Leave at 0.
    adv_norm=1,
    adv_scale="group",     # was fstsp_advfix: rep>=2 scale across the group;
                           # "pair" restores plain fstsp_train_v4 bit for bit
    leader_alpha=0.0,      # was fstsp_leader: 0/1 off, >1 Alg.1, inf Alg.2
    aux_coef=0.10,
    aux_mask=1,
    entropy_coef=0.001,

    # ---- self-imitation ---------------------------------------------------
    sil=0,
    sil_coef=0.05,
    sil_every=10,
    sil_frac=0.25,
    sil_moves="oropt1",
    sil_passes=1,

    # ---- decoder population -----------------------------------------------
    n_dec=1,
    pop_loss=0,
    pop_mix=0.15,
    init_from="",
    init_noise=0.02,
    init_pad_noise=0.0,

    # ---- model ------------------------------------------------------------
    dim=128, heads=8, layers=4, ff=512,
    prenorm=0,
    use_edge_bias=1,
    use_cand_bias=1,
    use_dp_state=1,

    # ---- optimisation -----------------------------------------------------
    epochs=60000,
    roll_per_ep=1280,
    batch=0,
    lr=1e-4,
    lr_pop=3e-5,
    lr_warm=5e-5,
    weight_decay=1e-6,
    max_grad_norm=1.0,
    max_minutes=0,
    tf32=1,

    # ---- evaluation / model selection -------------------------------------
    val_size=256,
    val_seed=987654,
    val_data="",
    val_n=0,
    val_elig_mode="rate_cap",
    eval_budget=0,
    eval_beam=0,
    eval_kexp=0,
    eval_every=250,
    eval_chunk=64,

    # ---- stopping / bookkeeping -------------------------------------------
    plateau_w=0,           # >0: stop when best improved < plateau_delta over
    plateau_delta=0.05,    # the last plateau_w epochs (checked at evals)
    epoch_offset=0,        # added to the LOGGED epoch only (cumulative books)
    phase="",              # "" | depot | leader | tail   (see PHASES)

    # ---- housekeeping -----------------------------------------------------
    resume=0,
    seed=2026,
    device="auto",
    out="runs/v4",
)

PHASES = {
    "": {},
    "depot":  {},
    "leader": dict(leader_alpha=float("inf"), lr=5e-5, entropy_coef=0.0),
    "tail":   dict(leader_alpha=float("inf"), lr=1e-5, entropy_coef=0.0),
}

# ---------------------------------------------------------------------------
# presets.  LEGACY blocks are verbatim from fstsp_train_v4 / fstsp_train_n50 /
# fstsp_train_depot (with the depot launcher's implicit depot_mode="murray3"
# default baked in where a preset relied on it).  CANONICAL is the per-size
# family to use going forward.
# ---------------------------------------------------------------------------
PRESETS = {
    # ---- legacy: fstsp_train_v4 -------------------------------------------
    "n20_v4":      dict(n=20),
    "n20_v4_noLA": dict(n=20, look_w=-1),
    "n20_v4_sil":  dict(n=20, sil=1),
    "n20_v4_pop":  dict(n=20, n_dec=4, pop_loss=1, roll_per_ep=2560, sil=1),
    "n20_v4_big":  dict(n=20, dim=192, layers=6, ff=768, prenorm=1, sil=1),
    "n20_v4_mix":  dict(n_min=10, n_max=20, val_n=20, sil=1),

    # ---- legacy: fstsp_train_n50 ------------------------------------------
    "n50":         dict(n=50, roll_per_ep=2560, val_size=128, eval_every=250,
                        eval_chunk=32),
    "n50_scratch": dict(n=50, roll_per_ep=5120, val_size=128, eval_every=250,
                        eval_chunk=32, lr=1e-4),
    "n50_big":     dict(n=50, roll_per_ep=10240, val_size=128, eval_every=250,
                        eval_chunk=32, lr=1e-4),
    "n50_s2":      dict(n=50, roll_per_ep=5120, val_size=128, eval_every=250,
                        eval_chunk=32, lr=1e-4, seed=7),
    "n50_lite":    dict(n=50, roll_per_ep=1280, val_size=64, eval_every=250,
                        eval_chunk=32),

    # ---- legacy: fstsp_train_depot ----------------------------------------
    "n20_depot":     dict(n=20, depot_mode="murray3", rep=2, roll_per_ep=2560),
    "n20_depot_s2":  dict(n=20, depot_mode="murray3", rep=2, roll_per_ep=2560,
                          seed=7),
    "n20_depot4":    dict(n=20, depot_mode="mix4", rep=2, roll_per_ep=2560),
    "n50_depot":     dict(n=50, depot_mode="murray3", rep=2, roll_per_ep=10240,
                          val_size=128, eval_every=250, eval_chunk=32, lr=1e-4),
    "n50_depot_s2":  dict(n=50, depot_mode="murray3", rep=2, roll_per_ep=10240,
                          val_size=128, eval_every=250, eval_chunk=32, lr=1e-4,
                          seed=7),
    "n50_depot_lite": dict(n=50, depot_mode="murray3", rep=2, roll_per_ep=5120,
                           val_size=128, eval_every=250, eval_chunk=32,
                           lr=1e-4),
    "n50_depot_rep1": dict(n=50, depot_mode="murray3", rep=1, roll_per_ep=5120,
                           val_size=128, eval_every=250, eval_chunk=32,
                           lr=1e-4),

    # ---- CANONICAL per-size family (mix4, rep=1, the live lineage) --------
    # n50_mix4 == the running card-2 config: n50_depot_rep1 + depot_mode mix4.
    "n10_mix4":  dict(n=10, depot_mode="mix4", rep=1, roll_per_ep=1280,
                      val_size=256, eval_every=250, eval_chunk=64, lr=1e-4),
    "n20_mix4":  dict(n=20, depot_mode="mix4", rep=1, roll_per_ep=1280,
                      val_size=256, eval_every=250, eval_chunk=64, lr=1e-4),
    "n20_mix4_noLA": dict(n=20, depot_mode="mix4", rep=1, roll_per_ep=1280,
                          val_size=256, eval_every=250, eval_chunk=64, lr=1e-4,
                          look_w=-1),
    "n50_mix4":  dict(n=50, depot_mode="mix4", rep=1, roll_per_ep=5120,
                      val_size=128, eval_every=250, eval_chunk=32, lr=1e-4),
    "n100_mix4": dict(n=100, depot_mode="mix4", rep=1, roll_per_ep=10240,
                      val_size=128, eval_every=500, eval_chunk=16, lr=1e-4),
}
# ===========================================================================


def group_size(cfg):
    K = 1 if cfg["start_mode"] == "none" else \
        (cfg["n"] if cfg["starts"] in (0, None) else int(cfg["starts"]))
    return K, K * max(1, int(cfg["rep"]))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, choices=list(PRESETS))
    for k, v in CFG.items():
        t = int if isinstance(v, bool) else type(v)
        ap.add_argument(f"--{k}", type=t, default=None)
    return ap


def resolve(argv=None):
    ap = build_parser()
    if argv is None:
        argv = sys.argv[1:]
        if any(a == "-f" or a.endswith(".json") for a in argv):
            argv = []
    ns, unknown = ap.parse_known_args(argv)
    if unknown:
        print(json.dumps({"event": "warn_unknown_args", "unknown": unknown}))

    cfg = dict(CFG)
    name = ns.preset or PRESET
    if name:
        cfg.update(PRESETS[name])
        cfg["preset"] = name
    else:
        cfg["preset"] = "custom"

    # phase defaults sit between the preset and the CLI
    phase = ns.phase if ns.phase is not None else cfg.get("phase", "")
    if phase not in PHASES:
        raise SystemExit(f"--phase must be one of {list(PHASES)}, got {phase!r}")
    cfg.update(PHASES[phase])
    cfg["phase"] = phase

    for k in CFG:
        v = getattr(ns, k, None)
        if v is not None:
            cfg[k] = v

    if cfg["out"] == CFG["out"]:
        if name and phase:
            cfg["out"] = f"runs/{name}/{phase}"
        elif name:
            cfg["out"] = f"runs/{name}"

    if cfg["n_min"] and not cfg["n_max"]:
        cfg["n_max"] = cfg["n"]
    if cfg["n_max"]:
        cfg["n"] = cfg["n_max"]
    if not cfg["val_n"]:
        cfg["val_n"] = cfg["n"]
    if not cfg["eval_budget"]:
        cfg["eval_budget"] = 8 * cfg["val_n"]
    if cfg["pop_loss"] and cfg["lr"] == CFG["lr"]:
        cfg["lr"] = cfg["lr_pop"]
    elif cfg["init_from"] and cfg["lr"] == CFG["lr"]:
        cfg["lr"] = cfg["lr_warm"]

    K, S = group_size(cfg)
    if int(cfg["batch"]) <= 0:
        cfg["batch"] = max(1, int(cfg["roll_per_ep"]) // (S * max(1, cfg["n_dec"])))
    cfg["K"], cfg["S"] = K, S
    cfg["rollouts_per_epoch"] = cfg["batch"] * S * max(1, cfg["n_dec"])
    cfg["ver"] = 4

    if cfg["pop_loss"] and cfg["n_dec"] < 2:
        raise SystemExit("pop_loss needs n_dec >= 2")
    if cfg["n_dec"] > 1 and not cfg["init_from"] and not cfg["resume"]:
        print(json.dumps({"event": "warn", "msg":
              "a population trained from scratch spends most of its budget "
              "rediscovering what one decoder already knows -- pass "
              "--init_from <best.pt>"}))
    if phase in ("leader", "tail") and not cfg["init_from"] and not cfg["resume"]:
        raise SystemExit(f"--phase {phase} is a fine-tuning segment: pass "
                         f"--init_from <previous phase best.pt> or --resume 1")
    return argparse.Namespace(**cfg)
