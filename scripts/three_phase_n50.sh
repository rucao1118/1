#!/usr/bin/env bash
# =============================================================================
# The three-segment training recipe, n=50 (the live lineage, spelled out).
#   A  depot   mix4 main training, POMO rep=1, adv group scale (inert at rep=1)
#   B  leader  Leader Reward final phase: alpha=inf, lr 5e-5, entropy 0
#   C  tail    same, lr 1e-5, short
# Segments are separate --out dirs chained by --init_from (NOT --resume:
# resume restores the optimizer state; a new segment wants a fresh AdamW at
# the new lr).  --epoch_offset only relabels the LOGGED epoch so the three
# log.jsonl files concatenate into one cumulative curve.
# Run `--preflight_only` once per card before booking a night.
# =============================================================================
set -euo pipefail

OUT=runs/n50_mix4
SEED=61001

# ---- A: main phase ----------------------------------------------------------
python -m fstsp.train --preset n50_mix4 --phase depot \
    --epochs 26000 --max_minutes 1400 \
    --plateau_w 2000 --plateau_delta 0.05 \
    --seed $SEED --out $OUT/depot
# interrupted night?  continue with:
#   python -m fstsp.train --preset n50_mix4 --phase depot --resume 1 \
#       --epochs 26000 --seed $((SEED+1)) --out $OUT/depot

BEST_A=$OUT/depot/best.pt
EP_A=$(python -c "import torch;print(int(torch.load('$BEST_A',map_location='cpu',weights_only=False)['epoch']))")

# ---- B: leader final phase (alpha=inf, lr 5e-5, entropy 0 via --phase) ------
python -m fstsp.train --preset n50_mix4 --phase leader \
    --init_from $BEST_A \
    --epochs 24000 --max_minutes 1400 \
    --plateau_w 2000 --plateau_delta 0.05 \
    --epoch_offset $EP_A \
    --seed $((SEED+2)) --out $OUT/leader

BEST_B=$OUT/leader/best.pt
EP_B=$(python -c "import torch;print(int(torch.load('$BEST_B',map_location='cpu',weights_only=False)['epoch']))")

# ---- C: tail (lr 1e-5, short) ------------------------------------------------
python -m fstsp.train --preset n50_mix4 --phase tail \
    --init_from $BEST_B \
    --epochs 6000 --max_minutes 470 \
    --plateau_w 1500 --plateau_delta 0.03 \
    --epoch_offset $((EP_A + EP_B)) \
    --seed $((SEED+4)) --out $OUT/tail

# ---- final evaluation ---------------------------------------------------------
python -m fstsp.evaluate --data data/test_n50_mix4.pt \
    --ckpt $OUT/tail/best.pt --tag n50 --topk_ls 1 4 --ils_passes 10 \
    --strata --out eval/n50_final.csv

# n=20 is the same script with:  --preset n20_mix4, epochs ~55000/12000/3000,
# data data/test_n20_mix4.pt.  The noLA ablation arm is the same three lines
# with --preset n20_mix4_noLA and a different --out.
