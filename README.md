# fstsp — clean pipeline

One package, no monkey-patching, every number the old stack produced. The
cost convention everywhere is **strict Murray ("current")** — the one proven
72/72 on Set M; `SplitDPTorch` is the reward, the evaluator, and the thing the
Gurobi MILP and `fstsp.exact` are checked against.

```
fstsp/
  config.py        CFG + PRESETS (all legacy names kept) + PHASES + resolve()
  data.py          v3+v4+depot merged; generate_batch(depot_mode=..., OOD sets)
  dp.py            SplitDPTorch (reward) + SplitDPTorchV4 (lookahead/beam)
  model.py         FSTSPv4, inflate_v3_state_dict (ckpt-key compatible)
  rollout.py       rollout_v4, beam_decode(w_dp,w_lp), solve_budget_v4
  losses.py        advantage = leader ∘ advfix ∘ base, flattened; SIL bits
  train.py         the ONE trainer: --phase depot|leader|tail, guards, probe,
                   preflight, plateau stop
  evaluate.py      post_v4 rows + --strata + beam row (--w_dp 0 = ablation)
  testset.py       freeze sets, incl. OOD --v_drone_set / --endurance_set
  exact.py         split_dp variants + exact subset DP (numpy)
  io_murray.py     .pt -> Murray folders (for run_hgatac.jl), round-trip checked
  baselines/
    classic.py     OR-Tools TSP + split, ALNS, truck_only_cost column
    gurobi_milp.py MILP of the pipeline convention + --selftest + dual bounds
scripts/
  check_equivalence.py   THE GATE — run before switching (see below)
  three_phase_n50.sh     the A/B/C recipe, spelled out
  run_setM_exact.py      unchanged protocol, imports fstsp.exact
```

## Old → new

| old | new |
|---|---|
| `fstsp_leader.py --leader_alpha inf --preset n50_depot_rep1 --depot_mode mix4 ...` | `python -m fstsp.train --preset n50_mix4 --phase leader ...` |
| `fstsp_advfix.py --adv_scale group ...` | `--adv_scale group` (the default) |
| `fstsp_train_depot.py --preset X --depot_mode m` | `python -m fstsp.train --preset X --depot_mode m` |
| `fstsp_train_n50.py --preflight_only` | `python -m fstsp.train ... --preflight_only` |
| `fstsp_train_v4.py --preset n20_v4` | `python -m fstsp.train --preset n20_v4` (legacy presets kept) |
| `fstsp_post_v4.py` | `python -m fstsp.evaluate` (default flags = same rows/CSVs) |
| `fstsp_testset_v4.py` / `fstsp_data_depot.py --make_testset` | `python -m fstsp.testset` |
| `fstsp_n20_baselines.py` | `python -m fstsp.baselines.classic` |
| `fstsp_pt_to_murray.py` | `python -m fstsp.io_murray` |

Old checkpoints load unchanged (`{model,args,ver,epoch,metric,best}` format
kept, state-dict keys identical); new checkpoints load in the old scripts.

## Before switching: the gate

```
cd <repo root with the old fstsp_*.py>
python <new>/scripts/check_equivalence.py --new_dir <new> \
    --ckpt runs/n50_lead_inf/best.pt --device cuda
```

Bit-equality on data streams, the advantage chain (pair/group × α 0/10/inf),
greedy+sampled rollouts, one full train_step loss, strict ckpt load, and
`construct`. Anything printing FAIL → don't switch, report the section.

## Three-phase training (full commands)

`scripts/three_phase_n50.sh` is the runnable version. In short (n=50):

```
# A  main (mix4 depots, rep=1 — the live lineage)
python -m fstsp.train --preset n50_mix4 --phase depot \
    --epochs 26000 --max_minutes 1400 --plateau_w 2000 --plateau_delta 0.05 \
    --seed 61001 --out runs/n50_mix4/depot

# B  Leader Reward final phase (alpha=inf, lr 5e-5, entropy 0 come from --phase)
python -m fstsp.train --preset n50_mix4 --phase leader \
    --init_from runs/n50_mix4/depot/best.pt \
    --epochs 24000 --epoch_offset <epoch_of_A_best> \
    --seed 61003 --out runs/n50_mix4/leader

# C  tail (lr 1e-5)
python -m fstsp.train --preset n50_mix4 --phase tail \
    --init_from runs/n50_mix4/leader/best.pt \
    --epochs 6000 --epoch_offset <cumulative> \
    --seed 61005 --out runs/n50_mix4/tail
```

Rules encoded: segments chain by `--init_from` (fresh AdamW at the new lr),
never `--resume` across phases; `--resume 1` continues an interrupted segment
in the same `--out` (change `--seed`, re-pass non-default flags);
`--epoch_offset` relabels logged epochs only, so the three `log.jsonl`
concatenate into one cumulative curve; `--phase leader` refuses to start
without `--init_from`/`--resume`. `n10_mix4 / n20_mix4 / n100_mix4` are the
same recipe at other sizes (`n100_mix4`: run `--preflight_only` first, expect
to lower `--eval_chunk` before `roll_per_ep`).

## Ablation arms

* **Lookahead (needs a retrain):** train the control with
  `--preset n20_mix4_noLA` (that is `look_w=-1`: candidate channels 10–12 held
  at exact zero, so it is the v3 function under the v4 recipe). Same three
  phases, same seeds, different `--out`. Evaluate both at n=20.
* **DP-guided beam vs vanilla NCO beam (inference only, no retrain):**
  ```
  python -m fstsp.evaluate --data data/test_n50_mix4.pt --ckpt <final best.pt> \
      --beam 0 --k_expand 5 --w_dp 1.0 --w_lp 0.1 --out eval/n50_beam.csv
  python -m fstsp.evaluate ... --w_dp 0.0 --w_lp 1.0 --out eval/n50_beam_nco.csv
  ```

## Test sets (incl. OOD)

```
python -m fstsp.testset --n 50  --num 128 --seed 50711 --depot_mode mix4 \
    --elig_mode rate_cap --out data/test_n50_mix4.pt
python -m fstsp.testset --n 100 --num 128 --seed 100711 --depot_mode mix4 \
    --elig_mode rate_cap --out data/test_n100_mix4.pt
# OOD speed / endurance
python -m fstsp.testset --n 50 --num 128 --seed 50733 --depot_mode mix4 \
    --v_drone_set 45 --out data/test_n50_v45.pt
```

A NEW n=10 set (mix4) needs a NEW exact run: export with `fstsp.io_murray`,
then `scripts/run_setM_exact.py` — never rebuild `test_n10_v3.pt`/`n13`.

## Evaluation & baselines

```
python -m fstsp.evaluate --data data/test_n50_mix4.pt --ckpt <best.pt> \
    --tag n50 --topk_ls 1 4 --ils_passes 10 --strata \
    --hga_csv eval/hga_n50.csv --hga_col hga_best --out eval/n50_final.csv
python -m fstsp.baselines.classic --data data/test_n50_mix4.pt \
    --methods ortools --ortools_time 5 --out eval/n50_ortools.csv
```

`--strata` writes `<out>_strata.csv`: means (and gaps, if `--hga_csv`) by
v_drone, endurance, n_inelig and depot mode — the Murray-5.1-style tables.
`classic` now also records `truck_only_cost` (the savings denominator).

## Gurobi reference (protocol)

```
python -m fstsp.baselines.gurobi_milp --selftest          # MUST pass first
python -m fstsp.baselines.gurobi_milp --data data/test_n20_mix4.pt \
    --sample 24 --time_limit 1800 --threads 8 --out eval/n20_gurobi.csv
```

Selftest solves n∈{6,7,8} to zero gap and asserts equality (1e-5) with
`fstsp.exact.exact_dp(variant="current")` — the solver behind the proven n=10
optima. The runner samples stratified over (v_drone, e). Report
solved/K and mean MIPGap; keep the `bound` column: it certifies
`gap(method) ≤ cost/bound − 1` for **every** method on those instances, even
where Gurobi never closes. Expect n=20 to be hard and n=50 to mostly stall —
that is the point of the row. (`--mipfocus 3` pushes the dual bound.)
