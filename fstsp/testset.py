"""
fstsp.testset -- freeze test sets (replaces fstsp_testset_v4.py and
fstsp_data_depot.py --make_testset with one CLI).

depot_mode="center" reproduces the old fstsp_testset_v4 instance stream bit
for bit (same seed => same coordinates/speeds/endurance/eligibility), because
generate_batch(depot_mode="center") is proven bit-exact against
generate_batch_v4.  Self-checks run before anything is written.

    # headline n=50 set, mixed depots, capped eligibility band
    python -m fstsp.testset --n 50 --num 128 --seed 50711 \
        --depot_mode mix4 --elig_mode rate_cap --out data/test_n50_mix4.pt

    # OOD: faster drone, everything else identical in distribution
    python -m fstsp.testset --n 50 --num 128 --seed 50733 \
        --depot_mode mix4 --elig_mode rate_cap --v_drone_set 45 \
        --out data/test_n50_v45.pt

WARNING, the same one as always: the eligibility draw sits in the middle of
the RNG stream, so any change to n / seed / elig_mode / value sets is a
DIFFERENT instance set.  Never rebuild data/test_n10_v3.pt or
data/test_n13_v3.pt (their .exact.csv companions are proven optima for the
v3 stream); a NEW n=10 set needs a NEW exact run.
"""

import argparse

from .data import (ALL_MODES, assert_matches_v4, assert_classify,
                   make_testset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=50711)
    ap.add_argument("--depot_mode", default="mix4", choices=ALL_MODES)
    ap.add_argument("--elig_mode", default="rate_cap",
                    choices=["strat", "rate_cap", "rate", "count"])
    ap.add_argument("--inelig_rate", type=float, default=0.15)
    ap.add_argument("--min_inelig", type=int, default=1)
    ap.add_argument("--max_inelig", type=int, default=0,
                    help="0 -> ceil(0.25 n)")
    ap.add_argument("--v_drone_set", nargs="+", type=float, default=None,
                    help="override {15,25,35}; e.g. --v_drone_set 45 for OOD")
    ap.add_argument("--endurance_set", nargs="+", type=float, default=None,
                    help="override {20,40}; e.g. --endurance_set 60 for OOD")
    ap.add_argument("--out", default="data/test_n50_mix4.pt")
    args = ap.parse_args()

    # a testset from a broken generator is worse than no testset
    assert_matches_v4()
    assert_classify()
    make_testset(args)


if __name__ == "__main__":
    main()
