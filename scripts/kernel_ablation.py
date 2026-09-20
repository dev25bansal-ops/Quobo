"""Plan/resume capped fixed-ZZ ablations; never launch a full sweep by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quobo.quantum_experiments import ablation_grid, load_split, run_ablation, smoke_split


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="group-disjoint train/test split NPZ")
    source.add_argument("--smoke", action="store_true", help="execute one tiny synthetic cell")
    parser.add_argument(
        "--output", type=Path, required=True, help="directory for keyed manifests/results"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true", help="execute at most --max-runs missing cells"
    )
    mode.add_argument("--dry-run", action="store_true", help="manifest only (default)")
    parser.add_argument("--reps", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument(
        "--entanglement", nargs="+", choices=("linear", "full"), default=["linear", "full"]
    )
    parser.add_argument(
        "--budget", type=int, nargs="+", choices=(4, 6, 8, 12), default=[4, 6, 8, 12]
    )
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=48)
    parser.add_argument("--max-test", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    execute = (args.execute or args.smoke) and not args.dry_run
    if execute and not (args.input or args.smoke):
        parser.error("execution requires --input or --smoke")
    try:
        split = load_split(args.input) if args.input else smoke_split(args.seed)
        configs = ablation_grid(args.reps, args.entanglement, args.budget)
        result = run_ablation(
            split,
            args.output,
            configs=configs,
            seed=args.seed,
            max_train=min(args.max_train, 12) if args.smoke else args.max_train,
            max_test=min(args.max_test, 6) if args.smoke else args.max_test,
            dry_run=not execute,
            max_runs=1 if args.smoke else args.max_runs,
        )
        result["synthetic"] = not bool(args.input)
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
