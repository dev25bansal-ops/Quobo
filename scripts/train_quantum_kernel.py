"""CLI for bounded train-only target-alignment learning (no remote execution)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quobo.quantum_experiments import (
    IMPLEMENTATION_VERSION,
    load_split,
    package_versions,
    smoke_split,
    train_kernel_experiment,
    write_new_json,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="pickle-free group-disjoint split NPZ")
    source.add_argument("--smoke", action="store_true", help="run tiny synthetic 2-iteration test")
    parser.add_argument(
        "--output", type=Path, help="new JSON file; existing files are never overwritten"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="execute local statevector training")
    mode.add_argument("--dry-run", action="store_true", help="plan only (default without --smoke)")
    parser.add_argument("--budget", type=int, choices=range(2, 13), default=4)
    parser.add_argument("--reps", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--entanglement", choices=("linear", "full"), default="linear")
    parser.add_argument("--iterations", type=int, default=10, help="SPSA steps, bounded 0..100")
    parser.add_argument("--max-train", type=int, default=48, help="cap, bounded 2..128")
    parser.add_argument("--max-test", type=int, default=64, help="cap, bounded 1..256")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    execute = (args.execute or args.smoke) and not args.dry_run
    if execute and not (args.input or args.smoke):
        parser.error("execution requires --input or --smoke")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new path")
    if (
        not 0 <= args.iterations <= 100
        or not 2 <= args.max_train <= 128
        or not 1 <= args.max_test <= 256
    ):
        parser.error("iterations/caps out of bounds; see --help")
    kwargs = {
        "budget": args.budget,
        "reps": args.reps,
        "entanglement": args.entanglement,
        "seed": args.seed,
        "iterations": min(args.iterations, 2) if args.smoke else args.iterations,
        "max_train": min(args.max_train, 12) if args.smoke else args.max_train,
        "max_test": min(args.max_test, 6) if args.smoke else args.max_test,
    }
    try:
        split = load_split(args.input) if args.input else smoke_split(args.seed)
        result = {
            "implementation": IMPLEMENTATION_VERSION,
            "dry_run": not execute,
            "synthetic": not bool(args.input),
            "parameters": kwargs,
            "split_sha256": split.fingerprint(),
            "packages": package_versions(),
            "planned_kernel_evaluations_for_training": 1 + 3 * kwargs["iterations"],
        }
        if args.budget > split.X_train.shape[1]:
            raise ValueError("budget exceeds input width")
        if execute:
            result["result"] = train_kernel_experiment(split, **kwargs)
        if args.output:
            write_new_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
