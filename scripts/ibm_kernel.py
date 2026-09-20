"""IBM SamplerV2 fidelity harness. Default is credential-free, network-free dry run."""

from __future__ import annotations

import argparse
import getpass
import json
import warnings
from pathlib import Path

import numpy as np

from quobo.quantum_experiments import ibm_kernel, write_new_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="NPZ with pre-scaled X and optional Y; <=32 rows each"
    )
    parser.add_argument("--output", type=Path, help="new JSON file, never overwrite")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true", help="permit network/hardware submission explicitly"
    )
    mode.add_argument("--dry-run", action="store_true", help="circuits/plan only (default)")
    parser.add_argument("--backend", help="explicit IBM backend name, required to execute")
    parser.add_argument("--instance", help="explicit IBM instance CRN, required to execute")
    parser.add_argument(
        "--channel", choices=("ibm_quantum_platform",), default="ibm_quantum_platform"
    )
    parser.add_argument(
        "--prompt-token",
        action="store_true",
        help="secure interactive token prompt; no saved-account search",
    )
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument(
        "--seed", type=int, default=42, help="transpilation seed, NOT hardware randomness seed"
    )
    parser.add_argument("--reps", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--entanglement", choices=("linear", "full"), default="linear")
    parser.add_argument(
        "--mitigation",
        choices=("none", "zne"),
        default="none",
        help="zne is unsupported and errors",
    )
    args = parser.parse_args(argv)
    if args.mitigation != "none":
        parser.error(
            "SamplerV2 ZNE is unsupported by this harness; only --mitigation none is valid"
        )
    if args.execute and not all((args.input, args.backend, args.instance, args.prompt_token)):
        parser.error("--execute requires --input, --backend, --instance and --prompt-token")
    if args.prompt_token and not args.execute:
        parser.error("--prompt-token requires --execute; dry runs never read credentials")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.execute and not args.output:
        parser.error("--execute requires --output for a durable submitted-job record")
    record_path = args.output.with_suffix(args.output.suffix + ".job.json") if args.output else None
    if args.execute and record_path.exists():
        parser.error("job record exists; retrieve that job instead of submitting again")
    try:
        if args.input:
            with np.load(args.input, allow_pickle=False) as data:
                if "X" not in data:
                    raise ValueError("input NPZ requires X and optionally Y, already angle-scaled")
                X, Y = data["X"], data.get("Y", None)
        else:
            X, Y = np.array([[0.1, 0.2], [0.7, 1.1]]), None
        kwargs = {
            "backend": args.backend,
            "instance": args.instance,
            "channel": args.channel,
            "shots": args.shots,
            "seed": args.seed,
            "reps": args.reps,
            "entanglement": args.entanglement,
            "mitigation": args.mitigation,
        }
        # Validate/build the complete plan before prompting for credentials.
        result = ibm_kernel(X, Y, **kwargs)
        if args.execute:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                try:
                    token = getpass.getpass("IBM token (not saved): ")
                except getpass.GetPassWarning as exc:
                    raise ValueError(
                        "no secure (non-echoing) terminal available; refusing to read a token"
                    ) from exc

            def record_job(job_id):
                write_new_json(
                    record_path,
                    {
                        "job_id": job_id,
                        "backend": args.backend,
                        "instance": args.instance,
                        "plan": result,
                    },
                )

            result = ibm_kernel(X, Y, execute=True, token=token, on_submitted=record_job, **kwargs)
        result["synthetic"] = not bool(args.input)
        if args.output:
            write_new_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
