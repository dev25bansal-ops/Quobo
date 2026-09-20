"""Explicit offline validation CLI. --help does not import pipeline dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, help="COCO annotation JSON")
    parser.add_argument("--images-dir", type=Path, help="Verify all images under this root")
    parser.add_argument("--provenance", type=Path, help="Pinned annotations_provenance.json")
    parser.add_argument("--crops-dir", type=Path, help="Verify all crops and class layout")
    parser.add_argument("--crop-meta", type=Path, help="data_prep_stats.json (requires crops)")
    parser.add_argument("--classes", nargs="+", help="Class names in original extractor order")
    parser.add_argument("--cache", type=Path, help="Raw embedding NPZ (not legacy PCA CSV)")
    parser.add_argument("--cache-meta", type=Path, help="Matching raw embedding metadata JSON")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1280,
        help="Expected raw width (default: MobileNetV2's 1280)",
    )
    parser.add_argument(
        "--max-cache-bytes",
        type=int,
        default=2 * 1024**3,
        help="Maximum uncompressed NPZ bytes (default: 2 GiB)",
    )
    parser.add_argument(
        "--cache-path-base",
        type=Path,
        help="Base for relative cache paths (default: current directory)",
    )
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.quobo.data_validation import validate_data

    report = validate_data(**vars(args))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
