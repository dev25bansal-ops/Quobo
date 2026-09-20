#!/usr/bin/env bash
# M-10: POSIX shim — same quality gate as continuous_build.ps1, via the shared
# Python core, so local (Windows) and CI (ubuntu) runners execute identical logic.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-full}"
exec python -m scripts.build_core --profile "$PROFILE"
