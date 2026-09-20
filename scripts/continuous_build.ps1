<#
.SYNOPSIS
  Quobo continuous build — 24/7 quality gate with Obsidian vault reporting.
.DESCRIPTION
  M-10: thin wrapper around the shared Python core (scripts/build_core.py), so
  this Windows runner and the POSIX shim (scripts/continuous_build.sh) / ubuntu
  CI execute identical logic. Results land in three places:
    1. outputs/ci/last_build.json   (machine-readable, latest)
    2. outputs/ci/build_history.csv (append-only history)
    3. vault/50 Build-Ops/          (Build Status.md block + Build Log.md row)
  Tests are the gate; ruff is advisory. Exit 0 = green, 1 = failed build.
.PROFILE
  quick : ruff + pytest -m "not integration" + config-schema import check
  full  : ruff + pytest (ALL, incl. integration) + schema check   [default]
  docs  : full + Sphinx HTML build
.EXAMPLE
  pwsh scripts/continuous_build.ps1 -Profile full
#>
[CmdletBinding()]
param(
  [ValidateSet('quick', 'full', 'docs')] [string]$Profile = 'full',
  [string]$Python = '',
  [string]$RepoRoot = ''
)

$ErrorActionPreference = 'Stop'
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
Set-Location $RepoRoot

if (-not $Python) {
  $cand = Join-Path $RepoRoot '.venv\Scripts\python.exe'
  if (Test-Path $cand) { $Python = $cand } else { $Python = 'python' }
}

& $Python -m scripts.build_core --profile $Profile
exit $LASTEXITCODE
