# Reproducing the paper's numbers

This page is the executable form of the paper's runbook. The claim under
test is narrow and checkable: *given the frozen config, the pinned
environment and the dataset, the pipeline produces the numbers in the
paper's tables.*

`scripts/reproduce_paper.py` checks exactly that, in the order a reviewer
would have to trust it.

## One command

```bash
.venv/Scripts/python.exe -m scripts.reproduce_paper
```

or, via the Makefile:

```bash
make reproduce
```

The script performs five checks and prints a verdict:

1. **Config** — `configs/experiment_6class.yaml` is loaded and validated
   through the real loader and validator (`src.quobo.config.load_config` /
   `validate_config`). A malformed config fails here, not twenty minutes into
   a run.
2. **Environment** — the interpreter version and every pinned package version
   are compared with `constraints.txt`. Drift is **reported, not fatal**: a
   developer machine is not a reproduction environment, and a hard failure
   would make the tool useless for the person trying to fix the drift. Pass
   `--strict` to turn drift into a failure.
3. **Data** — the on-disk dataset is checked with
   `src.quobo.data_validation.validate_data`, the offline contract validator.
   A dataset that has not been downloaded yet is a `SKIP` with an actionable
   message, not a failure.
4. **Run** — the experiment runs, or an existing completed run is reused
   (`--run-id <id>` or `--skip-run`). The experiment is the only step that
   cannot be dry-run.
5. **Golden comparison** — the produced summary is compared against the
   frozen paper baseline and reported as a PASS/FAIL table per arm, with
   accuracy and macro F1 side by side.

Exit codes: `0` = every check passed, `1` = a check failed (a mismatch, an
invalid config, or drift under `--strict`), `2` = the script could not run.

## The tolerance convention is shared, not re-derived

A metric passes when it sits within **±1.5 sigma of the baseline's own
standard deviation** (with a `0.01` floor when the baseline has no spread).
That convention is not written down twice: the script imports the Layer-4
harness (`tests/test_golden_results.py`) and reads `SIGMA_TOLERANCE` and the
frozen baseline from it. If the harness's tolerance changes, the script
follows automatically — the two cannot drift apart.

## Dry run (the cheap CI path)

```bash
.venv/Scripts/python.exe -m scripts.reproduce_paper --dry-run
```

The dry run performs **every check except the experiment**, so it is cheap
enough to run on every push. The dataset check defaults to `quick` (the
crop-inventory fingerprint identity check); pass `--data-check full` for the
complete contract, or `--data-check off` to skip it.

`make verify` is the CI gate built from these pieces: lint, the dry-run
reproduction, and the leakage audit.

## The leakage audit

```bash
.venv/Scripts/python.exe -m scripts.leak_audit
```

`scripts/leak_audit.py` re-derives the paper's leak-free claim from the
artifacts a run left on disk: it checks that no source photo's crops
straddle the train/test split, and that every arm of a repeat is evaluated
on the same held-out set. It never retrains, never replays a split and
never writes into the run directory, so it is safe in CI. A run without a
`run_manifest.json` marked `complete` is refused — an incomplete run is not
evidence.

Exit codes: `0` = verdict PASS, `1` = verdict not PASS, `2` = the run is
unauditable (no completed run found).

## Where the baseline comes from

The comparison resolves in this order:

1. an explicit `--golden-summary <path>`;
2. `results/tables/summary_<run>.csv` — the paper's own summary artifact.
   `results/` is gitignored, so **a fresh clone does not have it**;
3. the Layer-4 harness's frozen accuracy baseline
   (`tests/test_golden_results.py::GOLDEN_ACCURACY`).

On a fresh clone the comparison therefore falls back to (3), which carries
accuracy only. Macro F1 then has no baseline and is reported as
`NO_BASELINE` — not comparable, so neither a pass nor a failure. That is
reported honestly rather than papered over.

## The Makefile

`make` is a prerequisite in its own right and is **not** installed by
`pip install -e .`:

```bash
scoop install make        # Windows (user-level, no admin)
sudo apt-get install make # Debian/Ubuntu
brew install make        # macOS
```

Then every project verb is one target:

| Target | What it does |
| --- | --- |
| `make help` | list the targets (default) |
| `make test` | the full pytest suite |
| `make lint` | `ruff check` over `src/`, `scripts/`, `tests/` |
| `make fmt` | `ruff format` in place (rewrites files) |
| `make fmt-check` | `ruff format --check` (non-mutating) |
| `make docs` | Sphinx HTML build with `-W` (warnings are errors) |
| `make export` | ONNX export for edge inference |
| `make run` | the full ablation experiment |
| `make dashboard` | render the pollution dashboard |
| `make validate-data` | the offline data-contract validation |
| `make audit` | leakage audit of the newest complete run |
| `make verify` | CI gate: lint + dry-run reproduction + leak audit |
| `make reproduce` | reproduce the paper's numbers |
| `make build` | the continuous build (`PROFILE=quick\|full\|docs`) |
| `make clean` | remove caches and build artifacts |

Every recipe uses the venv interpreter explicitly
(`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere) so
`make` never picks up a different Python than the one the pins were
installed into.

## A note on cost

Importing `src.quobo.run_experiment` used to cost **~208 seconds** on the
development machine (measured 2026-10-20) because that module imported
TensorFlow at module scope, and that single import dominated the whole
test suite. `src/quobo/features.py` now imports TensorFlow lazily, which
brought the import down to ~8 seconds.

`scripts/reproduce_paper.py` and `tests/test_reproduce.py` therefore import
nothing heavy at module scope — the pipeline imports live inside the
functions that need them. `tests/test_reproduce.py` asserts this property,
so the cost cannot silently come back.
