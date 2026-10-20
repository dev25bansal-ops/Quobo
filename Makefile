# Quobo — project verbs as make targets.
#
# PREREQUISITE (not in the repo's own setup docs until now): `make` itself.
# GNU Make is NOT a Python package and is not installed by `pip install -e .`.
# On Windows the least invasive install is a user-level one:
#
#     scoop install make          # -> %USERPROFILE%\scoop\shims\make.exe
#
# On Debian/Ubuntu: `sudo apt-get install make`. On macOS: `xcode-select --install`
# (or `brew install make`). Verify with `make --version` (expect GNU Make >= 4.0).
# Every recipe below therefore shells out to a POSIX-ish shell on Windows
# (Git Bash / MSYS2) and uses forward slashes throughout.
#
# COST: the targets are NOT all cheap. Importing `src.quobo.run_experiment`
# used to cost ~208s on this machine (measured 2026-10-20) because that module
# imported TensorFlow at module scope; `src/quobo/features.py` now imports TF
# lazily, which cut the import to ~8s. Treat the import-heavy targets
# (`test`, `build`, `run`, `export`) as minutes-scale, and the cheap ones
# (`help`, `lint`, `fmt-check`, `reproduce`) as seconds-scale. `reproduce`
# with `--dry-run` is the cheap CI-friendly path: it runs every check except
# the experiment itself.
#
# Every recipe uses the venv interpreter explicitly (.venv/Scripts/python.exe
# on Windows, .venv/bin/python elsewhere) so `make` never picks up a different
# Python than the one the paper's pins were installed into.

# The venv interpreter. Override with `make PYTHON=/path/to/python test`.
ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif

# Build profile for the `build` target: quick | full | docs.
PROFILE ?= full

# The frozen paper config. Every target that needs a config uses this one.
CONFIG ?= configs/experiment_6class.yaml

.DEFAULT_GOAL := help
.PHONY: help test test-file lint fmt fmt-check docs export run dashboard \
        validate-data audit verify reproduce build clean

## help: list the available targets (default target)
help:
	@echo "Quobo — available targets:"
	@echo "  make test           run the full pytest suite (minutes-scale)"
	@echo "  make test-file F=tests/test_reproduce.py   run one test file"
	@echo "  make lint           ruff check (advisory, matches the pre-commit hook)"
	@echo "  make fmt            ruff format in place (REWRITES files)"
	@echo "  make fmt-check      ruff format --check (non-mutating)"
	@echo "  make docs           build the Sphinx API reference with -W"
	@echo "  make export         export the ONNX edge-inference bundle"
	@echo "  make run            run the full ablation experiment"
	@echo "  make dashboard      render the pollution dashboard HTML"
	@echo "  make validate-data  validate the on-disk data contract (offline)"
	@echo "  make audit          audit the newest complete run for leakage"
	@echo "  make verify         CI gate: lint + dry-run reproduction + leak audit"
	@echo "  make reproduce      reproduce the paper's numbers (see below)"
	@echo "  make build          continuous build (PROFILE=$(PROFILE): quick|full|docs)"
	@echo "  make clean          remove caches and build artifacts"
	@echo ""
	@echo "Variables: PYTHON=$(PYTHON)  CONFIG=$(CONFIG)  PROFILE=$(PROFILE)"

## test: run the whole test suite
test:
	$(PYTHON) -m pytest tests/ -q --tb=short

## test-file: run a single test file (F=tests/test_foo.py)
test-file:
	$(PYTHON) -m pytest $(F) -q --tb=short

## lint: ruff check over the source, scripts and tests
lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/

## fmt: reformat with ruff (REWRITES files in place)
fmt:
	$(PYTHON) -m ruff format src/ scripts/ tests/

## fmt-check: verify formatting without writing anything
fmt-check:
	$(PYTHON) -m ruff format --check src/ scripts/ tests/

## docs: build the Sphinx HTML reference. -W turns warnings into errors,
## matching CI, so a broken autodoc import cannot pass locally.
docs:
	$(PYTHON) -m sphinx -W -b html docs docs/_build/html

## export: export the ONNX bundle for edge inference
export:
	$(PYTHON) -m scripts.export_onnx --config $(CONFIG)

## run: run the full ablation experiment with the frozen paper config
run:
	$(PYTHON) -m src.quobo.run_experiment $(CONFIG)

## dashboard: render the pollution dashboard (HTML + zone table)
dashboard:
	$(PYTHON) -m scripts.pollution_dashboard

## validate-data: check the on-disk data contract. Offline, read-only, and
## dependency-light: it never loads a model and never unpickles an array.
validate-data:
	$(PYTHON) -m scripts.validate_data \
	  --crops-dir data/processed/crops \
	  --crop-meta data/processed/data_prep_stats.json \
	  --classes bottle can cigarette carton cup lid \
	  --cache data/features/raw_bottle_can_carton_cigarette_cup_lid.npz \
	  --cache-meta data/features/raw_bottle_can_carton_cigarette_cup_lid_meta.json

## audit: audit the newest complete run for train/test leakage.
## Read-only and offline: it never retrains and never replays a split, so it is
## safe in CI. Exit 0 = PASS, 1 = verdict not PASS, 2 = the run is unauditable.
audit:
	$(PYTHON) -m scripts.leak_audit

## verify: the CI gate. Every step is cheap and offline, so it is safe to run
## on every push: lint, the dry-run reproduction (all checks but the
## experiment), and the leakage audit of the newest complete run.
verify: lint
	$(PYTHON) -m scripts.reproduce_paper --dry-run
	$(PYTHON) -m scripts.leak_audit

## reproduce: one command from a fresh clone to the paper's numbers.
## Loads and validates the frozen config, reports environment drift, checks the
## dataset, runs (or reuses) the experiment, and asserts the produced summary
## against the frozen golden baseline. Use `--dry-run` for every check except
## the experiment; pass extra flags after the target, e.g.
##   make reproduce ARGS=--dry-run
##   make reproduce ARGS="--run-id 20260920_161627"
reproduce:
	$(PYTHON) -m scripts.reproduce_paper $(ARGS)

## build: the continuous build (the same core the 24/7 wrappers call).
## PROFILE=quick runs everything except the integration tests; PROFILE=docs
## additionally builds the Sphinx reference.
build:
	$(PYTHON) -m scripts.build_core --profile $(PROFILE)

## clean: remove caches and build artifacts. Deliberately does NOT touch
## experiments/ or results/ — those are the paper's evidence, not build output.
## The sweep is limited to the source trees on purpose: a repo-wide rglob
## would descend into .venv and spend minutes deleting the interpreter's own
## bytecode caches.
clean:
	$(PYTHON) -c "import pathlib, shutil; \
	roots = [pathlib.Path(p) for p in ('src', 'scripts', 'tests', 'docs')]; \
	dirs = [d for r in roots if r.is_dir() for d in r.rglob('__pycache__')]; \
	dirs += [pathlib.Path(p) for p in ('docs/_build', '.pytest_cache', '.ruff_cache')]; \
	[shutil.rmtree(d, ignore_errors=True) for d in dirs]; \
	print(f'cleaned {len(dirs)} cache dir(s)')"
