"""Pydantic-based configuration schema with validation and IDE autocomplete.

Wraps the YAML loading in config.py: load_config() still returns a plain dict
(backward compatible), but validate_config() enforces the full schema and
raises with a readable error on any malformed key, out-of-range value,
unknown arm name or unknown nested key (Q17) — catching config typos at load
time instead of at rep 12. Cross-field constraints (Q19) are enforced too.

``PaperConfig`` additionally freezes the published run's configuration
(catalog Part 2): the paper's numbers are only the paper's numbers if the
config that produced them is the config that is shipped, so the frozen
values are enforced rather than merely described in the YAML comment. A
config that drifted from the paper is reported field by field, so the
answer to "is this still the paper's run?" is a list of drifted fields
rather than a single "invalid config".
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class _StrictModel(BaseModel):
    """Base for every config section: unknown keys are rejected, not ignored.

    Q17 — a mistyped key (e.g. ``qsvm.repz``) must fail loudly instead of
    silently falling back to a default.
    """

    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictModel):
    """Dataset configuration."""

    classes: list[str] = Field(min_length=2, description="Class names for classification")
    min_images_per_class: int = Field(ge=10, description="Minimum crops per class")
    crops_dir_override: str | None = Field(
        default=None, description="Override default crops directory (for cross-dataset replication)"
    )

    @field_validator("classes")
    @classmethod
    def validate_classes_unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("classes must be unique")
        # A class name becomes a DIRECTORY under the crops root, and a class
        # below min_images_per_class has that directory removed. A name with a
        # path separator or a dot-segment would therefore point outside the
        # crops root (a class named ".." resolves to its parent). Names must
        # also be portable filenames: a space, colon or other separator is
        # invalid in a Windows path and would fail at rmtree time.
        for name in v:
            if (
                not name
                or name in (".", "..")
                or "/" in name
                or "\\" in name
                or os.sep in name
                or (os.altsep and os.altsep in name)
                or any(c in name for c in '<>:"|?*')
                or name != name.strip()
            ):
                raise ValueError(f"class names must be plain directory names, got {name!r}")
        return v


class FeaturesConfig(_StrictModel):
    """Feature extraction configuration."""

    backbone: Literal["MobileNetV2"] = Field(
        default="MobileNetV2", description="CNN backbone for embedding extraction"
    )
    pca_components: int = Field(ge=10, le=200, description="Number of PCA components")


class SelectionConfig(_StrictModel):
    """Feature selection configuration."""

    k: int = Field(ge=2, le=50, description="Number of features to select")


class QUBOConfig(_StrictModel):
    """QUBO solver configuration."""

    mi_bins: int = Field(ge=4, le=32, description="Number of bins for MI estimation")
    sa_sweeps: int = Field(ge=1000, description="Simulated annealing sweeps")
    sa_repeats: int = Field(ge=5, le=50, description="SA reads per alpha evaluation")
    parallel: bool = Field(
        default=False, description="Parallel coarse alpha scan (threaded SA; ~3x on multi-core)"
    )


class QSVMConfig(_StrictModel):
    """Quantum SVM configuration."""

    reps: int = Field(ge=1, le=5, description="ZZFeatureMap repetitions")
    entanglement: Literal["linear", "full"] = Field(
        default="linear", description="Entanglement strategy"
    )
    max_train_samples: int = Field(ge=50, description="Max training samples per repeat")
    test_fraction: float = Field(ge=0.1, le=0.5, description="Test set fraction")
    tune_rbf: bool = Field(default=True, description="Include tuned RBF-SVM baseline")
    max_qsvm_features: int = Field(
        default=20,
        ge=1,
        le=30,
        description="Skip QSVM above this feature count (statevector 2^k limit)",
    )


class ExperimentConfig(_StrictModel):
    """Experiment orchestration configuration."""

    n_repeats: int = Field(
        ge=5,
        le=100,
        description=(
            "Number of repeated splits. NOTE: the two-sided Wilcoxon used for "
            "significance has a minimum attainable p-value of 0.0625 at n=5 — "
            "above the 0.05 threshold — so a 5-repeat run can never report a "
            "significant result. Use >= 6 for a meaningful test; the paper run "
            "uses 25."
        ),
    )
    arms: list[Literal["A_random", "B_pca", "C_mi", "D_qubo", "E_lasso", "F_mrmr", "Z_full"]] = (
        Field(min_length=1, description="Selection arms to run")
    )
    early_stop: bool = Field(default=False, description="Enable statistical early stopping")
    track_mlflow: bool = Field(
        default=False,
        description="Log the run to MLflow (local file store; needs the mlflow extra)",
    )
    # The former 'baselines' key was removed: it was never read by any code
    # (the RBF-SVM baseline is mandatory per the Huang 2021 caveat and is
    # gated by qsvm.tune_rbf), so setting it had no effect at all. A config
    # key that does nothing is worse than no key.

    @field_validator("arms")
    @classmethod
    def validate_arms_unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("arms must be unique")
        return v


class QuoboConfig(_StrictModel):
    """Root configuration schema."""

    seed: int = Field(ge=0, description="Random seed")
    data: DataConfig
    features: FeaturesConfig
    selection: SelectionConfig
    qubo: QUBOConfig
    qsvm: QSVMConfig
    experiment: ExperimentConfig

    @model_validator(mode="after")
    def validate_cross_field(self):
        """Q19 — the selection budget cannot exceed the candidate PCA pool.

        Selecting k features from a pool of fewer than k columns is not
        executable; fail before any extraction work rather than at rep 12.
        """
        if self.selection.k > self.features.pca_components:
            raise ValueError(
                f"selection.k ({self.selection.k}) must be <= "
                f"features.pca_components ({self.features.pca_components})"
            )
        # The QSVM is skipped (with only a log warning) when k exceeds the
        # statevector memory guard, so a config above the limit would run to
        # completion with ZERO quantum results while still reporting
        # status="complete" — a silent loss of the paper's quantum arm.
        if self.selection.k > self.qsvm.max_qsvm_features:
            raise ValueError(
                f"selection.k ({self.selection.k}) exceeds "
                f"qsvm.max_qsvm_features ({self.qsvm.max_qsvm_features}); "
                "the QSVM would be silently skipped for every arm, producing a "
                "run with no quantum results"
            )
        # Early stopping needs enough repeats to be meaningful: the runner
        # only consults it from rep 9 and requires min_reps=10, so a shorter
        # run would ignore the flag entirely while appearing to enable it.
        if self.experiment.early_stop and self.experiment.n_repeats < 10:
            raise ValueError(
                f"experiment.early_stop requires n_repeats >= 10 (got "
                f"{self.experiment.n_repeats}); the early-stop check only runs "
                "from repeat 10 onward, so a shorter run would ignore the flag"
            )
        return self


# The published run's frozen values. The paper's numbers are only the paper's
# numbers if the config that produced them is the config that is shipped, so
# the values are enforced by the schema below rather than merely described in
# a YAML comment (catalog Part 2: "the published configuration is enforced,
# not just described"). The class list and the arm list are ordered: the
# published tables index them by position, so a reordering is a different run.
PAPER_CLASSES: tuple[str, ...] = ("bottle", "can", "cigarette", "carton", "cup", "lid")
PAPER_ARMS: tuple[str, ...] = ("A_random", "B_pca", "C_mi", "D_qubo", "E_lasso", "F_mrmr", "Z_full")


class PaperDataConfig(DataConfig):
    """``data`` section frozen to the published run's values.

    Every leaf defaults to the paper's value, so a config that only states a
    deviation still validates and the deviation is what gets reported.
    """

    classes: list[str] = Field(
        default_factory=lambda: list(PAPER_CLASSES),
        min_length=2,
        description="Class names for classification",
    )
    min_images_per_class: Literal[40] = 40
    # The paper ran on the shipped TACO crops. An override points the run at
    # a different dataset, which is a different experiment, not a different
    # spelling of the paper's.
    crops_dir_override: Literal[None] = None

    @field_validator("classes")
    @classmethod
    def validate_paper_classes(cls, v: list[str]) -> list[str]:
        if tuple(v) != PAPER_CLASSES:
            raise ValueError(f"paper classes are {list(PAPER_CLASSES)}, got {v}")
        return v


class PaperFeaturesConfig(FeaturesConfig):
    """``features`` section frozen to the published run's values."""

    backbone: Literal["MobileNetV2"] = "MobileNetV2"
    pca_components: Literal[50] = 50


class PaperSelectionConfig(SelectionConfig):
    """``selection`` section frozen to the published run's values."""

    k: Literal[8] = 8


class PaperQUBOConfig(QUBOConfig):
    """``qubo`` section frozen to the published run's values.

    ``parallel`` is deliberately left free: the threaded coarse alpha scan
    selects the same features (verified on the real data), so it changes how
    long the run takes, not what it measures.
    """

    mi_bins: Literal[16] = 16
    sa_sweeps: Literal[20000] = 20000
    sa_repeats: Literal[10] = 10


class PaperQSVMConfig(QSVMConfig):
    """``qsvm`` section frozen to the published run's values."""

    reps: Literal[1] = 1
    entanglement: Literal["linear"] = "linear"
    max_train_samples: Literal[400] = 400
    test_fraction: Literal[0.25] = 0.25
    tune_rbf: Literal[True] = True
    max_qsvm_features: Literal[20] = 20


class PaperExperimentConfig(ExperimentConfig):
    """``experiment`` section frozen to the published run's values.

    ``track_mlflow`` is deliberately left free: it only decides whether the
    run is logged, not what is measured. ``early_stop`` is frozen because it
    changes how many repeats the published statistics are computed from.
    """

    # Deliberately wider than the parent's Literal list: a pydantic literal
    # error would say "input should be 'A_random', 'B_pca', ...", which reads
    # like a menu rather than like "this is not the arm set the paper ran".
    # The validator below rejects anything that is not the published list,
    # in the published order, and says which one it got.
    arms: list[str] = Field(
        default_factory=lambda: list(PAPER_ARMS),
        min_length=1,
        description="Selection arms to run",
    )
    n_repeats: Literal[25] = 25
    early_stop: Literal[False] = False

    @field_validator("arms")
    @classmethod
    def validate_paper_arms(cls, v: list[str]) -> list[str]:
        if tuple(v) != PAPER_ARMS:
            raise ValueError(f"paper arms are {list(PAPER_ARMS)}, got {v}")
        return v


class PaperConfig(QuoboConfig):
    """The published run's configuration, frozen.

    Same shape as :class:`QuoboConfig` — so a paper config is a valid
    experiment config and the unchanged pipeline can run it — but every value
    that shapes the published numbers is pinned. A config that drifted from
    the paper is rejected with the offending field named, because "invalid
    config" does not tell a reader which of a hundred numbers stopped being
    the paper's.

    The six sections are required: a config missing a section is not the
    paper's config, and letting a section default to the paper's values
    would report that omission as a pass.
    """

    seed: Literal[42] = 42
    data: PaperDataConfig
    features: PaperFeaturesConfig
    selection: PaperSelectionConfig
    qubo: PaperQUBOConfig
    qsvm: PaperQSVMConfig
    experiment: PaperExperimentConfig


def _describe_error(err: dict) -> str:
    """Render one pydantic error as ``field: expected X, got Y``."""
    loc = ".".join(str(part) for part in err.get("loc", ())) or "config"
    got = err.get("input")
    msg = str(err.get("msg", "invalid"))
    if err.get("type") == "literal_error":
        # pydantic states the accepted value in the message ("Input should be
        # 8", "Input should be 'MobileNetV2'") but not the value it got, and
        # its ctx["expected"] is the *string* form, so the message is the
        # only place the expected value is rendered in its own type.
        expected = msg.removeprefix("Input should be ")
        return f"{loc}: expected {expected}, got {got!r}"
    if msg.startswith("Value error, "):
        msg = msg[len("Value error, ") :]
    return f"{loc}: {msg}"


def paper_drift(cfg: dict) -> list[str]:
    """Return the frozen paper fields that ``cfg`` does not match.

    An empty list means the config is the paper's config. Each entry names
    the field path and both values, e.g. ``"selection.k: expected 8, got
    10"``, so a drifted config says what moved instead of only that it is
    invalid.
    """
    try:
        PaperConfig.model_validate(cfg)
    except ValidationError as e:
        return [_describe_error(err) for err in e.errors()]
    return []


def validate_paper_config(cfg: dict) -> PaperConfig:
    """Enforce the frozen paper configuration.

    Raises ValueError listing every drifted field, so the reason a config is
    not the paper's config is legible without reading a pydantic traceback.
    """
    drifts = paper_drift(cfg)
    if drifts:
        raise ValueError("paper config drift: " + "; ".join(drifts))
    return PaperConfig.model_validate(cfg)


def load_paper_config(path: str | Path = "configs/experiment_6class.yaml") -> PaperConfig:
    """Load a YAML config and check it against the frozen paper configuration.

    Mirrors config.load_config/validate_config, but for the published run:
    the returned model is a PaperConfig, so the caller cannot accidentally
    treat a drifted config as the paper's. The import is local because
    config.py imports this module — a module-level back-import would be
    circular.
    """
    from .config import load_config

    return validate_paper_config(load_config(path))
