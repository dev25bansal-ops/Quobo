"""Pydantic-based configuration schema with validation and IDE autocomplete.

Wraps the YAML loading in config.py: load_config() still returns a plain dict
(backward compatible), but validate_config() enforces the full schema and
raises with a readable error on any malformed key, out-of-range value,
unknown arm name or unknown nested key (Q17) — catching config typos at load
time instead of at rep 12. Cross-field constraints (Q19) are enforced too.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
