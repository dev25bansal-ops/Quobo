"""Pydantic-based configuration schema with validation and IDE autocomplete.

Wraps the YAML loading in config.py: load_config() still returns a plain dict
(backward compatible), but validate_config() enforces the full schema and
raises with a readable error on any malformed key, out-of-range value, or
unknown arm name — catching config typos at load time instead of at rep 12.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Dataset configuration."""
    classes: list[str] = Field(min_length=2, description="Class names for classification")
    min_images_per_class: int = Field(ge=10, description="Minimum crops per class")
    crops_dir_override: str | None = Field(
        default=None,
        description="Override default crops directory (for cross-dataset replication)"
    )

    @field_validator("classes")
    @classmethod
    def validate_classes_unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("classes must be unique")
        return v


class FeaturesConfig(BaseModel):
    """Feature extraction configuration."""
    backbone: Literal["MobileNetV2"] = Field(
        default="MobileNetV2",
        description="CNN backbone for embedding extraction"
    )
    pca_components: int = Field(ge=10, le=200, description="Number of PCA components")


class SelectionConfig(BaseModel):
    """Feature selection configuration."""
    k: int = Field(ge=2, le=50, description="Number of features to select")


class QUBOConfig(BaseModel):
    """QUBO solver configuration."""
    mi_bins: int = Field(ge=4, le=32, description="Number of bins for MI estimation")
    sa_sweeps: int = Field(ge=1000, description="Simulated annealing sweeps")
    sa_repeats: int = Field(ge=5, le=50, description="SA reads per alpha evaluation")
    parallel: bool = Field(
        default=False,
        description="Parallel coarse alpha scan (threaded SA; ~3x on multi-core)"
    )


class QSVMConfig(BaseModel):
    """Quantum SVM configuration."""
    reps: int = Field(ge=1, le=5, description="ZZFeatureMap repetitions")
    entanglement: Literal["linear", "full"] = Field(
        default="linear",
        description="Entanglement strategy"
    )
    max_train_samples: int = Field(ge=50, description="Max training samples per repeat")
    test_fraction: float = Field(ge=0.1, le=0.5, description="Test set fraction")
    tune_rbf: bool = Field(default=True, description="Include tuned RBF-SVM baseline")
    max_qsvm_features: int = Field(
        default=20,
        description="Skip QSVM above this feature count (statevector 2^k limit)"
    )


class ExperimentConfig(BaseModel):
    """Experiment orchestration configuration."""
    n_repeats: int = Field(ge=5, le=100, description="Number of repeated splits")
    arms: list[Literal[
        "A_random", "B_pca", "C_mi", "D_qubo",
        "E_lasso", "F_mrmr", "Z_full"
    ]] = Field(min_length=1, description="Selection arms to run")
    early_stop: bool = Field(
        default=False,
        description="Enable statistical early stopping"
    )

    @field_validator("arms")
    @classmethod
    def validate_arms_unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("arms must be unique")
        return v


class QuoboConfig(BaseModel):
    """Root configuration schema."""
    seed: int = Field(ge=0, description="Random seed")
    data: DataConfig
    features: FeaturesConfig
    selection: SelectionConfig
    qubo: QUBOConfig
    qsvm: QSVMConfig
    experiment: ExperimentConfig
