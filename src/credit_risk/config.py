"""Pydantic models that load and validate the YAML files in ``configs/``.

Every config model forbids unknown keys, so a typo in a YAML file fails loudly
instead of being silently ignored. Paths in configs are relative to the repo root.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class StrictModel(BaseModel):
    """Base for all config models: unknown keys are an error, instances are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictModel):
    """Project directories, relative to the repo root."""

    data_raw: Path
    data_interim: Path
    data_processed: Path
    models: Path
    results: Path
    reports: Path


class BaseConfig(StrictModel):
    """``configs/base.yaml``: paths and the single global seed."""

    global_seed: int = Field(ge=0)
    paths: PathsConfig


def load_config[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    """Read a YAML file and validate it against ``model``.

    Raises ``ValueError`` if the file is not a YAML mapping and
    ``pydantic.ValidationError`` if it does not match the model.
    """
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return model.model_validate(raw)


def load_base_config(config_dir: Path = CONFIG_DIR) -> BaseConfig:
    """Load ``base.yaml`` from ``config_dir``."""
    return load_config(config_dir / "base.yaml", BaseConfig)
