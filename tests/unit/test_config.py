"""Step 0.3 (E-02): YAML configs are loaded and validated by pydantic models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_risk.config import CONFIG_DIR, BaseConfig, load_base_config, load_config

VALID_BASE = """
global_seed: 42
paths:
  data_raw: data/raw
  data_interim: data/interim
  data_processed: data/processed
  models: models
  results: results
  reports: reports
"""


def _write(tmp_path: Path, text: str, name: str = "base.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_returns_validated_model(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID_BASE), BaseConfig)
    assert cfg.global_seed == 42
    assert cfg.paths.results == Path("results")


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, VALID_BASE + "typo_key: 1\n"), BaseConfig)


def test_load_config_rejects_negative_seed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, VALID_BASE.replace("42", "-1")), BaseConfig)


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_config(_write(tmp_path, "- just\n- a list\n"), BaseConfig)


def test_repo_base_yaml_validates_with_global_seed_42() -> None:
    cfg = load_base_config()
    assert cfg.global_seed == 42
    assert (CONFIG_DIR / "base.yaml").is_file()
