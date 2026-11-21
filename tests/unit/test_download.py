"""Step 1.1 (D-02): downloads are hash-checked; a mismatch raises DataIntegrityError."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from credit_risk.config import DataConfig, load_base_config
from credit_risk.data.download import (
    DataIntegrityError,
    download,
    file_hash,
    verify_file,
    zenodo_url,
)

PAYLOAD = b"id,loan_amnt\n1,1000\n2,2500\n"


def _data_cfg(payload: bytes = PAYLOAD) -> DataConfig:
    """A DataConfig whose Kaggle and Zenodo entries both expect ``payload``."""
    return DataConfig.model_validate(
        {
            "source": "kaggle",
            "kaggle": {
                "dataset": "owner/name",
                "version": 1,
                "file": "loans.csv.gz",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "zenodo": {
                "record": 123,
                "file": "loans.csv",
                "md5": hashlib.md5(payload).hexdigest(),
            },
        }
    )


def _writer(payload: bytes, calls: list[Path]):  # type: ignore[no-untyped-def]
    def fetch(dest: Path) -> None:
        calls.append(dest)
        dest.write_bytes(payload)

    return fetch


def test_file_hash_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD)
    assert file_hash(path, "sha256") == hashlib.sha256(PAYLOAD).hexdigest()
    assert file_hash(path, "md5") == hashlib.md5(PAYLOAD).hexdigest()


def test_verify_file_passes_on_matching_hash(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD)
    verify_file(path, "sha256", hashlib.sha256(PAYLOAD).hexdigest(), size_bytes=len(PAYLOAD))


def test_verify_file_raises_on_corrupted_file(tmp_path: Path) -> None:
    """Acceptance test of D-02: one flipped byte is caught."""
    path = tmp_path / "f.bin"
    expected = hashlib.sha256(PAYLOAD).hexdigest()
    corrupted = bytearray(PAYLOAD)
    corrupted[5] ^= 0xFF
    path.write_bytes(bytes(corrupted))
    with pytest.raises(DataIntegrityError, match="sha256"):
        verify_file(path, "sha256", expected)


def test_verify_file_raises_on_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD)
    with pytest.raises(DataIntegrityError, match="bytes"):
        verify_file(path, "sha256", hashlib.sha256(PAYLOAD).hexdigest(), size_bytes=1)


def test_verify_file_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError, match="missing"):
        verify_file(tmp_path / "nope.bin", "md5", "0" * 32)


@pytest.mark.parametrize("source", ["kaggle", "zenodo"])
def test_download_returns_existing_verified_file_without_fetching(
    tmp_path: Path, source: str
) -> None:
    cfg = _data_cfg()
    name = cfg.kaggle.file if source == "kaggle" else cfg.zenodo.file
    (tmp_path / name).write_bytes(PAYLOAD)
    calls: list[Path] = []
    out = download(source, cfg, tmp_path, fetch=_writer(PAYLOAD, calls))  # type: ignore[arg-type]
    assert out == tmp_path / name
    assert calls == []


def test_download_raises_if_existing_file_is_corrupted(tmp_path: Path) -> None:
    cfg = _data_cfg()
    (tmp_path / cfg.kaggle.file).write_bytes(PAYLOAD + b"tampered")
    calls: list[Path] = []
    with pytest.raises(DataIntegrityError):
        download("kaggle", cfg, tmp_path, fetch=_writer(PAYLOAD, calls))
    assert calls == []


@pytest.mark.parametrize("source", ["kaggle", "zenodo"])
def test_download_fetches_missing_file_and_verifies(tmp_path: Path, source: str) -> None:
    cfg = _data_cfg()
    calls: list[Path] = []
    out = download(source, cfg, tmp_path / "raw", fetch=_writer(PAYLOAD, calls))  # type: ignore[arg-type]
    assert out.read_bytes() == PAYLOAD
    assert len(calls) == 1
    assert list(out.parent.iterdir()) == [out]


def test_download_discards_fetched_file_with_wrong_hash(tmp_path: Path) -> None:
    cfg = _data_cfg()
    calls: list[Path] = []
    with pytest.raises(DataIntegrityError):
        download("zenodo", cfg, tmp_path, fetch=_writer(b"not the real file", calls))
    assert list(tmp_path.iterdir()) == []


def test_zenodo_url_points_at_record_file() -> None:
    cfg = _data_cfg()
    assert zenodo_url(cfg.zenodo) == "https://zenodo.org/records/123/files/loans.csv?download=1"


def test_repo_config_names_both_source_files() -> None:
    data = load_base_config().data
    assert data.kaggle.file == "accepted_2007_to_2018Q4.csv.gz"
    assert data.zenodo.file == "LC_loans_granting_model_dataset.csv"
