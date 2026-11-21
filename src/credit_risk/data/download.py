"""Hash-checked download of the raw Lending Club file (D-02).

``download(source)`` returns the path of a verified raw file in ``data/raw/``:

- **kaggle** (primary, decided in D-01): ``accepted_2007_to_2018Q4.csv.gz`` via the Kaggle
  CLI, checked against the SHA-256 and byte size in ``configs/base.yaml``. Needs a Kaggle
  API token in ``~/.kaggle/kaggle.json`` (``%USERPROFILE%\\.kaggle`` on Windows), never in
  this repo.
- **zenodo** (fallback/benchmark): streamed over HTTPS, checked against the record's md5.

A file already on disk is verified, never silently replaced: a corrupted file raises
``DataIntegrityError`` and must be deleted by hand. A fresh download goes to a ``.part``
file first and is renamed only after it verifies, so a failed or tampered download never
leaves a file under the final name.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx

from credit_risk.config import REPO_ROOT, DataConfig, KaggleSource, ZenodoSource, load_base_config

Source = Literal["kaggle", "zenodo"]
HashAlgorithm = Literal["sha256", "md5"]
Fetcher = Callable[[Path], None]
"""Writes the downloaded bytes to the given path."""

_CHUNK = 1 << 20


class DataIntegrityError(Exception):
    """A raw data file is missing or does not match its recorded hash or size."""


def file_hash(path: Path, algorithm: HashAlgorithm) -> str:
    """Hex digest of ``path``, read in 1 MiB chunks (works for multi-GB files)."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path, algorithm: HashAlgorithm, expected: str, size_bytes: int | None = None
) -> None:
    """Raise ``DataIntegrityError`` unless ``path`` exists and matches size and hash."""
    if not path.is_file():
        raise DataIntegrityError(f"{path} is missing")
    if size_bytes is not None and (actual_size := path.stat().st_size) != size_bytes:
        raise DataIntegrityError(f"{path} has {actual_size} bytes, expected {size_bytes}")
    if (actual := file_hash(path, algorithm)) != expected:
        raise DataIntegrityError(f"{path} {algorithm} is {actual}, expected {expected}")


def zenodo_url(src: ZenodoSource) -> str:
    """Direct download URL of the Zenodo record file."""
    return f"https://zenodo.org/records/{src.record}/files/{src.file}?download=1"


def _fetch_zenodo(src: ZenodoSource) -> Fetcher:
    def fetch(dest: Path) -> None:  # pragma: no cover - network
        with (
            httpx.stream("GET", zenodo_url(src), follow_redirects=True, timeout=60.0) as resp,
            dest.open("wb") as f,
        ):
            resp.raise_for_status()
            for chunk in resp.iter_bytes(_CHUNK):
                f.write(chunk)

    return fetch


def _fetch_kaggle(src: KaggleSource) -> Fetcher:
    def fetch(dest: Path) -> None:  # pragma: no cover - network + credentials
        exe = shutil.which("kaggle")
        if exe is None:
            raise RuntimeError("kaggle CLI not found: run `uv sync` (it is a dev dependency)")
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            cmd = [exe, "datasets", "download", src.dataset, "-f", src.file, "-p", tmp]
            subprocess.run(cmd, check=True)
            got = Path(tmp) / src.file
            zipped = Path(tmp) / f"{src.file}.zip"
            if not got.exists() and zipped.exists():
                with zipfile.ZipFile(zipped) as zf:
                    zf.extract(src.file, tmp)
            shutil.move(got, dest)

    return fetch


def download(
    source: Source,
    cfg: DataConfig | None = None,
    dest_dir: Path | None = None,
    fetch: Fetcher | None = None,
) -> Path:
    """Return the path of the verified raw file for ``source``, downloading it if needed.

    ``cfg`` and ``dest_dir`` default to ``configs/base.yaml`` and its ``paths.data_raw``.
    ``fetch`` replaces the network call (used by tests).
    Raises ``DataIntegrityError`` if the existing or downloaded file fails its hash check.
    """
    if cfg is None or dest_dir is None:
        base = load_base_config()
        cfg = cfg or base.data
        dest_dir = dest_dir or REPO_ROOT / base.paths.data_raw

    algorithm: HashAlgorithm
    if source == "kaggle":
        name, algorithm, expected = cfg.kaggle.file, "sha256", cfg.kaggle.sha256
        size: int | None = cfg.kaggle.size_bytes
        fetch = fetch or _fetch_kaggle(cfg.kaggle)
    else:
        name, algorithm, expected, size = cfg.zenodo.file, "md5", cfg.zenodo.md5, None
        fetch = fetch or _fetch_zenodo(cfg.zenodo)

    target = dest_dir / name
    if target.exists():
        verify_file(target, algorithm, expected, size)
        return target

    dest_dir.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f"{name}.part")
    try:
        fetch(part)
        verify_file(part, algorithm, expected, size)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)
    return target
