"""Zenodo fetch, md5 verification and extraction of the MAUD v1 dataset.

This is the only M0 module that touches the network, and it is written to
be skippable: if `data/raw/data/` already contains 152 contract files and
all three MAUD CSVs, it is treated as a valid, already-extracted dataset and
nothing is downloaded (spec: "data/raw/ may already hold the extracted
dataset; verify the md5/file count ... and reuse it if valid instead of
re-downloading").
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile

from dealpoint.config import (
    CONTRACTS_DIR,
    CSV_PATHS,
    N_CONTRACTS,
    RAW_DIR,
    RAW_ZIP_PATH,
    ZENODO_URL,
    ZIP_BYTES,
    ZIP_MD5,
)

_DOWNLOAD_CHUNK_BYTES = 1 << 20


def is_extracted_dataset_valid() -> bool:
    """True if data/raw/data already holds a complete, valid extraction."""
    if not CONTRACTS_DIR.is_dir():
        return False
    contract_files = list(CONTRACTS_DIR.glob("contract_*.txt"))
    if len(contract_files) != N_CONTRACTS:
        return False
    return all(path.exists() for path in CSV_PATHS.values())


def _md5_of(path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_zip() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(ZENODO_URL, RAW_ZIP_PATH)


def _verify_zip() -> None:
    size = RAW_ZIP_PATH.stat().st_size
    if size != ZIP_BYTES:
        raise AssertionError(f"downloaded zip is {size} bytes, expected {ZIP_BYTES}")
    md5 = _md5_of(RAW_ZIP_PATH)
    if md5 != ZIP_MD5:
        raise AssertionError(f"downloaded zip md5 {md5} != expected {ZIP_MD5}")


def _extract_zip() -> None:
    with zipfile.ZipFile(RAW_ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)


def ensure_dataset(force_download: bool = False, skip_download: bool = False) -> None:
    """Ensure the MAUD dataset is present and valid under data/raw/.

    - If already extracted and valid and not `force_download`: no-op.
    - If `skip_download`: raise if not already valid (caller asked us not to
      hit the network).
    - Otherwise: download the zip, verify size + md5, extract, then delete
      the zip (spec: never keep the zip around after extraction).
    """
    if not force_download and is_extracted_dataset_valid():
        return
    if skip_download:
        raise RuntimeError(
            "dataset not present/valid under data/raw/ and --skip-download was given"
        )
    _download_zip()
    _verify_zip()
    _extract_zip()
    RAW_ZIP_PATH.unlink(missing_ok=True)
    if not is_extracted_dataset_valid():
        raise AssertionError("extraction completed but dataset still fails validation")
