"""Installed framework versions, recorded in every M7a report + Braintrust
experiment metadata (spec: "framework_versions" in the common metadata list).

Pure `importlib.metadata` lookups -- never imports the packages themselves,
so this module works whether or not `rag-lab`/`deepeval` are installed (a
missing package just reports `None`, not an ImportError).
"""

from __future__ import annotations

import json
import platform
from importlib import metadata

from dealpoint.config import FRAMEWORK_VERSIONS_PATH
from dealpoint.eval.cases import git_sha7

PACKAGES: tuple[str, ...] = (
    "llama-index-core",
    "llama-index-retrievers-bm25",
    "deepeval",
    "braintrust",
    "qdrant-client",
    "fastembed",
    "bm25s",
    "openai",
    "pydantic",
)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def framework_versions() -> dict:
    versions = {pkg: _version(pkg) for pkg in PACKAGES}
    versions["python"] = platform.python_version()
    versions["git_sha7"] = git_sha7()
    return versions


def write_framework_versions(path=FRAMEWORK_VERSIONS_PATH) -> dict:
    payload = framework_versions()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    payload = write_framework_versions()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
