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


def _pi_version() -> str | None:
    """`pi --version`'s stdout, trimmed. `None` (not raised) when `pi` isn't
    on PATH or the invocation fails -- this module must stay usable in an
    environment without pi installed.
    """
    import subprocess

    try:
        result = subprocess.run(["pi", "--version"], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    version = result.stdout.strip()
    return version or None


def _pi_mcp_extension_version() -> str | dict:
    """pi-mcp-extension's version (spec 'Operator setup' item 4). `pi
    extension list` writes only diagnostic `$ref` warnings to stderr and
    nothing parseable to stdout in this environment (judged by exit status,
    never by scanning its output) -- the documented fallback is the
    installed extension's own `package.json` under `~/.pi`. Returns a
    `{"unavailable": reason}` dict, never a guess, when neither path works.
    """
    import subprocess
    from pathlib import Path

    try:
        subprocess.run(["pi", "extension", "list"], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        pass  # fall through to the package.json lookup regardless

    for package_json in Path.home().glob(".pi/**/node_modules/pi-mcp-extension/package.json"):
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        version = payload.get("version")
        if version:
            return version
    return {"unavailable": "pi-mcp-extension not found under ~/.pi and 'pi extension list' prints no parseable version"}


def framework_versions() -> dict:
    versions = {pkg: _version(pkg) for pkg in PACKAGES}
    versions["python"] = platform.python_version()
    versions["git_sha7"] = git_sha7()
    versions["pi"] = _pi_version()
    versions["pi-mcp-extension"] = _pi_mcp_extension_version()
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
