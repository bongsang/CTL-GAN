"""Versioned model registry and verified, atomic downloads (no pickle loading)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class MakerError(Exception):
    """An actionable error to show without a Python traceback."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registry() -> dict:
    return json.loads((ROOT / "models" / "registry.json").read_text(encoding="utf-8"))


def cache_root() -> Path:
    override = os.environ.get("CTL_GAN_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ctl-gan"


def verify(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise MakerError(f"Model verification failed: {path}. Move this file aside and retry; it was not loaded.")


def ensure_model(spec: dict, cache: Path, offline: bool = False) -> Path:
    target = cache / spec["filename"]
    if target.is_file():
        verify(target, spec)
        return target
    url = spec.get("url")
    if offline or not url:
        reason = "Offline mode is enabled" if offline else "Public weights are not yet released"
        raise MakerError(f"{reason}. Place the verified {spec['filename']} in {cache}. See docs/models.md.")
    if not url.startswith("https://"):
        raise MakerError("The model registry must use an HTTPS download URL.")
    cache.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CTL-GAN-maker/0.1"})
        with urllib.request.urlopen(request, timeout=60) as source:
            if not source.url.startswith("https://"):
                raise MakerError("Refusing an insecure model download redirect.")
            with tempfile.NamedTemporaryFile(dir=cache, prefix=".download-", delete=False) as output:
                temporary = Path(output.name)
                count = 0
                while block := source.read(1024 * 1024):
                    count += len(block)
                    if count > spec["bytes"]:
                        raise MakerError("Downloaded model exceeds its registered size.")
                    output.write(block)
        verify(temporary, spec)
        temporary.replace(target)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MakerError(f"Could not download the model: {exc}. Retry, or use the manual download in docs/models.md.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
