from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CACHE_ROOT = "speech_cache"
DEFAULT_DICT_SUBDIR = "dictionary_words"
MANIFEST_NAME = "manifest.json"


def normalize_cache_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def safe_filename(text: str, max_len: int = 80) -> str:
    normalized = normalize_cache_key(text)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in normalized)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = "utterance"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:max_len]}_{digest}"


def cache_root_from(base_dir: str | Path) -> Path:
    return Path(base_dir).resolve()


def dictionary_cache_dir(base_dir: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    return cache_root_from(base_dir) / DEFAULT_DICT_SUBDIR


def manifest_path(base_dir: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    return dictionary_cache_dir(base_dir) / MANIFEST_NAME


def build_cache_path(text: str, base_dir: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    return dictionary_cache_dir(base_dir) / f"{safe_filename(text)}.wav"


def load_manifest(base_dir: str | Path = DEFAULT_CACHE_ROOT) -> dict[str, str]:
    path = manifest_path(base_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(mapping: dict[str, str], base_dir: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    path = manifest_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    return path


def resolve_cached_audio(text: str, base_dir: str | Path = DEFAULT_CACHE_ROOT) -> Path | None:
    key = normalize_cache_key(text)
    if not key:
        return None
    manifest = load_manifest(base_dir)
    rel = manifest.get(key)
    if rel:
        candidate = cache_root_from(base_dir) / rel
        if candidate.exists():
            return candidate
    fallback = build_cache_path(key, base_dir)
    if fallback.exists():
        return fallback
    return None


def play_wav(path: str | Path) -> None:
    """Play a WAV file asynchronously (returns immediately)."""
    wav_path = Path(path).resolve()
    if sys.platform.startswith("win"):
        import winsound

        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        return
    if sys.platform == "darwin":
        subprocess.Popen(["afplay", str(wav_path)])
        return
    raise RuntimeError(f"Automatic playback is only wired for Windows/macOS right now: {wav_path}")


def play_wav_sync(path: str | Path) -> None:
    """Play a WAV file and block until playback finishes."""
    wav_path = Path(path).resolve()
    if sys.platform.startswith("win"):
        import winsound

        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
        return
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(wav_path)], check=False)
        return
    raise RuntimeError(f"Automatic playback is only wired for Windows/macOS right now: {wav_path}")
