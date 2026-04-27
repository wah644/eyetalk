"""
Pre-generate speech cache for fixed phrases and emergency message.

Shares the same speech_cache/ directory as build_word_cache.py so
phrases spoken via the Fixed Phrases panel play back instantly.

Usage:
    python src/eyetrax/app/build_phrase_cache.py --speaker-wav speaker.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path


def _setup_mecab():
    if "unidic" in sys.modules:
        return
    try:
        import unidic_lite
        dicdir = Path(unidic_lite.__file__).resolve().parent / "dicdir"
        os.environ.setdefault("MECABRC", str(dicdir / "mecabrc"))
        os.environ.setdefault("UNIDIC_LITE_DIR", str(dicdir))
        sys.modules["unidic"] = types.SimpleNamespace(DICDIR=str(dicdir), VERSION="lite")
    except ImportError:
        pass

_setup_mecab()

import nltk
for _pkg, _kind in [("averaged_perceptron_tagger_eng", "taggers"), ("cmudict", "corpora")]:
    try:
        nltk.data.find(f"{_kind}/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "OpenVoice"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from eyetrax.utils.openvoice_engine import OpenVoiceEngine
from eyetrax.utils.speech_cache import (
    build_cache_path,
    cache_root_from,
    load_manifest,
    normalize_cache_key,
    save_manifest,
)

# Fixed phrases available in the UI panel
FIXED_PHRASES = [
    "Yes",
    "No",
    "Thank you",
    "Please help me",
    "I need water",
    "I need food",
    "I am in pain",
    "Call the nurse",
    "I feel sick",
    "I am cold",
    "I am hot",
    "Turn on the light",
    "Turn off the light",
]

EMERGENCY_MESSAGE = (
    "Hello, This is an emergency, I am a person of determination with locked in syndrome and this is a preset message, "
    "I repeat, This is an emergency, I am a person of determination with locked in syndrome and this is a preset message. "
    "Please come to my location in Khalifa University urgently"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-generate speech cache for fixed phrases and emergency message"
    )
    parser.add_argument("--speaker-wav", default=str(REPO_ROOT / "speaker.wav"),
                        help="Reference voice WAV for voice cloning")
    parser.add_argument("--language", default="en")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "speech_cache"))
    parser.add_argument("--work-dir",  default=str(REPO_ROOT / "openvoice_runtime"))
    parser.add_argument("--force", action="store_true", help="Regenerate even if cached")
    args = parser.parse_args()

    speaker_wav = Path(args.speaker_wav)
    cache_dir   = Path(args.cache_dir)
    work_dir    = Path(args.work_dir)

    if not speaker_wav.exists():
        raise SystemExit(f"speaker.wav not found: {speaker_wav}")

    all_texts = list(FIXED_PHRASES) + [EMERGENCY_MESSAGE]
    print(f"[cache] {len(all_texts)} phrases to process ({len(FIXED_PHRASES)} fixed + 1 emergency)")

    print("[cache] Loading models (first time ~30 s) ...")
    engine = OpenVoiceEngine(
        openvoice_repo=REPO_ROOT / "OpenVoice",
        work_dir=work_dir,
        use_gpu=False,
    )
    engine.warm(language=args.language, speaker_wav=speaker_wav)
    print("[cache] Models ready. Starting synthesis...")

    manifest = load_manifest(cache_dir)
    generated = 0
    skipped   = 0

    for i, text in enumerate(all_texts, 1):
        key      = normalize_cache_key(text)
        out_path = build_cache_path(key, cache_dir)
        rel      = str(out_path.relative_to(cache_root_from(cache_dir)))

        if out_path.exists() and not args.force:
            manifest[key] = rel
            skipped += 1
            print(f"[{i}/{len(all_texts)}] skip (cached)  {text[:60]}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            engine.synthesize(
                text=text,
                language=args.language,
                speaker_wav=speaker_wav,
                output_path=out_path,
                play_audio=False,
            )
            manifest[key] = rel
            save_manifest(manifest, cache_dir)
            generated += 1
            print(f"[{i}/{len(all_texts)}] generated     {text[:60]}")
        except Exception as exc:
            print(f"[{i}/{len(all_texts)}] ERROR {text[:40]}: {exc}")

    save_manifest(manifest, cache_dir)
    print(f"\n[cache] Done — {generated} generated, {skipped} already cached")
    print(f"[cache] Total manifest: {len(manifest)} entries in {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
