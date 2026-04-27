"""
Pre-generate speech cache for all dictionary words.

Runs OpenVoiceEngine directly — no HTTP server required.

Usage:
    conda run -n openvoice-mac python src/eyetrax/app/build_word_cache.py \\
        --speaker-wav speaker.wav

Words already cached (in speech_cache/dictionary_words/) are skipped.
Use --force to regenerate everything.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

# Bootstrap MeCab before any melo import
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


def load_words(words_file: Path) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for raw in words_file.read_text(encoding="utf-8").splitlines():
        word = normalize_cache_key(raw)
        if word and word not in seen:
            seen.add(word)
            words.append(word)
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-generate speech cache for dictionary words")
    parser.add_argument("--speaker-wav", default=str(REPO_ROOT / "speaker.wav"),
                        help="Reference voice WAV for voice cloning")
    parser.add_argument("--words-file", default=str(REPO_ROOT / "src" / "eyetrax" / "app" / "words.txt"),
                        help="Path to words.txt")
    parser.add_argument("--language", default="en")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "speech_cache"))
    parser.add_argument("--work-dir",  default=str(REPO_ROOT / "openvoice_runtime"))
    parser.add_argument("--force", action="store_true", help="Regenerate even if cached")
    args = parser.parse_args()

    words_file  = Path(args.words_file)
    speaker_wav = Path(args.speaker_wav)
    cache_dir   = Path(args.cache_dir)
    work_dir    = Path(args.work_dir)

    if not words_file.exists():
        raise SystemExit(f"words.txt not found: {words_file}")
    if not speaker_wav.exists():
        raise SystemExit(f"speaker.wav not found: {speaker_wav}")

    words = load_words(words_file)
    print(f"[cache] {len(words)} words to process from {words_file}")

    print("[cache] Loading models (first time ~30 s) ...")
    engine = OpenVoiceEngine(
        openvoice_repo=REPO_ROOT / "OpenVoice",
        work_dir=work_dir,
        use_gpu=False,
    )
    engine.warm(language=args.language, speaker_wav=speaker_wav)
    print("[cache] Models ready. Starting synthesis...")

    manifest = load_manifest(cache_dir)

    for i, word in enumerate(words, 1):
        out_path = build_cache_path(word, cache_dir)
        rel      = str(out_path.relative_to(cache_root_from(cache_dir)))

        if out_path.exists() and not args.force:
            manifest[word] = rel
            print(f"[{i}/{len(words)}] skip (cached)  {word}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            engine.synthesize(
                text=word,
                language=args.language,
                speaker_wav=speaker_wav,
                output_path=out_path,
                play_audio=False,
            )
            manifest[word] = rel
            save_manifest(manifest, cache_dir)
            print(f"[{i}/{len(words)}] generated     {word}")
        except Exception as exc:
            print(f"[{i}/{len(words)}] ERROR {word}: {exc}")

    save_manifest(manifest, cache_dir)
    print(f"\n[cache] Done — {len(manifest)} words cached in {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
