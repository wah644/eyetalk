"""
Pre-generate speech cache for COCA dictionary words used in SynthSwipe mode.

Reads COCA_WordFrequency.csv, takes the top N words by frequency, and
synthesizes audio for any not already cached. Shares the same speech_cache/
directory as build_word_cache.py so words common to both dictionaries are
only synthesized once.

Usage:
    python src/eyetrax/app/build_synth_speech_cache.py --speaker-wav speaker.wav
    python src/eyetrax/app/build_synth_speech_cache.py --speaker-wav speaker.wav --top 1000
    python src/eyetrax/app/build_synth_speech_cache.py --speaker-wav speaker.wav --top 0  # all words
"""
from __future__ import annotations

import argparse
import csv
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


def load_coca_words(csv_path: Path, top: int) -> list[str]:
    """Load words from COCA CSV sorted by frequency, filtered to a-z only."""
    rows: list[tuple[float, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            word = row["lemma"].strip().lower()
            if not word.isalpha():
                continue
            try:
                per_mil = float(row["perMil"])
            except (ValueError, KeyError):
                continue
            rows.append((per_mil, word))

    # Already sorted by rank in the CSV (descending frequency), but sort explicitly
    rows.sort(key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    words: list[str] = []
    for _, word in rows:
        key = normalize_cache_key(word)
        if key and key not in seen:
            seen.add(key)
            words.append(key)

    if top and top > 0:
        words = words[:top]
    return words


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-generate speech cache for COCA/SynthSwipe dictionary words"
    )
    parser.add_argument("--speaker-wav", default=str(REPO_ROOT / "speaker.wav"),
                        help="Reference voice WAV for voice cloning")
    parser.add_argument("--coca-csv",
                        default=str(REPO_ROOT / "src" / "eyetrax" / "app" / "COCA_WordFrequency.csv"),
                        help="Path to COCA_WordFrequency.csv")
    parser.add_argument("--top", type=int, default=500,
                        help="Cache only the top N most frequent words (0 = all 4362 words)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "speech_cache"))
    parser.add_argument("--work-dir",  default=str(REPO_ROOT / "openvoice_runtime"))
    parser.add_argument("--force", action="store_true", help="Regenerate even if cached")
    args = parser.parse_args()

    coca_csv    = Path(args.coca_csv)
    speaker_wav = Path(args.speaker_wav)
    cache_dir   = Path(args.cache_dir)
    work_dir    = Path(args.work_dir)

    if not coca_csv.exists():
        raise SystemExit(f"COCA CSV not found: {coca_csv}")
    if not speaker_wav.exists():
        raise SystemExit(f"speaker.wav not found: {speaker_wav}")

    words = load_coca_words(coca_csv, args.top)
    label = f"top {args.top}" if args.top > 0 else "all"
    print(f"[cache] {len(words)} COCA words to process ({label} by frequency)")

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

    for i, word in enumerate(words, 1):
        out_path = build_cache_path(word, cache_dir)
        rel      = str(out_path.relative_to(cache_root_from(cache_dir)))

        if out_path.exists() and not args.force:
            manifest[word] = rel
            skipped += 1
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
            generated += 1
            print(f"[{i}/{len(words)}] generated     {word}")
        except Exception as exc:
            print(f"[{i}/{len(words)}] ERROR {word}: {exc}")

    save_manifest(manifest, cache_dir)
    print(f"\n[cache] Done — {generated} generated, {skipped} already cached")
    print(f"[cache] Total manifest: {len(manifest)} words in {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
