from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from eyetrax.utils.speech_cache import (
    DEFAULT_CACHE_ROOT,
    build_cache_path,
    dictionary_cache_dir,
    normalize_cache_key,
    save_manifest,
)


def load_words(words_file: Path) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for raw in words_file.read_text(encoding="utf-8").splitlines():
        word = normalize_cache_key(raw)
        if not word or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def post_json(base_url: str, path: str, payload: dict) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-generate cached dictionary-word speech clips.")
    parser.add_argument(
        "--words-file",
        default=str(Path(__file__).with_name("words.txt")),
        help="Path to the source dictionary file.",
    )
    parser.add_argument(
        "--speaker-wav",
        required=True,
        help="Reference voice WAV path for OpenVoice cloning.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for synthesis.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_ROOT,
        help="Base cache directory that will contain dictionary_words/ and manifest.json.",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("EYETRAX_OPENVOICE_SERVER", "http://127.0.0.1:8765"),
        help="Warm OpenVoice server base URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files even when a cached clip already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    words_file = Path(args.words_file).resolve()
    speaker_wav = Path(args.speaker_wav).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = dictionary_cache_dir(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not words_file.exists():
        raise SystemExit(f"Dictionary file not found: {words_file}")
    if not speaker_wav.exists():
        raise SystemExit(f"Speaker WAV not found: {speaker_wav}")

    words = load_words(words_file)
    manifest: dict[str, str] = {}

    for index, word in enumerate(words, start=1):
        out_path = build_cache_path(word, cache_dir)
        rel_path = str(out_path.relative_to(cache_dir))
        manifest[word] = rel_path

        if out_path.exists() and not args.force:
            print(f"[{index}/{len(words)}] cached {word} -> {out_path}")
            continue

        payload = {
            "text": word,
            "language": args.language,
            "speaker_wav": str(speaker_wav),
            "output_path": str(out_path),
            "play_audio": False,
        }
        try:
            response = post_json(args.server, "/synthesize", payload)
        except urllib.error.URLError as exc:
            raise SystemExit(
                "OpenVoice server is not reachable for cache generation. "
                "Start the warm server first. "
                f"Details: {exc}"
            )
        if not response.get("ok"):
            raise SystemExit(response.get("error", f"Failed generating clip for {word}"))
        print(f"[{index}/{len(words)}] built {word} -> {out_path}")

    manifest_path = save_manifest(manifest, cache_dir)
    print(f"Saved manifest to {manifest_path}")
    print(f"Generated cache in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
