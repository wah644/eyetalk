"""
EyeTrax TTS — in-process OpenVoice speaker for macOS.

Design:
- Models are loaded once in a background thread at startup (warm).
- speak_word: cache hit → afplay immediately; miss → generate async, save to cache, play.
- speak_sentence: synthesize full text in background, play when ready.
- All synthesis runs on a single background worker thread (queue-based).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types
from pathlib import Path
from queue import Queue, Empty

from eyetrax.utils.speech_cache import (
    build_cache_path,
    cache_root_from,
    load_manifest,
    normalize_cache_key,
    play_wav,
    resolve_cached_audio,
    save_manifest,
)


def _setup_mecab() -> None:
    """Inject unidic-lite shim so MeloTTS can import without a system MeCab dict."""
    if "unidic" in sys.modules:
        return
    try:
        import unidic_lite
        dicdir = Path(unidic_lite.__file__).resolve().parent / "dicdir"
        os.environ.setdefault("MECABRC", str(dicdir / "mecabrc"))
        os.environ.setdefault("UNIDIC_LITE_DIR", str(dicdir))
        sys.modules["unidic"] = types.SimpleNamespace(DICDIR=str(dicdir), VERSION="lite")
    except ImportError:
        pass  # unidic-lite not installed; MeCab may still work via system dict


def _ensure_nltk() -> None:
    """Download NLTK data required by g2p_en if not already present."""
    try:
        import nltk
        for pkg, kind in [("averaged_perceptron_tagger_eng", "taggers"),
                           ("cmudict", "corpora")]:
            try:
                nltk.data.find(f"{kind}/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
    except ImportError:
        pass


class OpenVoiceSpeaker:
    """Async OpenVoice TTS speaker — designed for in-process use on macOS."""

    def __init__(
        self,
        *,
        openvoice_repo: str | Path,
        work_dir: str | Path,
        speaker_wav: str | Path,
        language: str = "en",
        cache_dir: str | Path = "speech_cache",
        use_gpu: bool = False,
    ) -> None:
        self._openvoice_repo = Path(openvoice_repo).resolve()
        self._work_dir       = Path(work_dir).resolve()
        self._speaker_wav    = Path(speaker_wav).resolve()
        self._language       = language
        self._cache_dir      = Path(cache_dir).resolve()
        self._use_gpu        = use_gpu

        self._engine = None
        self._engine_ready = False
        self._engine_lock  = threading.Lock()

        self._q: Queue = Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="tts-worker")
        self._worker.start()

        # Warm models in background so first synthesis doesn't block the UI
        threading.Thread(target=self._warm, daemon=True, name="tts-warm").start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak_word(self, word: str) -> None:
        """Play a single word. Cache hit plays immediately; miss generates async."""
        key = normalize_cache_key(word)
        if not key:
            return
        cached = resolve_cached_audio(key, self._cache_dir)
        if cached is not None:
            play_wav(cached)
            return
        # Not cached — queue synthesis; it will save to cache and play when done
        self._q.put(("word", key))

    def speak_sentence(self, text: str) -> None:
        """Play a sentence.

        If the full sentence is cached (e.g. fixed phrases / emergency message),
        play that single file immediately.  Otherwise queue word-by-word playback:
        cached words play instantly, uncached words are synthesized, cached, then played.
        """
        text = text.strip()
        if not text:
            return
        # Check for full-sentence cache hit first (fixed phrases / emergency)
        key = normalize_cache_key(text)
        if key:
            cached = resolve_cached_audio(key, self._cache_dir)
            if cached is not None:
                play_wav(cached)
                return
        # No full-sentence cache — queue word-by-word playback
        self._q.put(("sentence", text))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _warm(self) -> None:
        try:
            _setup_mecab()
            _ensure_nltk()
            self._get_engine()
            self._engine_ready = True
            print("[tts] Engine warmed and ready")
        except Exception as exc:
            print(f"[tts] Warm failed (TTS will be slower on first use): {exc}")

    def _get_engine(self):
        with self._engine_lock:
            if self._engine is None:
                from eyetrax.utils.openvoice_engine import OpenVoiceEngine
                _setup_mecab()
                _ensure_nltk()
                self._work_dir.mkdir(parents=True, exist_ok=True)
                self._engine = OpenVoiceEngine(
                    openvoice_repo=self._openvoice_repo,
                    work_dir=self._work_dir,
                    use_gpu=self._use_gpu,
                )
                self._engine.warm(
                    language=self._language,
                    speaker_wav=self._speaker_wav,
                )
            return self._engine

    def _run(self) -> None:
        while True:
            try:
                kind, text = self._q.get(timeout=1.0)
            except Empty:
                continue
            try:
                if kind == "word":
                    self._synthesize_word(text)
                elif kind == "sentence":
                    self._synthesize_sentence(text)
            except Exception as exc:
                print(f"[tts] Synthesis error: {exc}")
            finally:
                self._q.task_done()

    def _synthesize_word(self, word: str) -> None:
        """Generate audio for a single word, save to cache, update manifest, play."""
        engine = self._get_engine()
        out_path = build_cache_path(word, self._cache_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        engine.synthesize(
            text=word,
            language=self._language,
            speaker_wav=self._speaker_wav,
            output_path=out_path,
            play_audio=True,
        )
        # Persist to manifest so next lookup is instant
        manifest = load_manifest(self._cache_dir)
        rel = str(out_path.relative_to(cache_root_from(self._cache_dir)))
        manifest[word] = rel
        save_manifest(manifest, self._cache_dir)
        print(f"[tts] Cached new word '{word}' → {out_path.name}")

    def _synthesize_sentence(self, text: str) -> None:
        """Synthesize full text as one audio file, save to cache, and play."""
        engine = self._get_engine()
        key = normalize_cache_key(text)
        out_path = build_cache_path(key, self._cache_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        engine.synthesize(
            text=text,
            language=self._language,
            speaker_wav=self._speaker_wav,
            output_path=out_path,
            play_audio=True,
        )
        manifest = load_manifest(self._cache_dir)
        rel = str(out_path.relative_to(cache_root_from(self._cache_dir)))
        manifest[key] = rel
        save_manifest(manifest, self._cache_dir)
        print(f"[tts] Cached sentence → {out_path.name}")

