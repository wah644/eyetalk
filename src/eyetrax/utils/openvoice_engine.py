from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from threading import Lock


LANGUAGE_TO_MELO = {
    "en": "EN_NEWEST",
    "en-us": "EN_NEWEST",
    "en-gb": "EN",
    "es": "ES",
    "fr": "FR",
    "zh": "ZH",
    "zh-cn": "ZH",
    "ja": "JP",
    "jp": "JP",
    "ko": "KR",
    "kr": "KR",
}


def _speaker_cache_path(work_dir: Path, speaker_wav: Path) -> Path:
    key = hashlib.sha256(str(speaker_wav.resolve()).encode("utf-8")).hexdigest()[:16]
    cache_dir = work_dir / "processed" / "speaker_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = speaker_wav.stem or "speaker"
    return cache_dir / f"{stem}_{key}.pth"


def play_wav(path: Path) -> None:
    if sys.platform.startswith("win"):
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    if sys.platform == "darwin":
        import subprocess

        subprocess.run(["afplay", str(path)], check=False)
        return
    raise RuntimeError(f"Automatic playback is only wired for Windows right now: {path}")


class OpenVoiceEngine:
    def __init__(self, *, openvoice_repo: str | Path, work_dir: str | Path, use_gpu: bool = False):
        self.openvoice_repo = Path(openvoice_repo).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.use_gpu = bool(use_gpu)
        self.device = None
        self._torch = None
        self._MeloTTS = None
        self._ToneColorConverter = None
        self._converter = None
        self._models: dict[str, object] = {}
        self._source_ses: dict[str, object] = {}
        self._target_ses: dict[Path, object] = {}
        self._lock = Lock()

    def _prepare_imports(self) -> None:
        # Do NOT os.chdir — that clobbers the eyetrax app's working directory.
        # Just add OpenVoice to sys.path so its internal imports resolve.
        if str(self.openvoice_repo) not in sys.path:
            sys.path.insert(0, str(self.openvoice_repo))

        # MeloTTS imports Japanese tokenization at module import time, even for English.
        try:
            import unidic_lite

            dicdir = Path(unidic_lite.__file__).resolve().parent / "dicdir"
            os.environ.setdefault("MECABRC", str(dicdir / "mecabrc"))
            os.environ.setdefault("UNIDIC_LITE_DIR", str(dicdir))
            shim = types.SimpleNamespace(DICDIR=str(dicdir), VERSION="lite")
            sys.modules["unidic"] = shim
        except Exception:
            pass

    def _ensure_runtime(self) -> None:
        if self._torch is not None:
            return

        self._prepare_imports()

        import torch
        from melo.api import TTS as MeloTTS
        from openvoice.api import ToneColorConverter

        self._torch = torch
        self._MeloTTS = MeloTTS
        self._ToneColorConverter = ToneColorConverter
        self.device = "cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu"

    @property
    def checkpoints_v2(self) -> Path:
        return self.openvoice_repo / "checkpoints_v2"

    @property
    def converter_dir(self) -> Path:
        return self.checkpoints_v2 / "converter"

    @property
    def source_ses_dir(self) -> Path:
        return self.checkpoints_v2 / "base_speakers" / "ses"

    def get_converter(self):
        self._ensure_runtime()
        if self._converter is None:
            if not self.converter_dir.exists():
                raise FileNotFoundError(
                    "OpenVoice V2 checkpoints missing. Expected converter dir at "
                    f"{self.converter_dir}"
                )
            converter = self._ToneColorConverter(f"{self.converter_dir}/config.json", device=self.device)
            converter.load_ckpt(f"{self.converter_dir}/checkpoint.pth")
            self._converter = converter
        return self._converter

    def get_model(self, language: str):
        self._ensure_runtime()
        melo_language = LANGUAGE_TO_MELO.get(language.lower())
        if melo_language is None:
            raise ValueError(
                f"Unsupported TTS language `{language}` for OpenVoice backend. "
                f"Supported: {sorted(LANGUAGE_TO_MELO)}"
            )
        model = self._models.get(melo_language)
        if model is None:
            model = self._MeloTTS(language=melo_language, device=self.device)
            self._models[melo_language] = model
        return melo_language, model

    def get_source_se(self, model) -> tuple[int, object]:
        speaker_ids = model.hps.data.spk2id
        if not speaker_ids:
            raise RuntimeError("No speaker IDs available for MeloTTS model")
        speaker_key = list(speaker_ids.keys())[0]
        speaker_id = speaker_ids[speaker_key]
        source_key = speaker_key.lower().replace("_", "-")
        source_se = self._source_ses.get(source_key)
        if source_se is None:
            source_se_path = self.source_ses_dir / f"{source_key}.pth"
            if not source_se_path.exists():
                raise FileNotFoundError(f"OpenVoice source speaker embedding not found: {source_se_path}")
            source_se = self._torch.load(source_se_path, map_location=self.device)
            self._source_ses[source_key] = source_se
        return speaker_id, source_se

    def get_target_se(self, speaker_wav: Path, *, refresh_speaker: bool = False):
        speaker_wav = speaker_wav.resolve()
        speaker_se_path = _speaker_cache_path(self.work_dir, speaker_wav)
        if speaker_se_path in self._target_ses and not refresh_speaker:
            return self._target_ses[speaker_se_path]

        if speaker_se_path.exists() and not refresh_speaker:
            target_se = self._torch.load(speaker_se_path, map_location=self.device)
        else:
            target_se = self.get_converter().extract_se(
                [str(speaker_wav)],
                se_save_path=str(speaker_se_path),
            )
        self._target_ses[speaker_se_path] = target_se
        return target_se

    def warm(self, *, language: str = "en", speaker_wav: str | Path | None = None) -> None:
        with self._lock:
            _melo_language, model = self.get_model(language)
            self.get_converter()
            self.get_source_se(model)
            if speaker_wav is not None:
                self.get_target_se(Path(speaker_wav))

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        speaker_wav: str | Path,
        output_path: str | Path,
        refresh_speaker: bool = False,
        play_audio: bool = False,
    ) -> Path:
        output_path = Path(output_path).resolve()
        speaker_wav = Path(speaker_wav).resolve()
        if not speaker_wav.exists():
            raise FileNotFoundError(f"speaker_wav not found: {speaker_wav}")

        with self._lock:
            _melo_language, model = self.get_model(language)
            converter = self.get_converter()
            target_se = self.get_target_se(speaker_wav, refresh_speaker=refresh_speaker)
            speaker_id, source_se = self.get_source_se(model)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_dir = self.work_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            src_path = tmp_dir / "openvoice_base.wav"
            model.tts_to_file(text, speaker_id, str(src_path), speed=1.0, quiet=True)

            converter.convert(
                audio_src_path=str(src_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(output_path),
                message="@MyShell",
            )

        if play_audio:
            play_wav(output_path)
        return output_path


def load_request(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
