"""Quick TTS test — synthesizes a phrase and plays it using local eyetrax paths."""
from __future__ import annotations
import os, sys, types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

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

sys.path.insert(0, str(REPO_ROOT / "OpenVoice"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from eyetrax.utils.openvoice_engine import OpenVoiceEngine

OV_REPO     = REPO_ROOT / "OpenVoice"
WORK_DIR    = REPO_ROOT / "openvoice_runtime"
SPEAKER_WAV = REPO_ROOT / "speaker.wav"
OUTPUT_WAV  = REPO_ROOT / "tts_out" / "test_hello.wav"

print("Loading models (first run takes ~30 s) ...")
engine = OpenVoiceEngine(openvoice_repo=OV_REPO, work_dir=WORK_DIR, use_gpu=False)

print("Synthesizing ...")
engine.synthesize(
    text="This is an audio test to see how long processing takes to speak the audio given.",
    language="en",
    speaker_wav=SPEAKER_WAV,
    output_path=OUTPUT_WAV,
    play_audio=True,
)
print(f"Done — wrote {OUTPUT_WAV}")
