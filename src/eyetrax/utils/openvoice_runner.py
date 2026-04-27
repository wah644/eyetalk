from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eyetrax.utils.openvoice_engine import OpenVoiceEngine, load_request


def _main() -> int:
    if len(sys.argv) != 2:
        print("usage: python openvoice_runner.py <request.json>", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1]).resolve()
    req = load_request(request_path)
    engine = OpenVoiceEngine(
        openvoice_repo=req["openvoice_repo"],
        work_dir=req["work_dir"],
        use_gpu=bool(req.get("use_gpu", False)),
    )
    output_path = engine.synthesize(
        text=str(req["text"]),
        language=str(req["language"]).lower(),
        speaker_wav=req["speaker_wav"],
        output_path=req["output_path"],
        refresh_speaker=bool(req.get("refresh_speaker", False)),
        play_audio=bool(req.get("play_audio", False)),
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
