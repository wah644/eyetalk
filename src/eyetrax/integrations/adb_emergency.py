from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass


def _run(args: list[str], timeout_s: float = 20) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed ({cp.returncode}): {' '.join(args)}\n{cp.stdout}")
    return cp


def _default_adb_path() -> str:
    adb = shutil.which("adb")
    if adb:
        return adb
    # Common WinGet install location (works even if PATH isn't refreshed).
    winget_adb = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft",
        "WinGet",
        "Packages",
        "Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe",
        "platform-tools",
        "adb.exe",
    )
    if os.name == "nt" and os.path.isfile(winget_adb):
        return winget_adb
    return "adb"


@dataclass(frozen=True)
class EmergencyCallConfig:
    adb_path: str = _default_adb_path()
    device_serial: str | None = None
    phone_number: str = "+971554611264"
    message: str = "Hello, This is an emergency, I am a person of determination with locked in syndrome and this is a preset message, I repeat, This is an emergency, I am a person of determination with locked in syndrome and this is a preset message. Please come to my location in Khalifa University urgently"
    speak_delay_s: float = 0.0
    wait_for_answer_timeout_s: float = 45.0
    poll_interval_s: float = 0.6


def adb_cmd(cfg: EmergencyCallConfig, *parts: str) -> list[str]:
    cmd = [cfg.adb_path]
    if cfg.device_serial:
        cmd += ["-s", cfg.device_serial]
    cmd += list(parts)
    return cmd


def _pick_first_device_serial(cfg: EmergencyCallConfig) -> str | None:
    cp = _run([cfg.adb_path, "devices"], timeout_s=10)
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            return parts[0]
    return None


def ensure_device(cfg: EmergencyCallConfig) -> EmergencyCallConfig:
    serial = cfg.device_serial or _pick_first_device_serial(cfg)
    if not serial:
        raise RuntimeError("No ADB devices found. Run `adb pair`/`adb connect` first.")
    if cfg.device_serial == serial:
        return cfg
    return EmergencyCallConfig(
        adb_path=cfg.adb_path,
        device_serial=serial,
        phone_number=cfg.phone_number,
        message=cfg.message,
        speak_delay_s=cfg.speak_delay_s,
        wait_for_answer_timeout_s=cfg.wait_for_answer_timeout_s,
        poll_interval_s=cfg.poll_interval_s,
    )


def get_phone_screen_size(cfg: EmergencyCallConfig) -> tuple[int, int]:
    cfg = ensure_device(cfg)
    cp = _run(adb_cmd(cfg, "shell", "wm", "size"), timeout_s=10)
    # Example: "Physical size: 1080x2400"
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if "Physical size:" in line:
            size = line.split("Physical size:", 1)[1].strip()
            w_s, h_s = size.split("x", 1)
            return int(w_s), int(h_s)
    raise RuntimeError(f"Could not parse screen size from: {cp.stdout}")


def tap(cfg: EmergencyCallConfig, x: int, y: int) -> None:
    cfg = ensure_device(cfg)
    _run(adb_cmd(cfg, "shell", "input", "tap", str(int(x)), str(int(y))), timeout_s=10)


def open_url(cfg: EmergencyCallConfig, url: str) -> None:
    cfg = ensure_device(cfg)
    _run(adb_cmd(cfg, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url), timeout_s=10)


def dial_and_call(cfg: EmergencyCallConfig) -> None:
    number = cfg.phone_number.strip()
    cfg = ensure_device(cfg)

    # Open dialer with number filled.
    _run(adb_cmd(cfg, "shell", "am", "start", "-a", "android.intent.action.DIAL", "-d", f"tel:{number}"))
    time.sleep(0.6)

    # Press call key
    _run(adb_cmd(cfg, "shell", "input", "keyevent", "KEYCODE_CALL"))


def _is_call_active_from_telecom_dump(text: str) -> bool:
    # This format varies across Android versions/OEMs; handle common patterns.
    t = text.upper()
    # Some builds include explicit call states.
    if "CALLSTATE: ACTIVE" in t or "STATE=ACTIVE" in t:
        return True
    # As a fallback, detect in-call foreground state markers.
    if "FOREGROUND_CALL_STATE" in t and "ACTIVE" in t:
        return True
    return False


def _is_call_active_from_registry_dump(text: str) -> bool:
    # telephony.registry often contains: mCallState=<0|1|2>
    # 0=IDLE, 1=RINGING, 2=OFFHOOK (includes active in-call)
    for line in text.splitlines():
        line = line.strip()
        if "mCallState=" in line:
            try:
                val = int(line.split("mCallState=", 1)[1].split()[0])
            except Exception:
                continue
            return val == 2
    return False


def wait_for_call_answered(cfg: EmergencyCallConfig) -> bool:
    # Wait a minimum of 6 s before polling — Samsung's telecom dump reports
    # ACTIVE immediately after dialling (before the remote party picks up),
    # so we skip the ringing phase entirely with a hard floor.
    time.sleep(6.0)
    deadline = time.time() + max(1.0, cfg.wait_for_answer_timeout_s)
    while time.time() < deadline:
        try:
            cp = _run(adb_cmd(cfg, "shell", "dumpsys", "telecom"), timeout_s=10)
            if _is_call_active_from_telecom_dump(cp.stdout or ""):
                # Extra settle delay so the in-call UI is fully rendered
                time.sleep(1.5)
                return True
        except Exception:
            pass
        time.sleep(max(0.2, cfg.poll_interval_s))
    return False


def enable_speakerphone(cfg: EmergencyCallConfig) -> None:
    """Enable speakerphone during an active call by tapping the Speaker button in the in-call UI."""
    cfg = ensure_device(cfg)
    try:
        w, h = get_phone_screen_size(cfg)
        # Speaker button is bottom-left of the 2x3 in-call button grid.
        # Verified on Samsung Galaxy: ~25% from left, ~67.5% from top.
        sx, sy = int(w * 0.25), int(h * 0.675)
        _run(adb_cmd(cfg, "shell", "input", "tap", str(sx), str(sy)), timeout_s=10)
        print(f"[emergency] Speakerphone enabled (tapped in-call UI at {sx},{sy})")
    except Exception as e:
        print(f"[emergency] Failed to enable speakerphone: {e}")


def trigger_emergency_call(cfg: EmergencyCallConfig, speak_fn=None) -> None:
    dial_and_call(cfg)
    if cfg.message.strip():
        answered = wait_for_call_answered(cfg)
        if not answered:
            return
        enable_speakerphone(cfg)
        if cfg.speak_delay_s > 0:
            time.sleep(cfg.speak_delay_s)
        if speak_fn is not None:
            speak_fn(cfg.message)
