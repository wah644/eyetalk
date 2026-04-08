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
    phone_number: str = "+971559877486"
    message: str = "hi this is a test"
    speak_delay_s: float = 3.0


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


def dial_and_call(cfg: EmergencyCallConfig) -> None:
    number = cfg.phone_number.strip()
    serial = cfg.device_serial or _pick_first_device_serial(cfg)
    if not serial:
        raise RuntimeError("No ADB devices found. Run `adb pair`/`adb connect` first.")
    cfg = EmergencyCallConfig(
        adb_path=cfg.adb_path,
        device_serial=serial,
        phone_number=cfg.phone_number,
        message=cfg.message,
        speak_delay_s=cfg.speak_delay_s,
    )

    # Open dialer with number filled.
    _run(adb_cmd(cfg, "shell", "am", "start", "-a", "android.intent.action.DIAL", "-d", f"tel:{number}"))
    time.sleep(0.6)

    # Press call key
    _run(adb_cmd(cfg, "shell", "input", "keyevent", "KEYCODE_CALL"))


def speak_message_windows_sapi(message: str) -> None:
    # Built-in on Windows; avoids extra dependencies.
    # Build script safely (escape single quotes).
    msg = message.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$speak.Rate = 0; "
        f"$speak.Speak('{msg}');"
    )
    _run(["powershell", "-NoProfile", "-Command", ps], timeout_s=max(30, len(message) / 8 + 10))


def trigger_emergency_call(cfg: EmergencyCallConfig) -> None:
    dial_and_call(cfg)
    if cfg.message.strip():
        time.sleep(max(0.0, cfg.speak_delay_s))
        if os.name == "nt":
            speak_message_windows_sapi(cfg.message)

