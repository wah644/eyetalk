"""ESP32 LED control for EyeTrax HOME IoT lights menu."""
from __future__ import annotations

import threading

import requests

ESP32_IP = "10.218.46.74"
BASE_URL = f"http://{ESP32_IP}"


def send_command(state: int) -> None:
    """Send ON (1) or OFF (0) command to the ESP32 LED controller."""
    endpoint = "/on" if state == 1 else "/off"
    try:
        response = requests.get(BASE_URL + endpoint, timeout=3)
        print(f"[lights] ESP32 says: {response.text}")
    except requests.exceptions.ConnectionError:
        print("[lights] Error: Could not reach ESP32. Check IP and WiFi connection.")
    except requests.exceptions.Timeout:
        print("[lights] Error: Request timed out.")


def lights_on_async() -> None:
    """Turn lights ON in a daemon thread (non-blocking for UI loop)."""
    threading.Thread(target=send_command, args=(1,), daemon=True).start()


def lights_off_async() -> None:
    """Turn lights OFF in a daemon thread (non-blocking for UI loop)."""
    threading.Thread(target=send_command, args=(0,), daemon=True).start()
