"""
Keyword-to-Sentence Generator using Local LLM (qwen2.5:0.5b via Ollama).

Converts telegraphic keyword input into fluent first-person sentences
for patients with Locked-In Syndrome.

Core API:
    generate_sentence_from_keywords(keyword_line: str) -> str
    generate_sentence_async(keyword_line: str) -> None   (non-blocking)
    get_pending_result() -> Optional[str]
    is_generating() -> bool
"""

import requests
import json
import os
import threading
from typing import Dict, Optional


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:0.5b"

_CACHE_FILE_PATH = os.path.join(os.path.dirname(__file__), "sentence_cache.json")
_sentence_cache: Dict[str, str] = {}


def _load_cache_from_disk() -> None:
    global _sentence_cache
    if not os.path.exists(_CACHE_FILE_PATH):
        return
    try:
        with open(_CACHE_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _sentence_cache = {str(k): str(v) for k, v in data.items() if isinstance(k, str)}
    except Exception as e:
        print(f"Warning: could not load cache from disk: {e}")


def _save_cache_to_disk() -> None:
    try:
        with open(_CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(_sentence_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: could not save cache to disk: {e}")


_load_cache_from_disk()


def check_ollama_available() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def normalize_input(keyword_line: str) -> str:
    return " ".join(keyword_line.lower().strip().split())


SYSTEM_PROMPT = (
    "You help a patient with Locked-In Syndrome communicate. "
    "They type keywords and you turn them into one short first-person sentence "
    "expressing what the patient wants or feels. Include EVERY keyword. "
    "Only reply with the sentence."
)

_FEW_SHOT = [
    ("cold blanket", "I am cold and I need a blanket."),
    ("pain chest left", "I have pain in my left chest."),
    ("need doctor", "I need a doctor."),
    ("thirsty water", "I am thirsty and I need water."),
    ("head hurts medicine", "My head hurts and I need medicine."),
    ("call nurse", "Please call the nurse."),
]


def _build_chat_messages(keywords: str) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for kw, sent in _FEW_SHOT:
        messages.append({"role": "user", "content": kw})
        messages.append({"role": "assistant", "content": sent})
    messages.append({"role": "user", "content": keywords})
    return messages


def call_ollama_llm(keywords: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    try:
        payload = {
            "model": model,
            "messages": _build_chat_messages(keywords),
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 50,
                "num_ctx": 1024,
            },
        }
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
            text = result.get("message", {}).get("content", "").strip()
            if text:
                text = text.strip("\"'").replace("**", "").replace("*", "")
                for end in (".", "!", "?"):
                    if end in text:
                        text = text.split(end)[0] + end
                        break
                else:
                    text += "."
                text = text.replace("..", ".").strip()
            return text
        else:
            print(f"Error: Ollama returned status {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("Error: Ollama timed out")
        return None
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def _keywords_preserved(keywords: str, sentence: str) -> bool:
    significant = {w for w in keywords.lower().split() if len(w) >= 3}
    if not significant:
        return True
    sent_lower = sentence.lower()
    found = sum(1 for w in significant if w in sent_lower)
    return found >= len(significant) * 0.6


def generate_sentence_from_keywords(keyword_line: str) -> str:
    normalized = normalize_input(keyword_line)
    if normalized in _sentence_cache:
        return _sentence_cache[normalized]
    if not normalized:
        return "I am here."

    if check_ollama_available():
        result = call_ollama_llm(normalized)
        if result and _keywords_preserved(normalized, result):
            _sentence_cache[normalized] = result
            _save_cache_to_disk()
            return result
        elif result:
            print(f"Warning: LLM dropped keywords ({normalized} -> {result}), using fallback")

    result = fallback_generation(normalized)
    _sentence_cache[normalized] = result
    _save_cache_to_disk()
    return result


def fallback_generation(keywords: str) -> str:
    words = keywords.split()
    if "need" in words:
        idx = words.index("need")
        before = " ".join(words[:idx]).strip()
        after = " ".join(words[idx + 1:]).strip() or "help"
        if len(after.split()) == 1 and after not in ("help", "water", "food", "medicine", "sleep"):
            after = "a " + after
        return f"I am {before} and I need {after}." if before else f"I need {after}."
    elif "want" in words:
        idx = words.index("want")
        before = " ".join(words[:idx]).strip()
        after = " ".join(words[idx + 1:]).strip() or "something"
        return f"I am {before} and I want {after}." if before else f"I want {after}."
    elif "pain" in words:
        idx = words.index("pain")
        rest = " ".join(words[:idx] + words[idx + 1:]).strip() or "somewhere"
        return f"I have pain in my {rest}."
    elif "hurts" in words or "hurt" in words:
        hw = "hurts" if "hurts" in words else "hurt"
        idx = words.index(hw)
        part = " ".join(words[:idx]).strip() or "body"
        after = " ".join(words[idx + 1:]).strip()
        return f"My {part} hurts and I need {after}." if after else f"My {part} hurts."
    elif keywords.strip() in ("hi", "hello", "hey", "yes", "no", "ok", "thank you", "thanks"):
        return f"{keywords.strip().capitalize()}."
    elif "no" in words:
        rest = " ".join(w for w in words if w != "no").strip()
        return f"No, please {rest}." if rest else "No."
    elif "call" in words:
        rest = " ".join(words[words.index("call") + 1:]).strip() or "someone"
        return f"Please call {rest}."
    elif "bathroom" in words or "toilet" in words:
        return "I need to use the bathroom."
    elif len(words) == 1:
        return f"I need {keywords}."
    elif len(words) == 2:
        return f"I am {words[0]} and I need {words[1]}."
    else:
        return f"I {keywords}."


# ── Threaded generation (non-blocking for UI loop) ──────────────

_pending_result: Optional[str] = None
_generation_lock = threading.Lock()
_generation_busy = False


def is_generating() -> bool:
    return _generation_busy


def get_pending_result() -> Optional[str]:
    global _pending_result
    with _generation_lock:
        result = _pending_result
        _pending_result = None
        return result


def generate_sentence_async(keyword_line: str) -> None:
    global _generation_busy
    if _generation_busy:
        return

    def _worker():
        global _pending_result, _generation_busy
        _generation_busy = True
        try:
            sentence = generate_sentence_from_keywords(keyword_line)
            with _generation_lock:
                _pending_result = sentence
        finally:
            _generation_busy = False

    threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    print("Keyword-to-Sentence Generator")
    print(f"Model: {DEFAULT_MODEL}")
    print("Type keywords and press Enter. Type 'quit' to exit, 'clear' to clear cache.")
    print("-" * 60)
    while True:
        user_input = input("\nKeywords: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "clear":
            _sentence_cache.clear()
            _save_cache_to_disk()
            print("Cache cleared.")
            continue
        if user_input:
            print(f"Output: {generate_sentence_from_keywords(user_input)}")
