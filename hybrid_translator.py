"""
Live meeting transcription + translation, with a CLOUD / LOCAL switch.

  CLOUD  - streams audio to Deepgram (fast, speaker labels, costs money)
  LOCAL  - runs Whisper large-v3-turbo on your own GPU (free, private,
           no speaker labels)

Flip between them with the toggle in the top bar; the rest of the app
behaves the same either way.

Setup (once):
    pip3 install --user websockets

    # for LOCAL mode, build the turbo model once (~1.6GB, no re-download
    # if you already have the HF copy):
    ct2-transformers-converter --model openai/whisper-large-v3-turbo \
        --output_dir ./models/whisper-large-v3-turbo-ct2 \
        --copy_files tokenizer.json preprocessor_config.json \
        --quantization float16

.env next to this script:
    DEEPGRAM_API_KEY=...
    GEMINI_API_KEY=...                  # used for translation if present
    GEMINI_MODEL=gemini-2.5-flash

  ...or any OpenAI-compatible provider instead:
    TRANSLATE_API_KEY=...
    TRANSLATE_BASE_URL=https://api.groq.com/openai/v1
    TRANSLATE_MODEL=llama-3.3-70b-versatile

Run:
    python3 live_translator.py
"""

import asyncio
import json
import os
import platform
import queue
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime

import websockets
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = RATE * 2 * CHUNK_MS // 1000

# languages nova-3 streams with a dedicated model. everything else in
# LANGUAGES still works, but goes through the multilingual model.
DEEPGRAM_NATIVE = {"en", "es", "fr", "de", "it", "pt", "nl", "ro", "tr",
                   "ja", "hi", "ru", "sv", "da", "no", "id", "pl"}

LANGUAGES = {
    "English (US)": "en",
    "Spanish": "es",
    "Turkish": "tr",
    "German": "de",
    "French": "fr",
    "Italian": "it",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Dutch": "nl",
    "Japanese": "ja",
}

# ---- palette, matching the mockup -------------------------------------

BG = "#ffffff"
PANEL_BG = "#faf9ff"
BORDER = "#e8e6f0"
TEXT = "#1a1a2e"
MUTED = "#8b8ba0"
ACCENT = "#6c4ee8"
ACCENT_SOFT = "#f0edfd"
SPEAKER_COLORS = ["#6c4ee8", "#e84e8a", "#4eb8e8", "#e8a34e", "#4ee89a"]
SIDEBAR_BG = "#f7f7fa"
SIDEBAR_ACTIVE = "#efecfd"


def secure_file(path):
    """Chmod 0600 - transcripts and keys are private to this user."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_env():
    path = os.path.join(HERE, ".env")
    env = {}
    if os.path.exists(path):
        # .env holds API keys; tighten it if it was created world-readable
        try:
            if os.stat(path).st_mode & 0o077:
                os.chmod(path, 0o600)
                print("tightened .env permissions to 0600")
        except OSError:
            pass
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return env


ENV = load_env()


def flag(name, default="0"):
    return ENV.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# =======================================================================
#  EVERYTHING BELOW IS DRIVEN BY .env - THIS IS THE ONLY PLACE TO CONFIGURE
#
#    TRANSCRIBE_MODE = local | deepgram      (speech -> text)
#    TRANSLATE_MODE  = gemini | openai | off           (text -> translation)
#    DEBUG           = 1                     (log to the console)
# =======================================================================

TRANSCRIBE_MODE = ENV.get("TRANSCRIBE_MODE", "local").strip().lower()
TRANSLATE_MODE = ENV.get("TRANSLATE_MODE", "gemini").strip().lower()
DEBUG = flag("DEBUG", "1")

# --- keys ---
DEEPGRAM_KEY = ENV.get("DEEPGRAM_API_KEY") or os.environ.get("DEEPGRAM_API_KEY", "")
GEMINI_KEY = ENV.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
TRANSLATE_KEY = ENV.get("TRANSLATE_API_KEY") or os.environ.get("TRANSLATE_API_KEY", "")

# --- gemini ---
# translation model chain, tried in order:
#   1. gemini-3.5-flash-lite   free tier, ~0.7s, handles everything normally
#   2. gemini-3.7-flash        free tier, used when 1 is rate limited
#   3. whatever PAID_* points at, on a separate billed key - last resort
GEMINI_MODEL = ENV.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACKS = [m.strip() for m in ENV.get(
    "GEMINI_FALLBACKS", "gemini-3.7-flash").split(",") if m.strip()]

# optional billed key. left empty until you buy one; the app just skips this
# tier while it is blank.
PAID_API_KEY = ENV.get("PAID_API_KEY") or os.environ.get("PAID_API_KEY", "")
PAID_MODEL = ENV.get("PAID_MODEL", "gemini-3.5-flash-lite")
GEMINI_ACTIVE = GEMINI_MODEL
GEMINI_TIMEOUT = float(ENV.get("GEMINI_TIMEOUT", "30"))

# batching: how many transcript lines go into ONE api request, and how long
# we wait to fill a batch. this is what keeps us under the per-minute /
# per-day REQUEST limits - one call for 20 lines instead of 20 calls.
BATCH_SIZE = int(ENV.get("BATCH_SIZE", "20"))
BATCH_WAIT_MS = int(ENV.get("BATCH_WAIT_MS", "1200"))
MIN_REQUEST_GAP_MS = int(ENV.get("MIN_REQUEST_GAP_MS", "1500"))

# how much of the conversation to send when drafting a reply. recent context
# is what matters - the whole meeting would be slower, pricier and vaguer.
REPLY_CONTEXT_LINES = int(ENV.get("REPLY_CONTEXT_LINES", "15"))

# how hard the suggested reply is allowed to be, in CEFR terms
CEFR_LEVELS = {
    "A2 · basic": "Use only very common everyday words and short simple "
                  "sentences. Present and past tense mostly. Nothing idiomatic.",
    "B1 · simple": "Use common everyday vocabulary and short, clear sentences. "
                   "Avoid idioms, phrasal verbs and any specialist wording. "
                   "A learner at B1 level must understand every word.",
    "B2 · natural": "Use ordinary conversational vocabulary. Keep sentences "
                    "short and clear. A few common phrasal verbs are fine, but "
                    "no rare words, no idioms, no jargon.",
    "C1 · advanced": "Use natural fluent language, including idioms and "
                     "richer vocabulary where it fits.",
}
DEFAULT_CEFR = ENV.get("CEFR_LEVEL", "B2 · natural")

# --- openai-compatible provider (groq / openai / xai / ...) ---
TRANSLATE_BASE = ENV.get("TRANSLATE_BASE_URL", "https://api.groq.com/openai/v1")
TRANSLATE_MODEL = ENV.get("TRANSLATE_MODEL", "llama-3.3-70b-versatile")

# --- local whisper ---
import audio as audio_backend

# --- audio capture ---
CAPTURE_SYSTEM = ENV.get("CAPTURE_SYSTEM", "1") != "0"
CAPTURE_MIC = ENV.get("CAPTURE_MIC", "0") == "1"
OUTPUT_DEVICE = ENV.get("OUTPUT_DEVICE", "default")
MIC_DEVICE = ENV.get("MIC_DEVICE", "default")
MIC_GAIN = float(ENV.get("MIC_GAIN", "1.0"))
# how far the meeting audio drops while you are speaking (0.25 = a quarter)
MIC_DUCK = float(ENV.get("MIC_DUCK", "0.25"))

SESSION_DIR = ENV.get("SESSION_DIR", os.path.join(HERE, "sessions"))
AUTOSAVE_SECONDS = int(ENV.get("AUTOSAVE_SECONDS", "30"))

try:
    import keyterms
    KEYTERM_PROMPT = keyterms.prompt()
except Exception as e:                      # file missing or has a typo
    KEYTERM_PROMPT = ""
    print(f"keyterms.py not loaded ({e})")

LOCAL_MODEL_DIR = ENV.get(
    "LOCAL_MODEL_DIR", os.path.join(HERE, "models", "whisper-large-v3-turbo-ct2"))


def active_model_dir():
    """Whichever local model is installed, preferring the configured one."""
    if os.path.exists(os.path.join(LOCAL_MODEL_DIR, "model.bin")):
        return LOCAL_MODEL_DIR
    have = installed_models()
    return model_path(have[0]) if have else LOCAL_MODEL_DIR

# --- derived state (do not set these in .env) ---
PROVIDERS = []              # filled in below, once helpers are defined
USE_GEMINI = TRANSLATE_MODE == "gemini" and bool(GEMINI_KEY)
USE_OPENAI = TRANSLATE_MODE in ("openai", "groq") and bool(TRANSLATE_KEY)
HAVE_TRANSLATION = USE_GEMINI or USE_OPENAI
DEEPGRAM_ENABLED = TRANSCRIBE_MODE == "deepgram" and bool(DEEPGRAM_KEY)


SPEAKER_NAMES = {}          # deepgram index -> name the user typed


def post_json(url, headers, payload, timeout):
    """POST JSON and return (status, text).

    same request urllib does in under a second, which was blowing the
    timeout and leaving cards stuck on "Translating...".
    """
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode()
    head = dict(headers)
    head.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=head, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


# ---- translation providers, stored as a list the user can edit ---------

PROVIDERS_FILE = os.path.join(HERE, "providers.json")

PROVIDER_PRESETS = {
    "Google Gemini": {
        "kind": "gemini",
        "base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-3.5-flash-lite",
        "help": "aistudio.google.com/apikey  ·  generous free tier",
    },
    "Groq": {
        "kind": "openai",
        "base": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "help": "console.groq.com/keys  ·  very fast, free tier",
    },
    "OpenAI": {
        "kind": "openai",
        "base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "help": "platform.openai.com/api-keys",
    },
    "xAI (Grok)": {
        "kind": "openai",
        "base": "https://api.x.ai/v1",
        "model": "grok-3-mini",
        "help": "console.x.ai",
    },
    "OpenRouter": {
        "kind": "openai",
        "base": "https://openrouter.ai/api/v1",
        "model": "google/gemini-flash-1.5",
        "help": "openrouter.ai/keys  ·  one key, many models",
    },
    "Custom (OpenAI-compatible)": {
        "kind": "openai",
        "base": "",
        "model": "",
        "help": "Any server that speaks the OpenAI chat API.",
    },
}


def load_providers():
    """Translation providers, in priority order. First working one wins."""
    if os.path.exists(PROVIDERS_FILE):
        try:
            with open(PROVIDERS_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return [p for p in data if isinstance(p, dict) and p.get("key")]
        except Exception as e:
            print(f"providers.json unreadable ({e})")

    # first run: adopt whatever is already in .env so nothing breaks
    out = []
    if ENV.get("GEMINI_API_KEY"):
        out.append({
            "name": "Google Gemini", "kind": "gemini",
            "base": PROVIDER_PRESETS["Google Gemini"]["base"],
            "model": ENV.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            "key": ENV["GEMINI_API_KEY"], "enabled": True,
        })
    if ENV.get("PAID_API_KEY"):
        out.append({
            "name": "Gemini (paid)", "kind": "gemini",
            "base": PROVIDER_PRESETS["Google Gemini"]["base"],
            "model": ENV.get("PAID_MODEL", "gemini-3.5-flash-lite"),
            "key": ENV["PAID_API_KEY"], "enabled": True,
        })
    if ENV.get("TRANSLATE_API_KEY"):
        out.append({
            "name": "OpenAI-compatible", "kind": "openai",
            "base": ENV.get("TRANSLATE_BASE_URL",
                            "https://api.groq.com/openai/v1"),
            "model": ENV.get("TRANSLATE_MODEL", "llama-3.3-70b-versatile"),
            "key": ENV["TRANSLATE_API_KEY"], "enabled": True,
        })
    return out


def save_providers(providers):
    tmp = PROVIDERS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(providers, f, indent=1)
    os.chmod(tmp, 0o600)
    os.replace(tmp, PROVIDERS_FILE)


def active_providers():
    return [p for p in PROVIDERS if p.get("enabled", True) and p.get("key")]


def test_provider(provider, timeout=25):
    """Send one tiny translation. Returns (ok, message, seconds)."""
    import time
    t = time.time()
    try:
        if provider["kind"] == "gemini":
            url = (f"{provider['base'].rstrip('/')}/models/"
                   f"{provider['model']}:generateContent")
            status, text = post_json(
                url, {"x-goog-api-key": provider["key"]},
                {"contents": [{"parts": [{"text": "Say OK"}]}],
                 "generationConfig": {"temperature": 0, "maxOutputTokens": 20}},
                timeout)
            if status == 200:
                json.loads(text)["candidates"][0]["content"]["parts"][0]["text"]
        else:
            url = f"{provider['base'].rstrip('/')}/chat/completions"
            status, text = post_json(
                url, {"Authorization": f"Bearer {provider['key']}"},
                {"model": provider["model"], "max_tokens": 20,
                 "messages": [{"role": "user", "content": "Say OK"}]},
                timeout)
            if status == 200:
                json.loads(text)["choices"][0]["message"]["content"]
        took = time.time() - t
        if status == 200:
            return True, f"working · {took:.1f}s", took
        if status == 429:
            return False, "rate limited (key works, quota spent)", took
        detail = ""
        try:
            body = json.loads(text)
            detail = (body.get("error", {}) or {}).get("message", "")[:70]
        except Exception:
            detail = text[:70]
        return False, f"HTTP {status} · {detail}", took
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:60]}", time.time() - t


def save_env(updates):
    """Update .env in place, preserving comments and unrelated keys.

    Written 0600 - it holds API keys.
    """
    path = os.path.join(HERE, ".env")
    lines = []
    if os.path.exists(path):
        lines = open(path).read().rstrip("\n").split("\n")

    seen = set()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                seen.add(key)
                value = updates[key]
                if value:
                    out.append(f"{key}={value}")
                continue            # blank value -> drop the line entirely
        out.append(line)

    for key, value in updates.items():
        if key not in seen and value:
            out.append(f"{key}={value}")

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out).rstrip("\n") + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    ENV.update({k: v for k, v in updates.items() if v})
    for k, v in updates.items():
        if not v:
            ENV.pop(k, None)


def mask(secret):
    """Show enough of a key to recognise it, never enough to use it."""
    if not secret:
        return "not set"
    if len(secret) <= 10:
        return "•" * len(secret)
    return f"{secret[:4]}{'•' * 8}{secret[-4:]}"


def write_transcript(path, data):
    """Write one saved session out as txt / md / srt."""
    lines = data.get("lines", [])

    def who(item):
        s = item.get("speaker")
        return "" if s is None else speaker_name(s) + ": "

    with open(path, "w", encoding="utf-8") as f:
        if path.endswith(".srt"):
            for i, item in enumerate(lines, 1):
                start = f"00:00:{i * 3 % 60:02d},000"
                end = f"00:00:{(i * 3 + 2) % 60:02d},000"
                f.write(f"{i}\n{start} --> {end}\n{who(item)}"
                        f"{item.get('text', '')}\n")
                if item.get("translation"):
                    f.write(item["translation"] + "\n")
                f.write("\n")
        elif path.endswith(".md"):
            f.write(f"# Meeting {data.get('started', '')}\n\n")
            f.write(f"*{data.get('source', '')} → {data.get('target', '')}"
                    f" · {data.get('engine', '')}*\n\n")
            for item in lines:
                f.write(f"**{item.get('stamp', '')}** {who(item)}"
                        f"{item.get('text', '')}\n\n")
                if item.get("translation"):
                    f.write(f"> {item['translation']}\n\n")
        else:
            f.write(f"Meeting {data.get('started', '')}  "
                    f"({data.get('source', '')} -> {data.get('target', '')})\n\n")
            for item in lines:
                f.write(f"[{item.get('stamp', '')}] {who(item)}"
                        f"{item.get('text', '')}\n")
                if item.get("translation"):
                    f.write(f"           {item['translation']}\n")


def combo_style():
    """Shared dropdown styling - the popup list needs explicit colours or it
    renders white-on-white under some system themes."""
    return (
        f"QComboBox{{background:#fff;border:1px solid {BORDER};"
        f"border-radius:6px;padding:2px 8px;font-size:11px;color:{TEXT};}}"
        f"QComboBox:hover{{border-color:{ACCENT};}}"
        "QComboBox::drop-down{border:none;width:18px;}"
        f"QComboBox::down-arrow{{image:none;border-left:4px solid transparent;"
        f"border-right:4px solid transparent;border-top:5px solid {MUTED};"
        "margin-right:6px;}"
        f"QComboBox QAbstractItemView{{background:#fff;color:{TEXT};"
        f"border:1px solid {BORDER};selection-background-color:{ACCENT};"
        "selection-color:#fff;outline:none;}")


def speaker_name(idx):
    return SPEAKER_NAMES.get(idx, f"Speaker {idx + 1}")


def log(*a):
    """Console trace - turn off with DEBUG=0 in .env."""
    if DEBUG:
        print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def transcribe_summary(local=None):
    """What is turning speech into text (pass the window's live state)."""
    if local is None:
        local = not DEEPGRAM_ENABLED
    if not local:
        return "Deepgram · cloud"
    name = os.path.basename(active_model_dir()).replace("-ct2", "")
    return f"{name} · local"


def translate_summary():
    """What is turning text into another language."""
    live = active_providers()
    if live:
        return f"{live[0]['model']} · cloud"
    if USE_GEMINI:
        return f"{GEMINI_ACTIVE} · cloud"
    if USE_OPENAI:
        return f"{TRANSLATE_MODEL} · cloud"
    return "off"


def engine_summary():
    return f"{transcribe_summary()}  ·  {translate_summary()}"


PROVIDERS = load_providers()
HAVE_TRANSLATION = bool(active_providers())

log("config: transcribe =", TRANSCRIBE_MODE, "| translate =", TRANSLATE_MODE)
log(f"config: {len(active_providers())} translation provider(s) enabled")
log("config:", engine_summary())
log(f"config: {len(KEYTERM_PROMPT.split(',')) if KEYTERM_PROMPT else 0} keyterms loaded")

# --- local capture tuning (cuts on speech pauses, not a fixed timer) ---
LOCAL_FRAME_MS = 100
LOCAL_SILENCE_RMS = float(ENV.get("LOCAL_SILENCE_RMS", "0.006"))
LOCAL_PAUSE_MS = int(ENV.get("LOCAL_PAUSE_MS", "250"))
LOCAL_MAX_UTTERANCE_S = float(ENV.get("LOCAL_MAX_UTTERANCE_S", "6"))
LOCAL_MIN_UTTERANCE_MS = 400
LOCAL_PREROLL_MS = 300

SENTENCE_END = (".", "!", "?", "。", "！", "？")
# ---- transcription worker ---------------------------------------------

class Listener(QThread):
    """Streams system audio to Deepgram and emits transcript pieces."""

    interim = Signal(int, str)          # speaker, text
    final = Signal(int, str, str, bool)  # speaker, text, timestamp, ended
    status = Signal(str)

    def __init__(self, source_lang):
        super().__init__()
        self.source_lang = source_lang
        self.running = True
        self.capture = None

    def url(self):
        # nova-3 streams these natively (verified against /v1/models); anything
        # else falls back to the code-switching multilingual model
        lang = (self.source_lang if self.source_lang in DEEPGRAM_NATIVE
                else "multi")
        return (
            "wss://api.deepgram.com/v1/listen"
            "?model=nova-3"
            f"&language={lang}"
            "&encoding=linear16"
            f"&sample_rate={RATE}"
            "&channels=1"
            "&diarize=true"
            "&punctuate=true"
            "&interim_results=true"
            "&endpointing=300"
            "&utterance_end_ms=1000"
            "&no_delay=true"
            "&vad_events=true"
        )

    def run(self):
        try:
            asyncio.run(self.stream())
        except Exception as e:
            self.status.emit(f"Error: {e}")

    async def stream(self):
        async with websockets.connect(
            self.url(), additional_headers={"Authorization": f"Token {DEEPGRAM_KEY}"}
        ) as ws:
            self.status.emit("LIVE")
            await asyncio.gather(self.send(ws), self.receive(ws))

    async def send(self, ws):
        cap = audio_backend.Capture(
            with_system=CAPTURE_SYSTEM, with_mic=CAPTURE_MIC,
            output_device=OUTPUT_DEVICE, mic_device=MIC_DEVICE,
            mic_gain=MIC_GAIN, frame_ms=CHUNK_MS, open_mic=True,
            duck=MIC_DUCK)
        self.capture = cap                  # so the UI can mute mid-session
        try:
            while self.running:
                data = await asyncio.to_thread(cap.read, CHUNK_BYTES)
                if not data:
                    break
                await ws.send(data)
            await ws.send(json.dumps({"type": "CloseStream"}))
        finally:
            cap.close()

    async def receive(self, ws):
        async for message in ws:
            if not self.running:
                break
            msg = json.loads(message)
            if msg.get("type") != "Results":
                continue
            alt = msg["channel"]["alternatives"][0]
            if not alt.get("transcript"):
                continue

            words = alt.get("words", [])
            speaker = words[0].get("speaker", 0) if words else 0
            stamp = datetime.now().strftime("%H:%M:%S")

            if msg.get("is_final"):
                # speech_final = deepgram thinks the whole utterance ended,
                # not just that this piece is settled
                ended = bool(msg.get("speech_final"))
                self.final.emit(speaker, alt["transcript"], stamp, ended)
            else:
                self.interim.emit(speaker, alt["transcript"])

    def stop(self):
        self.running = False


# ---- local transcription worker ---------------------------------------

class LocalListener(QThread):
    """Whisper on this machine. Cuts audio on speech pauses, not on a timer.

    A reader thread keeps pulling audio no matter how busy the GPU is, so
    transcription can never fall behind the microphone.
    """

    interim = Signal(int, str)
    final = Signal(int, str, str, bool)
    status = Signal(str)

    def __init__(self, source_lang):
        super().__init__()
        self.source_lang = source_lang
        self.running = True
        self.capture = None
        self.model = None
        self.utterances = queue.Queue()
        self.context = ""          # last thing said, fed back to whisper

    def load(self):
        from faster_whisper import WhisperModel
        path = active_model_dir()
        if not os.path.exists(os.path.join(path, "model.bin")):
            raise RuntimeError(f"model not found: {path}")
        try:
            self.status.emit("Loading turbo model on GPU...")
            return WhisperModel(path, device="cuda", compute_type="float16")
        except Exception:
            self.status.emit("GPU unavailable - loading on CPU...")
            return WhisperModel(path, device="cpu", compute_type="int8")

    # -- capture: runs in its own thread so the GPU never stalls it --

    def capture(self):
        import numpy as np

        cap = audio_backend.Capture(
            with_system=CAPTURE_SYSTEM, with_mic=CAPTURE_MIC,
            output_device=OUTPUT_DEVICE, mic_device=MIC_DEVICE,
            mic_gain=MIC_GAIN, frame_ms=LOCAL_FRAME_MS, open_mic=True,
            duck=MIC_DUCK)
        self.capture = cap                  # so the UI can mute mid-session

        frame_bytes = RATE * 2 * LOCAL_FRAME_MS // 1000
        preroll_frames = max(1, LOCAL_PREROLL_MS // LOCAL_FRAME_MS)
        pause_frames = max(1, LOCAL_PAUSE_MS // LOCAL_FRAME_MS)
        max_frames = int(LOCAL_MAX_UTTERANCE_S * 1000 // LOCAL_FRAME_MS)

        recent = deque(maxlen=preroll_frames)   # audio from just before speech
        speech, silence_run, talking = [], 0, False

        try:
            while self.running:
                raw = cap.read(frame_bytes)
                if not raw:
                    break
                frame = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                loud = float(np.sqrt(np.mean(frame ** 2))) >= LOCAL_SILENCE_RMS

                if not talking:
                    recent.append(frame)
                    if loud:                       # speech just started
                        talking = True
                        speech = list(recent) + []
                        recent.clear()
                        silence_run = 0
                    continue

                speech.append(frame)
                silence_run = 0 if loud else silence_run + 1

                # long enough pause, or the speaker is running on forever
                if silence_run >= pause_frames or len(speech) >= max_frames:
                    ended = silence_run >= pause_frames
                    audio = np.concatenate(speech)
                    speech, talking, silence_run = [], False, 0
                    if len(audio) >= RATE * LOCAL_MIN_UTTERANCE_MS // 1000:
                        self.utterances.put((audio, ended))

            if speech:                              # flush on stop
                self.utterances.put((np.concatenate(speech), True))
        finally:
            cap.close()
            self.utterances.put(None)

    # -- transcription --

    def run(self):
        try:
            self.model = self.load()
        except Exception as e:
            self.status.emit(f"Error: {e}")
            return

        reader = threading.Thread(target=self.capture, daemon=True)
        reader.start()
        self.status.emit("LIVE (local)")

        while self.running:
            try:
                item = self.utterances.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                break

            audio, ended = item
            segments, _ = self.model.transcribe(
                audio,
                language=self.source_lang,
                vad_filter=True,
                beam_size=5,                     # turbo is fast; buy accuracy
                condition_on_previous_text=False,
                initial_prompt=self.hint() or None,
                temperature=0,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if not text or not self.running:
                continue
            log(f"whisper: {len(audio) / RATE:.1f}s -> {text[:70]}")

            # carry a little context forward so names stay consistent
            self.context = (self.context + " " + text)[-200:]

            # local whisper has no diarization -> everyone is speaker 0
            self.final.emit(0, text, datetime.now().strftime("%H:%M:%S"),
                            ended or text.endswith(SENTENCE_END))

    def hint(self):
        """Keyterms plus the tail of what was just said."""
        parts = [p for p in (KEYTERM_PROMPT, self.context.strip()) if p]
        return " ".join(parts)[-900:]

    def stop(self):
        self.running = False


# ---- translation worker -----------------------------------------------

class Translator(QThread):
    """Translates a BATCH of lines in a single API request.

    One request per batch instead of one per line is what keeps us inside
    the per-minute and per-day request quotas.
    """

    done = Signal(list, list)      # job ids, translations (same order)
    failed = Signal(list, str)     # job ids, error
    status = Signal(str)

    def __init__(self, job_ids, texts, target_name):
        super().__init__()
        self.job_ids = job_ids
        self.texts = texts
        self.target = target_name

    def run(self):
        log(f"translate: batch of {len(self.texts)} lines -> {self.target} "
            f"({sum(len(t) for t in self.texts)} chars) mode={TRANSLATE_MODE}")
        if not active_providers():
            self.failed.emit(self.job_ids,
                             "no translation provider - add one in API Keys")
            return
        try:
            out = self.via_providers()
            log(f"translate: batch done ({len(out)} lines)")
            self.done.emit(self.job_ids, out)
        except Exception as e:
            import traceback
            log(f"translate: batch FAILED: {type(e).__name__}: {e}")
            if DEBUG:
                traceback.print_exc()
            self.failed.emit(self.job_ids, f"{type(e).__name__}: {e}")

    def via_providers(self):
        """Try each enabled provider in order until one answers."""
        problems = []
        for p in active_providers():
            try:
                self.status.emit(f"translating via {p['name']}…")
                out = (self.call_gemini(p) if p["kind"] == "gemini"
                       else self.call_openai(p))
                log(f"  <- {p['name']}: {len(out)} lines")
                return out
            except Exception as e:
                log(f"  <- {p['name']} failed: {type(e).__name__}: {e}")
                problems.append(f"{p['name']}: {type(e).__name__}")
        if not problems:
            raise RuntimeError("no translation provider configured - "
                               "add one under API Keys")
        raise RuntimeError("all providers failed - " + "; ".join(problems))

    def call_gemini(self, p):
        url = (f"{p['base'].rstrip('/')}/models/"
               f"{p['model']}:generateContent")
        log(f"  -> {p['name']} ({p['model']})")
        status, text = post_json(
            url, {"x-goog-api-key": p["key"]}, self.payload(), GEMINI_TIMEOUT)
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {text[:150]}")
        raw = json.loads(text)["candidates"][0]["content"]["parts"][0]["text"]
        return self.unpack(json.loads(raw))

    def unpack(self, data):
        """Turn whatever the model replied with into one string per input line.

        Models occasionally merge or drop a line. Rather than throwing the
        whole batch away we place what we got and mark the rest, so the user
        can retry just those lines.
        """
        n = len(self.texts)

        if isinstance(data, dict):
            # preferred shape: {"1": "...", "2": "..."}
            if any(k.strip().isdigit() for k in data):
                out = []
                for i in range(1, n + 1):
                    val = data.get(str(i), data.get(f" {i}"))
                    out.append(str(val).strip() if val is not None else "")
                return out
            # or a wrapper like {"lines": [...]}
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break

        if isinstance(data, list):
            out = [str(x).strip() for x in data]
            if len(out) < n:                 # model merged some lines
                log(f"  !! model returned {len(out)} of {n} lines")
                out += [""] * (n - len(out))
            return out[:n]

        raise ValueError(f"unexpected reply shape: {type(data).__name__}")

    def call_openai(self, p):
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(self.texts))
        log(f"  -> {p['name']} ({p['model']})")
        status, text = post_json(
            f"{p['base'].rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {p['key']}"},
            {"model": p["model"], "temperature": 0,
             "response_format": {"type": "json_object"},
             "messages": [
                 {"role": "system", "content":
                  f"Translate each numbered line into {self.target}. "
                  'Reply as {"lines": [...]} - one string per input line.'},
                 {"role": "user", "content": numbered}]},
            GEMINI_TIMEOUT)
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {text[:150]}")
        data = json.loads(json.loads(text)["choices"][0]["message"]["content"])
        return self.unpack(data)

    def payload(self):
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(self.texts))
        return {
            "systemInstruction": {"parts": [{"text":
                f"Translate each numbered line into {self.target}.\n\n"
                "Return a JSON object whose keys are the line numbers as "
                'strings, e.g. {"1": "...", "2": "..."}.\n'
                f"There are {len(self.texts)} lines, so the object must have "
                f"exactly {len(self.texts)} keys: "
                f"\"1\" through \"{len(self.texts)}\".\n"
                "Translate every line separately - never merge two lines, "
                "never skip one, even if a line is short, repeated or looks "
                "like a fragment. Output the translation only."}]},
            "contents": [{"parts": [{"text": numbered}]}],
            "generationConfig": {"temperature": 0,
                                 "responseMimeType": "application/json"},
        }

    def via_openai(self):
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(self.texts))
        status, text = post_json(
            f"{TRANSLATE_BASE}/chat/completions",
            {"Authorization": f"Bearer {TRANSLATE_KEY}"},
            {"model": TRANSLATE_MODEL, "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [
                      {"role": "system", "content":
                       f"Translate each numbered line into {self.target}. "
                       'Reply as {"lines": [...]} - one string per input line.'},
                      {"role": "user", "content": numbered}]},
            GEMINI_TIMEOUT)
        if status >= 400:
            raise RuntimeError(f"HTTP {status}: {text[:200]}")
        data = json.loads(json.loads(text)["choices"][0]["message"]["content"])
        return data["lines"] if isinstance(data, dict) else data

    def via_gemini(self):
        """Google's own REST shape - not OpenAI-compatible.

        Tries the preferred model first; on a transient failure (overloaded,
        rate limited) it moves to the next model in the chain rather than
        hammering one that is clearly busy.
        """
        import time
        global GEMINI_ACTIVE

        payload = self.payload()

        # free-tier models first, then the paid key if one is configured
        free = [GEMINI_ACTIVE] + [m for m in [GEMINI_MODEL] + GEMINI_FALLBACKS
                                  if m != GEMINI_ACTIVE]
        chain = [(m, GEMINI_KEY, False) for m in free]
        if PAID_API_KEY:
            chain.append((PAID_MODEL, PAID_API_KEY, True))
        problems = []

        for model, key, paid in chain:
            for attempt in range(2):          # brief retry before moving on
                try:
                    log(f"  -> POST {model}{' [PAID KEY]' if paid else ''} "
                        f"(try {attempt + 1}, timeout {GEMINI_TIMEOUT}s)")
                    status, text = post_json(
                        "https://generativelanguage.googleapis.com/v1beta/"
                        f"models/{model}:generateContent",
                        {"x-goog-api-key": key}, payload, GEMINI_TIMEOUT)
                    log(f"  <- {model}: HTTP {status}")
                except Exception as e:
                    log(f"  <- {model}: {type(e).__name__}: {e}")
                    problems.append(f"{model}: {type(e).__name__}")
                    break

                if status in (429, 500, 502, 503, 504):
                    if status == 429:
                        log(f"  !! {model} rate limited (429)")
                    if attempt == 0:
                        time.sleep(1.2)
                        continue
                    problems.append(f"{model}: {status}")
                    self.status.emit(f"{model} busy - trying next model...")
                    break

                if status >= 400:             # real error, chain won't help
                    log(f"  !! {model} hard error: {text[:300]}")
                    raise RuntimeError(f"HTTP {status}: {text[:200]}")

                if paid:
                    self.status.emit(f"using paid key · {model}")
                elif model != GEMINI_ACTIVE:
                    GEMINI_ACTIVE = model     # stick with what works
                    self.status.emit(f"Switched to {model}")
                raw = json.loads(text)["candidates"][0]["content"]["parts"][0]["text"]
                out = json.loads(raw)
                if isinstance(out, dict):     # some models wrap it
                    out = next(v for v in out.values() if isinstance(v, list))
                return [str(x).strip() for x in out]

        tail = ("" if PAID_API_KEY else
                " (set PAID_API_KEY in .env to add a billed fallback)")
        raise RuntimeError("all models unavailable - " + "; ".join(problems) + tail)


# ---- ui pieces ---------------------------------------------------------

# ---- local model catalogue -------------------------------------------

MODELS_DIR = ENV.get("MODELS_DIR", os.path.join(HERE, "models"))

WHISPER_MODELS = [
    ("tiny",     "Systran/faster-whisper-tiny",     "75 MB",   "fastest, rough"),
    ("base",     "Systran/faster-whisper-base",     "145 MB",  "fast, basic"),
    ("small",    "Systran/faster-whisper-small",    "480 MB",  "decent balance"),
    ("medium",   "Systran/faster-whisper-medium",   "1.5 GB",  "good accuracy"),
    ("large-v3", "Systran/faster-whisper-large-v3", "3.1 GB",  "best accuracy"),
    ("large-v3-turbo", "deepdml/faster-whisper-large-v3-turbo-ct2",
                                                    "1.6 GB",  "best speed/accuracy"),
]


def model_path(name):
    return os.path.join(MODELS_DIR, f"whisper-{name}-ct2")


def installed_models():
    """Names of models that are actually usable on disk."""
    out = []
    for name, *_ in WHISPER_MODELS:
        p = model_path(name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.bin")):
            out.append(name)
    # the model converted by hand earlier lives under its own name
    legacy = os.path.join(MODELS_DIR, "whisper-large-v3-turbo-ct2")
    if os.path.exists(os.path.join(legacy, "model.bin")) \
            and "large-v3-turbo" not in out:
        out.append("large-v3-turbo")
    return out


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class ModelDownloader(QThread):
    """Pulls one faster-whisper model from HuggingFace, reporting progress."""

    progress = Signal(int, str)      # percent, human readable
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, name, repo):
        super().__init__()
        self.name = name
        self.repo = repo
        self.cancelled = False

    def run(self):
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            self.failed.emit("huggingface_hub not installed")
            return

        target = model_path(self.name)
        os.makedirs(target, exist_ok=True)
        log(f"downloading {self.repo} -> {target}")

        watcher = threading.Thread(target=self.watch, args=(target,), daemon=True)
        watcher.start()
        try:
            snapshot_download(
                repo_id=self.repo,
                local_dir=target,
                allow_patterns=["*.bin", "*.json", "*.txt", "*.model"],
                max_workers=4,
            )
            self.cancelled = True
            if not os.path.exists(os.path.join(target, "model.bin")):
                raise RuntimeError("model.bin missing after download")
            self.progress.emit(100, "done")
            self.done.emit(self.name)
        except Exception as e:
            self.cancelled = True
            log(f"download failed: {e}")
            self.failed.emit(str(e)[:160])

    def watch(self, target):
        """Size on disk is the only reliable progress signal here."""
        expected = {"tiny": 75e6, "base": 145e6, "small": 480e6,
                    "medium": 1.5e9, "large-v3": 3.1e9,
                    "large-v3-turbo": 1.6e9}.get(self.name, 1e9)
        import time
        while not self.cancelled:
            size = dir_size(target)
            pct = min(99, int(size / expected * 100))
            self.progress.emit(pct, f"{size / 1e6:.0f} MB / {expected / 1e6:.0f} MB")
            time.sleep(0.7)


class ProviderTester(QThread):
    """Runs one provider check off the UI thread."""

    result = Signal(int, bool, str)

    def __init__(self, index, provider):
        super().__init__()
        self.index = index
        self.provider = provider

    def run(self):
        ok, msg, _ = test_provider(self.provider)
        self.result.emit(self.index, ok, msg)


class ProviderDialog(QDialog):
    """Add or edit one translation provider."""

    def __init__(self, parent=None, provider=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QLineEdit

        self.setWindowTitle("Edit provider" if provider else "Add provider")
        self.resize(480, 340)
        self.setStyleSheet(f"background:{BG};")
        self.provider = dict(provider or {})

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(9)

        self.preset = QComboBox()
        self.preset.addItems(list(PROVIDER_PRESETS))
        self.preset.setFixedHeight(28)
        self.preset.setStyleSheet(combo_style())
        self.preset.view().setStyleSheet(
            f"QAbstractItemView{{background:#fff;color:{TEXT};"
            f"selection-background-color:{ACCENT};selection-color:#fff;}}")
        self.preset.currentTextChanged.connect(self.apply_preset)
        v.addWidget(self.labelled("Service", self.preset))

        self.help = QLabel("")
        self.help.setWordWrap(True)
        self.help.setStyleSheet(f"color:{MUTED};font-size:10px;")
        v.addWidget(self.help)

        def line(placeholder, secret=False):
            e = QLineEdit()
            e.setPlaceholderText(placeholder)
            e.setFixedHeight(28)
            if secret:
                e.setEchoMode(QLineEdit.Password)
            e.setStyleSheet(
                f"QLineEdit{{background:#fff;border:1px solid {BORDER};"
                f"border-radius:6px;padding:2px 8px;font-size:11px;"
                f"color:{TEXT};}}"
                f"QLineEdit:focus{{border-color:{ACCENT};}}")
            return e

        self.name = line("shown in the list")
        self.base = line("https://...")
        self.model = line("model id")
        self.key = line("paste your API key", secret=True)
        v.addWidget(self.labelled("Name", self.name))
        v.addWidget(self.labelled("Endpoint", self.base))
        v.addWidget(self.labelled("Model", self.model))
        v.addWidget(self.labelled("API key", self.key))

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};font-size:11px;")
        v.addWidget(self.status)
        v.addStretch()

        row = QHBoxLayout()
        self.test_btn = self.button("Test", self.test, False)
        row.addWidget(self.test_btn)
        row.addStretch()
        row.addWidget(self.button("Cancel", self.reject, False))
        row.addWidget(self.button("Save", self.save, True))
        v.addLayout(row)

        if provider:
            self.name.setText(provider.get("name", ""))
            self.base.setText(provider.get("base", ""))
            self.model.setText(provider.get("model", ""))
            self.key.setText(provider.get("key", ""))
            for label, spec in PROVIDER_PRESETS.items():
                if spec["base"] == provider.get("base"):
                    self.preset.blockSignals(True)
                    self.preset.setCurrentText(label)
                    self.preset.blockSignals(False)
                    self.help.setText(spec["help"])
                    break
        else:
            self.apply_preset(self.preset.currentText())

    def labelled(self, text, widget):
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        lab = QLabel(text)
        lab.setFixedWidth(72)
        lab.setStyleSheet(f"color:{MUTED};font-size:11px;")
        h.addWidget(lab)
        h.addWidget(widget, 1)
        return box

    def button(self, label, slot, primary):
        b = QPushButton(label)
        b.setCursor(Qt.PointingHandCursor)
        b.setFixedHeight(30)
        b.setFixedWidth(88)
        b.clicked.connect(slot)
        b.setStyleSheet(
            (f"QPushButton{{background:{ACCENT};color:#fff;border:none;"
             if primary else
             f"QPushButton{{background:#fff;color:{TEXT};"
             f"border:1px solid {BORDER};")
            + "border-radius:7px;font-size:11px;font-weight:600;}"
            + (f"QPushButton:hover{{background:#5b3fd4;}}" if primary else
               f"QPushButton:hover{{border-color:{ACCENT};}}")
            + "QPushButton:disabled{color:#b0aac6;}")
        return b

    def apply_preset(self, label):
        spec = PROVIDER_PRESETS[label]
        self.help.setText(spec["help"])
        self.provider["kind"] = spec["kind"]
        if not self.name.text() or self.name.text() in PROVIDER_PRESETS:
            self.name.setText(label)
        self.base.setText(spec["base"])
        self.model.setText(spec["model"])

    def collect(self):
        return {
            "name": self.name.text().strip() or self.preset.currentText(),
            "kind": PROVIDER_PRESETS[self.preset.currentText()]["kind"],
            "base": self.base.text().strip(),
            "model": self.model.text().strip(),
            "key": self.key.text().strip(),
            "enabled": self.provider.get("enabled", True),
        }

    def test(self):
        p = self.collect()
        if not (p["key"] and p["base"] and p["model"]):
            self.status.setText("fill in endpoint, model and key first")
            return
        self.test_btn.setEnabled(False)
        self.status.setText("testing…")
        QApplication.processEvents()
        ok, msg, _ = test_provider(p)
        self.status.setText(("✓ " if ok else "✗ ") + msg)
        self.status.setStyleSheet(
            f"color:{'#0a8f4e' if ok else '#c2413f'};font-size:11px;")
        self.test_btn.setEnabled(True)

    def save(self):
        p = self.collect()
        if not p["key"]:
            self.status.setText("an API key is required")
            return
        if not (p["base"] and p["model"]):
            self.status.setText("endpoint and model are required")
            return
        self.provider = p
        self.accept()


class KeyManager(QDialog):
    """Deepgram key plus a list of translation providers you can manage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API keys")
        self.resize(620, 560)
        self.setStyleSheet(f"background:{BG};")
        self.testers = []

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(11)

        head = QLabel("API keys")
        head.setStyleSheet(f"color:{TEXT};font-size:14px;font-weight:700;")
        v.addWidget(head)

        # --- transcription (only one option today) ---
        cap = QLabel("TRANSCRIPTION")
        cap.setStyleSheet(
            f"color:{MUTED};font-size:10px;font-weight:700;letter-spacing:1px;")
        v.addWidget(cap)
        v.addWidget(self.deepgram_card())

        # --- translation providers ---
        row = QHBoxLayout()
        cap2 = QLabel("TRANSLATION PROVIDERS")
        cap2.setStyleSheet(
            f"color:{MUTED};font-size:10px;font-weight:700;letter-spacing:1px;")
        row.addWidget(cap2)
        row.addStretch()
        add = QPushButton("+  Add provider")
        add.setCursor(Qt.PointingHandCursor)
        add.setFixedHeight(26)
        add.clicked.connect(self.add_provider)
        add.setStyleSheet(
            f"QPushButton{{background:{ACCENT_SOFT};color:{ACCENT};"
            f"border:1px solid #ded6fb;border-radius:6px;padding:0 12px;"
            "font-size:11px;font-weight:600;}"
            f"QPushButton:hover{{background:#e6e0fb;}}")
        row.addWidget(add)
        v.addLayout(row)

        note = QLabel("Tried top to bottom - the first one that answers is "
                      "used. Drag order by removing and re-adding.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED};font-size:10px;")
        v.addWidget(note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        self.rows = QVBoxLayout(inner)
        self.rows.setContentsMargins(0, 2, 0, 2)
        self.rows.setSpacing(7)
        self.rows.addStretch()
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};font-size:11px;")
        v.addWidget(self.status)

        bottom = QHBoxLayout()
        test_all = QPushButton("Test all")
        test_all.setCursor(Qt.PointingHandCursor)
        test_all.setFixedHeight(32)
        test_all.clicked.connect(self.test_all)
        test_all.setStyleSheet(
            f"QPushButton{{background:#fff;color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:8px;padding:0 16px;"
            "font-size:12px;font-weight:600;}"
            f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};}}")
        bottom.addWidget(test_all)
        bottom.addStretch()
        close = QPushButton("Done")
        close.setCursor(Qt.PointingHandCursor)
        close.setFixedHeight(32)
        close.setFixedWidth(96)
        close.clicked.connect(self.accept)
        close.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#fff;border:none;"
            "border-radius:8px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#5b3fd4;}")
        bottom.addWidget(close)
        v.addLayout(bottom)

        self.reload()

    # -- deepgram (fixed) --

    def deepgram_card(self):
        from PySide6.QtWidgets import QLineEdit

        box = QFrame()
        box.setObjectName("card")
        box.setStyleSheet(
            f"#card{{background:#fff;border:1px solid {BORDER};"
            "border-radius:9px;}")
        col = QVBoxLayout(box)
        col.setContentsMargins(13, 10, 13, 11)
        col.setSpacing(6)

        top = QHBoxLayout()
        t = QLabel("Deepgram")
        t.setStyleSheet(f"color:{TEXT};font-size:12px;font-weight:600;")
        top.addWidget(t)
        sub = QLabel("speaker labels · console.deepgram.com")
        sub.setStyleSheet(f"color:{MUTED};font-size:10px;")
        top.addWidget(sub)
        top.addStretch()
        self.dg_state = QLabel(mask(ENV.get("DEEPGRAM_API_KEY", "")))
        self.dg_state.setStyleSheet(
            f"color:{'#0a8f4e' if ENV.get('DEEPGRAM_API_KEY') else MUTED};"
            "font-size:10px;")
        top.addWidget(self.dg_state)
        col.addLayout(top)

        h = QHBoxLayout()
        h.setSpacing(6)
        self.dg_edit = QLineEdit()
        self.dg_edit.setEchoMode(QLineEdit.Password)
        self.dg_edit.setPlaceholderText(
            "leave blank to keep current" if ENV.get("DEEPGRAM_API_KEY")
            else "paste your Deepgram key")
        self.dg_edit.setFixedHeight(28)
        self.dg_edit.setStyleSheet(
            f"QLineEdit{{background:#fff;border:1px solid {BORDER};"
            f"border-radius:6px;padding:2px 8px;font-size:11px;color:{TEXT};}}"
            f"QLineEdit:focus{{border-color:{ACCENT};}}")
        h.addWidget(self.dg_edit, 1)
        for label, slot in (("Save", self.save_deepgram),
                            ("Remove", self.remove_deepgram)):
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(28)
            b.setFixedWidth(70)
            b.clicked.connect(slot)
            b.setStyleSheet(
                f"QPushButton{{background:#fff;border:1px solid {BORDER};"
                f"border-radius:6px;font-size:11px;color:{TEXT};}}"
                f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};}}")
            h.addWidget(b)
        col.addLayout(h)
        return box

    def save_deepgram(self):
        val = self.dg_edit.text().strip()
        if not val:
            return
        save_env({"DEEPGRAM_API_KEY": val})
        self.dg_edit.clear()
        self.dg_state.setText(mask(val))
        self.dg_state.setStyleSheet("color:#0a8f4e;font-size:10px;")
        self.status.setText("Deepgram key saved · restart to apply")

    def remove_deepgram(self):
        save_env({"DEEPGRAM_API_KEY": ""})
        self.dg_edit.clear()
        self.dg_state.setText("not set")
        self.dg_state.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self.status.setText("Deepgram key removed · restart to apply")

    # -- provider list --

    def reload(self):
        while self.rows.count() > 1:
            w = self.rows.takeAt(0).widget()
            if w:
                w.setParent(None)
        self.result_labels = {}
        if not PROVIDERS:
            empty = QLabel("No translation provider yet — click "
                           "“Add provider”.")
            empty.setStyleSheet(f"color:{MUTED};font-size:11px;")
            self.rows.insertWidget(0, empty)
            return
        for i, p in enumerate(PROVIDERS):
            self.rows.insertWidget(self.rows.count() - 1, self.provider_row(i, p))

    def provider_row(self, index, p):
        on = p.get("enabled", True)
        box = QFrame()
        box.setObjectName("card")
        box.setStyleSheet(
            f"#card{{background:#fff;border:1px solid "
            f"{'#cfe8da' if on else BORDER};border-radius:9px;}}")
        h = QHBoxLayout(box)
        h.setContentsMargins(13, 9, 11, 10)
        h.setSpacing(9)

        order = QLabel(f"{index + 1}")
        order.setFixedWidth(14)
        order.setStyleSheet(f"color:{MUTED};font-size:11px;font-weight:700;")
        h.addWidget(order)

        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel(p.get("name", "provider"))
        title.setStyleSheet(
            f"color:{TEXT if on else MUTED};font-size:12px;font-weight:600;")
        col.addWidget(title)
        meta = QLabel(f"{p.get('model', '')}  ·  {mask(p.get('key', ''))}")
        meta.setStyleSheet(f"color:{MUTED};font-size:10px;")
        col.addWidget(meta)
        h.addLayout(col, 1)

        res = QLabel("")
        res.setStyleSheet(f"color:{MUTED};font-size:10px;")
        res.setFixedWidth(150)
        res.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.result_labels[index] = res
        h.addWidget(res)

        for label, slot in (
            ("Test", lambda: self.test_one(index)),
            ("On" if on else "Off", lambda: self.toggle(index)),
            ("Edit", lambda: self.edit(index)),
            ("✕", lambda: self.remove(index)),
        ):
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(26)
            b.setFixedWidth(26 if label == "✕" else 52)
            b.clicked.connect(slot)
            danger = label == "✕"
            b.setStyleSheet(
                f"QPushButton{{background:#fff;"
                f"color:{'#c2413f' if danger else (ACCENT if on else MUTED)};"
                f"border:1px solid {'#f0c9c8' if danger else BORDER};"
                "border-radius:6px;font-size:10px;font-weight:600;}"
                f"QPushButton:hover{{border-color:"
                f"{'#c2413f' if danger else ACCENT};}}")
            h.addWidget(b)
        return box

    # -- actions --

    def add_provider(self):
        dlg = ProviderDialog(self)
        if dlg.exec():
            PROVIDERS.append(dlg.provider)
            save_providers(PROVIDERS)
            self.status.setText(f"added {dlg.provider['name']}")
            log(f"provider added: {dlg.provider['name']}")
            self.reload()

    def edit(self, index):
        dlg = ProviderDialog(self, PROVIDERS[index])
        if dlg.exec():
            PROVIDERS[index] = dlg.provider
            save_providers(PROVIDERS)
            self.status.setText(f"updated {dlg.provider['name']}")
            self.reload()

    def remove(self, index):
        name = PROVIDERS[index].get("name", "provider")
        if QMessageBox.question(self, "Remove provider",
                                f"Remove {name}?") != QMessageBox.Yes:
            return
        PROVIDERS.pop(index)
        save_providers(PROVIDERS)
        log(f"provider removed: {name}")
        self.status.setText(f"removed {name}")
        self.reload()

    def toggle(self, index):
        PROVIDERS[index]["enabled"] = not PROVIDERS[index].get("enabled", True)
        save_providers(PROVIDERS)
        self.reload()

    def test_one(self, index):
        label = self.result_labels.get(index)
        if label:
            label.setText("testing…")
        t = ProviderTester(index, PROVIDERS[index])
        t.result.connect(self.on_result)
        t.finished.connect(lambda w=t: QTimer.singleShot(
            0, lambda: self.testers.remove(w) if w in self.testers else None))
        self.testers.append(t)
        t.start()

    def test_all(self):
        self.status.setText("testing every provider…")
        for i in range(len(PROVIDERS)):
            self.test_one(i)

    def on_result(self, index, ok, msg):
        label = self.result_labels.get(index)
        if label:
            label.setText(("✓ " if ok else "✗ ") + msg)
            label.setStyleSheet(
                f"color:{'#0a8f4e' if ok else '#c2413f'};font-size:10px;")


class AudioSettings(QDialog):
    """Choose what gets recorded: the meeting, your microphone, or both."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QCheckBox, QSlider

        self.setWindowTitle("Audio sources")
        self.resize(540, 400)
        self.setStyleSheet(f"background:{BG};")

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(11)

        head = QLabel("Audio sources")
        head.setStyleSheet(f"color:{TEXT};font-size:14px;font-weight:700;")
        v.addWidget(head)

        plat = QLabel(f"{platform.system()} · {audio_backend.backend_name() or 'no backend'}")
        plat.setStyleSheet(f"color:{MUTED};font-size:10px;")
        v.addWidget(plat)

        problem = audio_backend.missing_dependency()
        if problem:
            warn = QLabel(problem)
            warn.setWordWrap(True)
            warn.setStyleSheet(
                "color:#c2413f;background:#fdf2f2;border:1px solid #f0c9c8;"
                "border-radius:8px;padding:9px;font-size:11px;")
            v.addWidget(warn)

        # --- system output ---
        self.sys_check = QCheckBox("  Meeting audio (what you hear)")
        self.sys_check.setChecked(CAPTURE_SYSTEM)
        self.sys_check.setStyleSheet(
            f"QCheckBox{{color:{TEXT};font-size:12px;font-weight:600;}}")
        v.addWidget(self.sys_check)

        self.out_picker = QComboBox()
        self.outputs = audio_backend.list_outputs()
        for dev_id, label in self.outputs:
            self.out_picker.addItem(label, dev_id)
        self._select(self.out_picker, OUTPUT_DEVICE)
        self.out_picker.setFixedHeight(28)
        self.out_picker.setStyleSheet(combo_style())
        v.addWidget(self.indent(self.out_picker))

        note = QLabel("Records everything your speakers play - Zoom, Teams, "
                      "Meet, any app. Notification sounds included.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED};font-size:10px;padding-left:22px;")
        v.addWidget(note)

        # --- microphone ---
        self.mic_check = QCheckBox("  My microphone (what you say)")
        self.mic_check.setChecked(CAPTURE_MIC)
        self.mic_check.setStyleSheet(
            f"QCheckBox{{color:{TEXT};font-size:12px;font-weight:600;}}")
        v.addWidget(self.mic_check)

        self.mic_picker = QComboBox()
        self.mics = audio_backend.list_microphones()
        for dev_id, label in self.mics:
            self.mic_picker.addItem(label, dev_id)
        self._select(self.mic_picker, MIC_DEVICE)
        self.mic_picker.setFixedHeight(28)
        self.mic_picker.setStyleSheet(combo_style())
        v.addWidget(self.indent(self.mic_picker))

        gain_row = QWidget()
        gl = QHBoxLayout(gain_row)
        gl.setContentsMargins(22, 0, 0, 0)
        gl.setSpacing(8)
        gl.addWidget(self._small("Mic level"))
        self.gain = QSlider(Qt.Horizontal)
        self.gain.setRange(25, 300)
        self.gain.setValue(int(MIC_GAIN * 100))
        self.gain.setFixedHeight(20)
        gl.addWidget(self.gain, 1)
        self.gain_label = self._small(f"{MIC_GAIN:.2f}x")
        self.gain_label.setFixedWidth(40)
        self.gain.valueChanged.connect(
            lambda v_: self.gain_label.setText(f"{v_ / 100:.2f}x"))
        gl.addWidget(self.gain_label)
        v.addWidget(gain_row)

        duck_row = QWidget()
        dl = QHBoxLayout(duck_row)
        dl.setContentsMargins(22, 0, 0, 0)
        dl.setSpacing(8)
        dl.addWidget(self._small("Duck others"))
        self.duck = QSlider(Qt.Horizontal)
        self.duck.setRange(10, 100)          # 100 = no ducking
        self.duck.setValue(int(MIC_DUCK * 100))
        self.duck.setFixedHeight(20)
        dl.addWidget(self.duck, 1)
        self.duck_label = self._small(f"{MIC_DUCK:.2f}x")
        self.duck_label.setFixedWidth(40)
        self.duck.valueChanged.connect(
            lambda v_: self.duck_label.setText(f"{v_ / 100:.2f}x"))
        dl.addWidget(self.duck_label)
        v.addWidget(duck_row)

        note2 = QLabel("Turn this on so your own voice appears in the "
                       "transcript too. “Duck others” quietens the meeting "
                       "while you speak, so your words are not buried under "
                       "it — the usual reason your own voice goes missing.")
        note2.setWordWrap(True)
        note2.setStyleSheet(f"color:{MUTED};font-size:10px;padding-left:22px;")
        v.addWidget(note2)

        v.addStretch()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};font-size:11px;")
        v.addWidget(self.status)

        row = QHBoxLayout()
        test = QPushButton("Test 2 seconds")
        test.setCursor(Qt.PointingHandCursor)
        test.setFixedHeight(32)
        test.clicked.connect(self.test)
        test.setStyleSheet(
            f"QPushButton{{background:#fff;color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:8px;padding:0 16px;"
            "font-size:12px;font-weight:600;}"
            f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};}}")
        row.addWidget(test)
        row.addStretch()
        save = QPushButton("Save")
        save.setCursor(Qt.PointingHandCursor)
        save.setFixedHeight(32)
        save.setFixedWidth(96)
        save.clicked.connect(self.save)
        save.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#fff;border:none;"
            "border-radius:8px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#5b3fd4;}")
        row.addWidget(save)
        v.addLayout(row)

    def _small(self, text):
        lab = QLabel(text)
        lab.setStyleSheet(f"color:{MUTED};font-size:11px;")
        return lab

    def indent(self, widget):
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(22, 0, 0, 0)
        h.addWidget(widget)
        return box

    def _select(self, picker, dev_id):
        for i in range(picker.count()):
            if picker.itemData(i) == dev_id:
                picker.setCurrentIndex(i)
                return

    def test(self):
        """Record briefly and report the level, so you can see it works."""
        if not (self.sys_check.isChecked() or self.mic_check.isChecked()):
            self.status.setText("enable at least one source")
            return
        self.status.setText("recording 2 seconds…")
        QApplication.processEvents()
        try:
            import time
            cap = audio_backend.Capture(
                with_system=self.sys_check.isChecked(),
                with_mic=self.mic_check.isChecked(),
                output_device=self.out_picker.currentData(),
                mic_device=self.mic_picker.currentData(),
                mic_gain=self.gain.value() / 100,
                duck=self.duck.value() / 100)
            start, buf = time.time(), b""
            while time.time() - start < 2:
                buf += cap.read()
            cap.close()
        except Exception as e:
            self.status.setText(f"failed: {e}")
            self.status.setStyleSheet("color:#c2413f;font-size:11px;")
            return

        import numpy as _np
        a = _np.frombuffer(buf, _np.int16).astype(float)
        rms = float(_np.sqrt(_np.mean(a ** 2))) if len(a) else 0.0
        level = min(20, int(rms / 800))
        bar = "█" * level + "·" * (20 - level)
        if rms < 30:
            msg, colour = f"{bar}  silent - is anything playing?", "#c2413f"
        else:
            msg, colour = f"{bar}  level {rms:.0f} - working", "#0a8f4e"
        self.status.setText(msg)
        self.status.setStyleSheet(f"color:{colour};font-size:11px;")

    def save(self):
        if not (self.sys_check.isChecked() or self.mic_check.isChecked()):
            self.status.setText("enable at least one source")
            return
        save_env({
            "CAPTURE_SYSTEM": "1" if self.sys_check.isChecked() else "0",
            "CAPTURE_MIC": "1" if self.mic_check.isChecked() else "0",
            "OUTPUT_DEVICE": self.out_picker.currentData() or "default",
            "MIC_DEVICE": self.mic_picker.currentData() or "default",
            "MIC_GAIN": f"{self.gain.value() / 100:.2f}",
            "MIC_DUCK": f"{self.duck.value() / 100:.2f}",
        })
        global CAPTURE_SYSTEM, CAPTURE_MIC, OUTPUT_DEVICE, MIC_DEVICE
        global MIC_GAIN, MIC_DUCK
        CAPTURE_SYSTEM = self.sys_check.isChecked()
        CAPTURE_MIC = self.mic_check.isChecked()
        OUTPUT_DEVICE = self.out_picker.currentData() or "default"
        MIC_DEVICE = self.mic_picker.currentData() or "default"
        MIC_GAIN = self.gain.value() / 100
        MIC_DUCK = self.duck.value() / 100
        log(f"audio: system={CAPTURE_SYSTEM} mic={CAPTURE_MIC} "
            f"gain={MIC_GAIN:.2f}")

        # push straight into a running session - no restart needed
        win = self.parent()
        if win is not None:
            cap = getattr(getattr(win, "listener", None), "capture", None)
            if cap is not None:
                cap.set_system(CAPTURE_SYSTEM)
                cap.set_mic(CAPTURE_MIC)
                cap.set_gain(MIC_GAIN)
                cap.duck = MIC_DUCK
            win.mic_on = CAPTURE_MIC
            win.paint_mic()
        self.accept()


class ModelManager(QDialog):
    """Popup: see what is installed, download, or delete."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Local Whisper models")
        self.resize(560, 440)
        self.setStyleSheet(f"background:{BG};")
        self.worker = None

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(10)

        head = QLabel("Local models")
        head.setStyleSheet(f"color:{TEXT};font-size:14px;font-weight:700;")
        v.addWidget(head)

        sub = QLabel("Downloaded models run entirely on your GPU. "
                     "Pick one, download it, then choose Local in the top bar.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{MUTED};font-size:11px;")
        v.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        self.rows = QVBoxLayout(inner)
        self.rows.setContentsMargins(0, 4, 0, 4)
        self.rows.setSpacing(7)
        self.rows.addStretch()
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color:{MUTED};font-size:11px;")
        v.addWidget(self.status)

        close = QPushButton("Close")
        close.setCursor(Qt.PointingHandCursor)
        close.setFixedHeight(32)
        close.clicked.connect(self.accept)
        close.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#fff;border:none;"
            "border-radius:8px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#5b3fd4;}")
        v.addWidget(close)

        self.reload()

    # -- rows --

    def reload(self):
        while self.rows.count() > 1:
            w = self.rows.takeAt(0).widget()
            if w:
                w.setParent(None)
        have = installed_models()
        for name, repo, size, note in WHISPER_MODELS:
            self.rows.insertWidget(self.rows.count() - 1,
                                   self.row(name, repo, size, note, name in have))

    def row(self, name, repo, size, note, installed):
        box = QFrame()
        box.setObjectName("card")
        box.setStyleSheet(
            f"#card{{background:#fff;border:1px solid "
            f"{'#cfe8da' if installed else BORDER};border-radius:9px;}}")
        h = QHBoxLayout(box)
        h.setContentsMargins(13, 9, 11, 10)
        h.setSpacing(10)

        left = QVBoxLayout()
        left.setSpacing(2)
        title = QLabel(f"whisper {name}" + ("   ✓ installed" if installed else ""))
        title.setStyleSheet(
            f"color:{'#0a8f4e' if installed else TEXT};font-size:12px;"
            "font-weight:600;")
        left.addWidget(title)
        meta = QLabel(f"{size} · {note}")
        meta.setStyleSheet(f"color:{MUTED};font-size:11px;")
        left.addWidget(meta)
        h.addLayout(left, 1)

        bar = QProgressBar()
        bar.setFixedWidth(130)
        bar.setFixedHeight(18)
        bar.setTextVisible(True)
        bar.setVisible(False)
        bar.setStyleSheet(
            f"QProgressBar{{border:1px solid {BORDER};border-radius:9px;"
            f"background:#f4f3f9;font-size:9px;color:{MUTED};text-align:center;}}"
            f"QProgressBar::chunk{{background:{ACCENT};border-radius:8px;}}")
        h.addWidget(bar)

        btn = QPushButton("Remove" if installed else "Download")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(28)
        btn.setFixedWidth(88)
        btn.setStyleSheet(
            f"QPushButton{{background:#fff;color:{'#c2413f' if installed else ACCENT};"
            f"border:1px solid {'#f0c9c8' if installed else '#ded6fb'};"
            "border-radius:7px;font-size:11px;font-weight:600;}"
            f"QPushButton:hover{{background:"
            f"{'#fdf2f2' if installed else ACCENT_SOFT};}}"
            "QPushButton:disabled{color:#b0aac6;border-color:#e5e2f0;}")
        if installed:
            btn.clicked.connect(lambda: self.remove(name))
        else:
            btn.clicked.connect(lambda: self.download(name, repo, bar, btn))
        h.addWidget(btn)
        return box

    # -- actions --

    def download(self, name, repo, bar, btn):
        if self.worker and self.worker.isRunning():
            self.status.setText("a download is already running")
            return
        bar.setVisible(True)
        bar.setValue(0)
        btn.setEnabled(False)
        btn.setText("…")
        self.status.setText(f"downloading whisper {name}…")

        self.worker = ModelDownloader(name, repo)
        self.worker.progress.connect(
            lambda pct, txt: (bar.setValue(pct), bar.setFormat(txt)))
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_done(self, name):
        self.status.setText(f"whisper {name} installed")
        self.reload()

    def on_failed(self, err):
        self.status.setText(f"failed: {err}")
        self.reload()

    def remove(self, name):
        path = model_path(name)
        if not os.path.isdir(path):
            path = os.path.join(MODELS_DIR, "whisper-large-v3-turbo-ct2")
        ok = QMessageBox.question(
            self, "Remove model",
            f"Delete whisper {name}?\n\n{path}\n"
            f"({dir_size(path) / 1e9:.1f} GB)")
        if ok != QMessageBox.Yes:
            return
        import shutil
        shutil.rmtree(path, ignore_errors=True)
        log(f"removed model {name}")
        self.status.setText(f"removed whisper {name}")
        self.reload()


class Replier(QThread):
    """Asks the model what to say next, given the recent conversation."""

    done = Signal(str)
    failed = Signal(str)

    def __init__(self, lines, language, level):
        super().__init__()
        self.lines = lines
        self.language = language
        self.level = level

    def run(self):
        log(f"reply: drafting from last {len(self.lines)} lines")
        if not active_providers():
            self.failed.emit("no translation provider - add one in API Keys")
            return
        convo = "\n".join(self.lines)
        payload = {
            "systemInstruction": {"parts": [{"text":
                "You are already in this conversation. Say the next thing.\n\n"
                f"Talk like a normal person in {self.language} - the way people "
                "actually speak out loud, not the way they write. Around 30 to "
                "60 seconds of talking.\n\n"
                f"LANGUAGE LEVEL - this matters most: {self.level}\n"
                "Keep every sentence easy to follow. If a simpler word exists, "
                "use it.\n\n"
                "Sound human:\n"
                "- Use everyday words. Short sentences. Contractions.\n"
                "- Start naturally, the way people really do - \"yeah\", "
                "\"okay so\", \"right\", \"hmm\", \"got it\".\n"
                "- React first, like a person would, then say your bit.\n"
                "- One simple idea is enough. Don't cram in three points.\n"
                "- It's fine to be casual, to trail off, to think out loud.\n\n"
                "Avoid completely: corporate words like leverage, align, "
                "circle back, deep dive, moving forward, synergy, "
                "double down, key takeaway, drive impact. No lists. No "
                "summarising what was just said back at them. Don't sound "
                "like an email or a consultant.\n\n"
                "Just the words to say - nothing else."}]},
            "contents": [{"parts": [{"text":
                f"{convo}\n\nWhat do I say?"}]}],
            "generationConfig": {"temperature": 0.9, "maxOutputTokens": 400},
        }
        for p in active_providers():
            try:
                if p["kind"] == "gemini":
                    status, text = post_json(
                        f"{p['base'].rstrip('/')}/models/"
                        f"{p['model']}:generateContent",
                        {"x-goog-api-key": p["key"]}, payload, GEMINI_TIMEOUT)
                    if status != 200:
                        raise RuntimeError(f"HTTP {status}")
                    out = json.loads(text)["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    sys_txt = payload["systemInstruction"]["parts"][0]["text"]
                    usr_txt = payload["contents"][0]["parts"][0]["text"]
                    status, text = post_json(
                        f"{p['base'].rstrip('/')}/chat/completions",
                        {"Authorization": f"Bearer {p['key']}"},
                        {"model": p["model"], "temperature": 0.9,
                         "max_tokens": 400,
                         "messages": [{"role": "system", "content": sys_txt},
                                      {"role": "user", "content": usr_txt}]},
                        GEMINI_TIMEOUT)
                    if status != 200:
                        raise RuntimeError(f"HTTP {status}")
                    out = json.loads(text)["choices"][0]["message"]["content"]
                log(f"reply: drafted via {p['name']}")
                self.done.emit(out.strip())
                return
            except Exception as e:
                log(f"  reply: {p['name']} failed: {type(e).__name__}: {e}")
        self.failed.emit("no provider answered - check API Keys")


class LineCard(QFrame):
    """One utterance: the text, its translation, and per-line controls.

    Hovering reveals retry / edit / copy so a bad line can be fixed without
    redoing the whole session.
    """

    retry = Signal(object)
    edit = Signal(object)

    def __init__(self, index, text, stamp, speaker=None):
        super().__init__()
        self.setObjectName("card")
        self.index = index
        self.text = text
        self.translation = ""
        self.state = "new"          # new | working | done | failed
        self.hovered = False
        self.speaker = speaker

        box = QVBoxLayout(self)
        box.setContentsMargins(13, 10, 13, 11)
        box.setSpacing(5)

        head = QHBoxLayout()
        head.setSpacing(8)

        if speaker is not None:
            colour = SPEAKER_COLORS[speaker % len(SPEAKER_COLORS)]
            chip = QLabel(f"S{speaker + 1}")
            chip.setFixedSize(24, 24)
            chip.setAlignment(Qt.AlignCenter)
            chip.setStyleSheet(
                f"background:{colour}1a;color:{colour};border-radius:12px;"
                "font-size:10px;font-weight:700;")
            head.addWidget(chip)
            who = QLabel(speaker_name(speaker))
            who.setStyleSheet(
                f"color:{colour};font-size:11px;font-weight:600;")
            head.addWidget(who)

        self.when = QLabel(stamp)
        self.when.setStyleSheet(f"color:{MUTED};font-size:10px;")
        head.addWidget(self.when)
        head.addStretch()

        self.tools = QWidget()
        tl = QHBoxLayout(self.tools)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(2)
        for glyph, tip, slot in (
            ("↻", "Translate again", lambda: self.retry.emit(self)),
            ("✎", "Edit this line", lambda: self.edit.emit(self)),
            ("⧉", "Copy", self.copy_out),
        ):
            b = QPushButton(glyph)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.setFixedSize(22, 22)
            b.setStyleSheet(
                f"QPushButton{{border:none;background:transparent;color:{MUTED};"
                "font-size:12px;border-radius:5px;}"
                f"QPushButton:hover{{color:{ACCENT};}}")
            b.clicked.connect(slot)
            tl.addWidget(b)
        # kept in the layout permanently: toggling visibility would reflow the
        # card and make it jump under the cursor
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self.tools_fade = QGraphicsOpacityEffect(self.tools)
        self.tools_fade.setOpacity(0.0)
        self.tools.setGraphicsEffect(self.tools_fade)
        head.addWidget(self.tools)
        box.addLayout(head)

        self.body = QLabel(text)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setStyleSheet(
            f"color:{TEXT};font-size:13px;background:transparent;")
        box.addWidget(self.body)

        self.out = QLabel("")
        self.out.setWordWrap(True)
        self.out.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.out.setStyleSheet(
            f"color:{ACCENT};font-size:13px;font-weight:600;"
            f"background:transparent;border-top:1px solid {BORDER};"
            "padding-top:6px;")
        self.out.setVisible(False)
        box.addWidget(self.out)

        self.paint()

    # -- interaction --

    def enterEvent(self, e):
        self.hovered = True
        self.tools_fade.setOpacity(1.0)
        self.paint()

    def leaveEvent(self, e):
        self.hovered = False
        self.tools_fade.setOpacity(0.0)
        self.paint()

    def copy_out(self):
        QApplication.clipboard().setText(self.translation or self.text)

    # -- state --

    def set_text(self, text):
        self.text = text
        self.body.setText(text)

    def set_state(self, state, translation=""):
        self.state = state
        if state == "working":
            self.out.setVisible(True)
            self.out.setText("Translating…")
            self.out.setStyleSheet(
                f"color:{MUTED};font-size:12px;font-style:italic;"
                f"background:transparent;border-top:1px solid {BORDER};"
                "padding-top:6px;")
        elif state == "done":
            self.translation = translation
            self.out.setVisible(True)
            self.out.setText(translation)
            self.out.setStyleSheet(
                f"color:{ACCENT};font-size:13px;font-weight:600;"
                f"background:transparent;border-top:1px solid {BORDER};"
                "padding-top:6px;")
        elif state == "failed":
            self.out.setVisible(True)
            self.out.setText(f"⚠ {translation}  — click ↻ to retry")
            self.out.setStyleSheet(
                "color:#c2413f;font-size:12px;background:transparent;"
                f"border-top:1px solid {BORDER};padding-top:6px;")
        self.paint()

    def paint(self):
        edge = {"new": BORDER, "working": "#d9d0f7",
                "done": "#cfe8da", "failed": "#f0c9c8"}[self.state]
        # the left stripe identifies the speaker; the outline shows progress
        stripe = (SPEAKER_COLORS[self.speaker % len(SPEAKER_COLORS)]
                  if self.speaker is not None else edge)
        bg = "#fbfaff" if self.hovered else "#ffffff"
        self.setStyleSheet(
            f"#card{{background:{bg};border:1px solid {edge};"
            f"border-left:3px solid {stripe};border-radius:10px;}}")


class Card(QFrame):
    """One utterance bubble."""

    def __init__(self, speaker=None, text="", stamp="", translated=False):
        super().__init__()
        self.translated = translated
        self.setObjectName("card")
        self.active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 11, 14, 12)
        outer.setSpacing(6)

        if not translated:
            head = QHBoxLayout()
            head.setSpacing(9)

            color = SPEAKER_COLORS[(speaker or 0) % len(SPEAKER_COLORS)]
            chip = QLabel(f"S{(speaker or 0) + 1}")
            chip.setFixedSize(26, 26)
            chip.setAlignment(Qt.AlignCenter)
            chip.setStyleSheet(
                f"background:{color}1a;color:{color};border-radius:13px;"
                "font-size:10px;font-weight:700;")
            head.addWidget(chip)

            name = QLabel("Speaker" if speaker is None
                          else f"Speaker {speaker + 1}")
            name.setStyleSheet(f"color:{TEXT};font-size:12px;font-weight:600;")
            head.addWidget(name)
            head.addStretch()

            when = QLabel(stamp)
            when.setStyleSheet(f"color:{MUTED};font-size:11px;")
            head.addWidget(when)
            outer.addLayout(head)

        self.body = QLabel(text)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body.setStyleSheet(
            f"color:{ACCENT if translated else TEXT};font-size:13px;"
            "background:transparent;")
        outer.addWidget(self.body)

        self.paint()

    def set_text(self, text):
        self.body.setText(text)

    def set_active(self, on):
        self.active = on
        self.paint()

    def paint(self):
        if self.active:
            edge = ACCENT
            left = f"border-left:3px solid {ACCENT};"
            bg = ACCENT_SOFT if self.translated else "#ffffff"
        else:
            edge = BORDER
            left = f"border-left:1px solid {BORDER};"
            bg = "#ffffff"
        self.setStyleSheet(
            f"#card{{background:{bg};border:1px solid {edge};{left}"
            "border-radius:10px;}")


class Pane(QWidget):
    """A titled, scrolling column of cards."""

    def __init__(self, title, languages, default, button=None):
        super().__init__()
        self.cards = []
        self.action = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bar = QWidget()
        bar.setFixedHeight(46)
        bar.setStyleSheet(f"background:transparent;border-bottom:1px solid {BORDER};")
        row = QHBoxLayout(bar)
        row.setContentsMargins(20, 0, 16, 0)

        label = QLabel(title)
        label.setStyleSheet(
            f"color:{MUTED};font-size:11px;font-weight:700;letter-spacing:1px;")
        row.addWidget(label)
        row.addStretch()

        self.picker = QComboBox()
        self.picker.addItems(languages)
        self.picker.setCurrentText(default)
        self.picker.setFixedHeight(26)
        self.picker.setStyleSheet(
            f"QComboBox{{background:#ffffff;border:1px solid {BORDER};"
            f"border-radius:6px;padding:2px 8px;font-size:11px;color:{TEXT};}}"
            f"QComboBox:hover{{border-color:{ACCENT};}}"
            "QComboBox::drop-down{border:none;width:18px;}"
            f"QComboBox::down-arrow{{image:none;border-left:4px solid transparent;"
            f"border-right:4px solid transparent;border-top:5px solid {MUTED};"
            "margin-right:6px;}"
            # the popup is a separate widget - without this it inherits the
            # system palette and can end up white text on a white list
            f"QComboBox QAbstractItemView{{background:#ffffff;color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:6px;padding:3px;"
            f"selection-background-color:{ACCENT};selection-color:#ffffff;"
            "outline:none;}")
        self.picker.view().setStyleSheet(
            f"QAbstractItemView{{background:#ffffff;color:{TEXT};"
            f"selection-background-color:{ACCENT};selection-color:#ffffff;}}")
        row.addWidget(self.picker)
        self.bar = bar

        if button:
            self.action = QPushButton(button)
            self.action.setCursor(Qt.PointingHandCursor)
            self.action.setFixedHeight(26)
            self.action.setStyleSheet(
                f"QPushButton{{background:{ACCENT};color:#fff;border:none;"
                "border-radius:13px;padding:0 14px;font-size:11px;font-weight:600;}"
                "QPushButton:hover{background:#5b3fd4;}"
                "QPushButton:disabled{background:#c9c4e0;}")
            row.addSpacing(8)
            row.addWidget(self.action)

        root.addWidget(bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        holder = QWidget()
        self.column = QVBoxLayout(holder)
        self.column.setContentsMargins(16, 14, 16, 14)
        self.column.setSpacing(10)
        self.column.addStretch()
        self.scroll.setWidget(holder)
        root.addWidget(self.scroll)

    def header_only(self):
        """The title + language picker bar, detached from the scroll area."""
        self.scroll.setVisible(False)
        return self.bar

    def add(self, card):
        self.column.insertWidget(self.column.count() - 1, card)
        self.cards.append(card)
        QTimer.singleShot(30, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))
        return card

    def lang_code(self):
        return LANGUAGES[self.picker.currentText()]

    def lang_name(self):
        return self.picker.currentText()


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.listener = None
        self.workers = []
        self.next_id = 0
        self.pairs = {}          # card id -> translation card
        self.interim_card = None
        self.seconds = 0
        self.local = TRANSCRIBE_MODE == "local"   # default: deepgram api

        # pending sentence being assembled
        self.cards = []            # LineCard per utterance
        self.queue = []            # cards waiting to go out in a batch
        self.requests = 0          # api calls made this session
        self.session_file = None   # where this session autosaves to
        self.generation = 0        # bumped when a re-translate invalidates work
        self.session_started = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.speakers_seen = set()
        self.mic_on = CAPTURE_MIC
        # translate through the cloud API when we have a key; the local 9B
        # does not fit next to whisper on an 8GB card

        self.setWindowTitle("LinguistLive")
        self.resize(1240, 800)
        self.setStyleSheet(f"background:{BG};")

        shell = QHBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        self.sidebar = self.build_sidebar()
        shell.addWidget(self.sidebar)

        # slim reopen tab, only visible while the sidebar is hidden
        self.show_sidebar_btn = QPushButton("»")
        self.show_sidebar_btn.setCursor(Qt.PointingHandCursor)
        self.show_sidebar_btn.setFixedWidth(22)
        self.show_sidebar_btn.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.show_sidebar_btn.setToolTip("Show sidebar")
        self.show_sidebar_btn.clicked.connect(self.toggle_sidebar)
        self.show_sidebar_btn.setStyleSheet(
            f"QPushButton{{border:none;border-right:1px solid {BORDER};"
            f"background:{SIDEBAR_BG};color:{MUTED};font-size:14px;}}"
            f"QPushButton:hover{{color:{ACCENT};}}")
        self.show_sidebar_btn.setVisible(False)
        shell.addWidget(self.show_sidebar_btn)

        right = QWidget()
        root = QVBoxLayout(right)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        shell.addWidget(right, 1)

        root.addWidget(self.build_topbar())

        from PySide6.QtWidgets import QStackedWidget
        self.pages = QStackedWidget()
        self.pages.addWidget(self.build_panes())
        self.pages.addWidget(self.build_history())
        root.addWidget(self.pages, 1)

        # floating control bar
        wrap = QWidget()
        wrap.setStyleSheet(f"background:{PANEL_BG};")
        tray = QHBoxLayout(wrap)
        tray.setContentsMargins(0, 14, 0, 18)
        tray.addStretch()
        tray.addWidget(self.build_controls())
        tray.addStretch()
        root.addWidget(wrap)

        self.clock = QTimer(self)
        self.clock.timeout.connect(self.tick)

        # fires shortly after lines stop arriving, so a burst goes out as one
        # request instead of one per line
        self.batch_timer = QTimer(self)
        self.batch_timer.setSingleShot(True)
        self.batch_timer.timeout.connect(self.flush_queue)

        # nothing is lost if the app closes unexpectedly
        # safety net: if a batch never reports back, un-stick its cards
        self.watchdog = QTimer(self)
        self.watchdog.timeout.connect(self.check_stalled)
        self.watchdog.start(5000)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(AUTOSAVE_SECONDS * 1000)

        self.paint_engine()
        self.paint_translator()
        self.paint_mic()
        self.refresh_labels()

        # Ctrl+M mutes from anywhere in the window
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("Ctrl+M"), self, activated=self.toggle_mic)

    # -- chrome --

    def build_sidebar(self):
        bar = QWidget()
        bar.setFixedWidth(180)
        bar.setStyleSheet(f"background:{SIDEBAR_BG};border-right:1px solid {BORDER};")
        col = QVBoxLayout(bar)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        top = QHBoxLayout()
        top.setContentsMargins(14, 0, 8, 0)
        brand = QLabel("LinguistLive")
        brand.setStyleSheet(f"color:{TEXT};font-size:15px;font-weight:700;")
        top.addWidget(brand)
        top.addStretch()
        shut = QPushButton("«")
        shut.setCursor(Qt.PointingHandCursor)
        shut.setFixedSize(22, 22)
        shut.setToolTip("Hide sidebar")
        shut.clicked.connect(self.toggle_sidebar)
        shut.setStyleSheet(
            f"QPushButton{{border:none;background:transparent;color:{MUTED};"
            "font-size:15px;}"
            f"QPushButton:hover{{color:{ACCENT};}}")
        top.addWidget(shut)
        holder = QWidget()
        holder.setFixedHeight(56)
        holder.setLayout(top)
        col.addWidget(holder)
        col.addSpacing(6)

        self.nav = {}
        for icon, name, slot in [("🗣", "Live Transcript", self.show_live),
                                 ("🕘", "History", self.show_history),
                                 ("🎙", "Audio", self.open_audio_settings),
                                 ("🔑", "API Keys", self.open_key_manager)]:
            item = QPushButton(f"  {icon}   {name}")
            item.setCursor(Qt.PointingHandCursor)
            item.setFixedHeight(38)
            item.clicked.connect(slot)
            self.nav[name] = item
            col.addWidget(item)
        self.paint_nav("Live Transcript")

        col.addStretch()

        self.session_label = QLabel("  Session Idle\n  Source: English")
        self.session_label.setStyleSheet(
            f"color:{MUTED};font-size:10px;padding:8px 0 8px 14px;")
        col.addWidget(self.session_label)

        self.stop_btn = QPushButton("⏹  Stop Session")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setFixedHeight(32)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.toggle)
        self.stop_btn.setStyleSheet(
            f"QPushButton{{background:#fff;color:{TEXT};border:1px solid {BORDER};"
            "border-radius:7px;font-size:11px;font-weight:600;margin:0 12px 14px 12px;}"
            "QPushButton:hover{border-color:#c9c4e0;}"
            "QPushButton:disabled{color:#b8b8c8;}")
        col.addWidget(self.stop_btn)
        return bar

    def paint_nav(self, active):
        for name, btn in self.nav.items():
            on = name == active
            btn.setStyleSheet(
                f"QPushButton{{text-align:left;border:none;"
                f"color:{ACCENT if on else '#5a5a70'};font-size:12px;"
                f"font-weight:{'600' if on else '500'};padding-left:14px;"
                f"background:{SIDEBAR_ACTIVE if on else 'transparent'};}}"
                f"QPushButton:hover{{background:{SIDEBAR_ACTIVE};}}")

    def toggle_sidebar(self):
        showing = self.sidebar.isVisible()
        self.sidebar.setVisible(not showing)
        self.show_sidebar_btn.setVisible(showing)

    def show_live(self):
        self.paint_nav("Live Transcript")
        self.pages.setCurrentIndex(0)

    def show_history(self):
        self.paint_nav("History")
        self.pages.setCurrentIndex(1)
        self.load_history()

    def build_topbar(self):
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet(f"background:{BG};border-bottom:1px solid {BORDER};")
        row = QHBoxLayout(bar)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(12)

        self.live = QLabel("● IDLE")
        self.live.setStyleSheet(
            f"color:{MUTED};background:#f2f2f6;border-radius:11px;"
            "padding:4px 11px;font-size:11px;font-weight:700;")
        row.addWidget(self.live)

        self.timer_label = QLabel("00:00:00")
        self.timer_label.setStyleSheet(
            f"color:{TEXT};background:#f2f2f6;border-radius:11px;"
            "padding:4px 11px;font-size:11px;font-weight:600;")
        row.addWidget(self.timer_label)
        row.addSpacing(6)

        self.stt_label = QLabel(transcribe_summary())
        self.stt_label.setStyleSheet(
            f"color:{TEXT};background:#eef7f1;border:1px solid #cfe8da;"
            "border-radius:11px;padding:3px 10px;font-size:11px;font-weight:600;")
        row.addWidget(self.stt_label)
        row.addSpacing(6)

        self.engine_picker = QComboBox()
        self.engine_picker.addItems(["API (Deepgram)", "Local (Whisper)"])
        self.engine_picker.setFixedHeight(26)
        self.engine_picker.setStyleSheet(combo_style())
        self.engine_picker.view().setStyleSheet(
            f"QAbstractItemView{{background:#fff;color:{TEXT};"
            f"selection-background-color:{ACCENT};selection-color:#fff;}}")
        self.engine_picker.currentTextChanged.connect(self.engine_chosen)
        row.addWidget(self.engine_picker)

        self.mic_btn = QPushButton()
        self.mic_btn.setCursor(Qt.PointingHandCursor)
        self.mic_btn.setFixedHeight(26)
        self.mic_btn.clicked.connect(self.toggle_mic)
        row.addWidget(self.mic_btn)

        self.models_btn = QPushButton("⚙ Models")
        self.models_btn.setCursor(Qt.PointingHandCursor)
        self.models_btn.setFixedHeight(26)
        self.models_btn.setToolTip("Manage local Whisper models")
        self.models_btn.clicked.connect(self.open_model_manager)
        self.models_btn.setStyleSheet(
            f"QPushButton{{background:#fff;border:1px solid {BORDER};"
            f"border-radius:6px;padding:0 10px;font-size:11px;color:{TEXT};}}"
            f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};}}")
        row.addWidget(self.models_btn)
        row.addStretch()

        self.note = QLabel("")
        self.note.setStyleSheet(f"color:{MUTED};font-size:11px;")
        row.addWidget(self.note)
        row.addSpacing(8)

        self.mt_label = QLabel(translate_summary())
        self.mt_label.setStyleSheet(
            f"color:{ACCENT};background:{ACCENT_SOFT};border:1px solid #ded6fb;"
            "border-radius:11px;padding:3px 10px;font-size:11px;font-weight:600;")
        row.addWidget(self.mt_label)
        return bar

    def refresh_labels(self):
        self.stt_label.setText(transcribe_summary(self.local))
        self.stt_label.setStyleSheet(
            (f"color:{TEXT};background:#eef7f1;border:1px solid #cfe8da;"
             if self.local else
             f"color:{ACCENT};background:{ACCENT_SOFT};border:1px solid #ded6fb;")
            + "border-radius:11px;padding:3px 10px;font-size:11px;font-weight:600;")
        self.mt_label.setText(translate_summary())

    def build_panes(self):
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self.source = Pane("TRANSCRIPT", list(LANGUAGES), "English (US)")
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.source)
        left.setStyleSheet(f"background:{BG};border-right:1px solid {BORDER};")
        row.addWidget(left, 3)

        # right rail: target language + the actions that apply to everything
        self.target = Pane("TRANSLATION", list(LANGUAGES),
                           ENV.get("TARGET_LANGUAGE", "Turkish"))
        right = QWidget()
        right.setStyleSheet(f"background:{PANEL_BG};")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)
        rv.addWidget(self.target.header_only())

        pad = QWidget()
        pv = QVBoxLayout(pad)
        pv.setContentsMargins(16, 4, 16, 16)
        pv.setSpacing(8)

        self.translate_btn = QPushButton("Translate new lines")
        self.retranslate_btn = QPushButton("↻  Re-translate all")
        self.reply_btn = QPushButton("💬  What should I say?")
        for b, slot, primary in (
            (self.translate_btn, self.translate_now, True),
            (self.retranslate_btn, self.retranslate_all, False),
            (self.reply_btn, self.suggest_reply, False),
        ):
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(34)
            b.setStyleSheet(
                (f"QPushButton{{background:{ACCENT};color:#fff;border:none;"
                 if primary else
                 f"QPushButton{{background:#fff;color:{TEXT};"
                 f"border:1px solid {BORDER};")
                + "border-radius:8px;font-size:12px;font-weight:600;}"
                + (f"QPushButton:hover{{background:#5b3fd4;}}" if primary else
                   f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};}}")
                + "QPushButton:disabled{background:#eceaf5;color:#b0aac6;"
                  "border-color:#e5e2f0;}")
            b.clicked.connect(slot)
            pv.addWidget(b)
        self.translate_btn.setEnabled(False)

        lvl = QHBoxLayout()
        lvl.setContentsMargins(0, 2, 0, 0)
        lab = QLabel("Reply level")
        lab.setStyleSheet(f"color:{MUTED};font-size:11px;")
        lvl.addWidget(lab)
        lvl.addStretch()
        self.level_picker = QComboBox()
        self.level_picker.addItems(list(CEFR_LEVELS))
        self.level_picker.setCurrentText(
            DEFAULT_CEFR if DEFAULT_CEFR in CEFR_LEVELS else "B2 · natural")
        self.level_picker.setFixedHeight(26)
        self.level_picker.setStyleSheet(
            f"QComboBox{{background:#fff;border:1px solid {BORDER};"
            f"border-radius:6px;padding:2px 8px;font-size:11px;color:{TEXT};}}"
            f"QComboBox:hover{{border-color:{ACCENT};}}"
            "QComboBox::drop-down{border:none;width:18px;}"
            f"QComboBox::down-arrow{{image:none;border-left:4px solid transparent;"
            f"border-right:4px solid transparent;border-top:5px solid {MUTED};"
            "margin-right:6px;}"
            f"QComboBox QAbstractItemView{{background:#fff;color:{TEXT};"
            f"border:1px solid {BORDER};selection-background-color:{ACCENT};"
            "selection-color:#fff;outline:none;}")
        self.level_picker.view().setStyleSheet(
            f"QAbstractItemView{{background:#fff;color:{TEXT};"
            f"selection-background-color:{ACCENT};selection-color:#fff;}}")
        lvl.addWidget(self.level_picker)
        pv.addLayout(lvl)

        hint = QLabel("Hover any line for ↻ retry, ✎ edit, ⧉ copy")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED};font-size:11px;padding-top:4px;")
        pv.addWidget(hint)

        self.reply_box = QFrame()
        self.reply_box.setObjectName("card")
        self.reply_box.setStyleSheet(
            f"#card{{background:#fff;border:1px solid {BORDER};"
            f"border-left:3px solid {ACCENT};border-radius:10px;}}")
        rb = QVBoxLayout(self.reply_box)
        rb.setContentsMargins(12, 10, 12, 11)
        rb.setSpacing(6)

        cap = QHBoxLayout()
        title = QLabel("WHAT TO SAY NEXT")
        title.setStyleSheet(
            f"color:{MUTED};font-size:10px;font-weight:700;letter-spacing:1px;")
        cap.addWidget(title)
        cap.addStretch()
        copy = QPushButton("⧉")
        copy.setCursor(Qt.PointingHandCursor)
        copy.setFixedSize(20, 20)
        copy.setToolTip("Copy")
        copy.setStyleSheet(
            f"QPushButton{{border:none;background:transparent;color:{MUTED};}}"
            f"QPushButton:hover{{color:{ACCENT};}}")
        copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.reply_text.text()))
        cap.addWidget(copy)
        rb.addLayout(cap)

        self.reply_text = QLabel("")
        self.reply_text.setWordWrap(True)
        self.reply_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.reply_text.setStyleSheet(
            f"color:{TEXT};font-size:12px;line-height:150%;")
        self.reply_text.setMinimumHeight(120)
        self.reply_text.setAlignment(Qt.AlignTop)
        rb.addWidget(self.reply_text)

        self.reply_box.setVisible(False)
        pv.addWidget(self.reply_box)
        pv.addStretch()
        rv.addWidget(pad, 1)

        row.addWidget(right, 1)
        return holder

    def session_payload(self):
        return {
            "started": self.session_started,
            "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engine": transcribe_summary(self.local),
            "source": self.source.lang_name(),
            "target": self.target.lang_name(),
            "speakers": {str(k): v for k, v in SPEAKER_NAMES.items()},
            "lines": [{"stamp": c.when.text(), "text": c.text,
                       "translation": c.translation,
                       "speaker": c.speaker} for c in self.cards],
        }

    def autosave(self):
        if not self.cards:
            return
        os.makedirs(SESSION_DIR, exist_ok=True)
        if self.session_file is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.session_file = os.path.join(SESSION_DIR, f"session-{stamp}.json")
        try:
            with open(self.session_file, "w") as f:
                json.dump(self.session_payload(), f, ensure_ascii=False, indent=1)
            secure_file(self.session_file)
        except Exception as e:
            log(f"autosave failed: {e}")

    def export_file(self, path):
        """Export a saved session without having to open it first."""
        try:
            data = json.load(open(path))
        except Exception as e:
            self.status_msg(f"could not read session: {e}")
            return
        lines = data.get("lines", [])
        if not lines:
            self.status_msg("that session is empty")
            return

        from PySide6.QtWidgets import QFileDialog
        base = os.path.splitext(os.path.basename(path))[0]
        target, _ = QFileDialog.getSaveFileName(
            self, "Export transcript", os.path.join(SESSION_DIR, base + ".txt"),
            "Text (*.txt);;Markdown (*.md);;Subtitles (*.srt)")
        if not target:
            return
        write_transcript(target, data)
        secure_file(target)
        log(f"exported {len(lines)} lines -> {target}")
        self.status_msg(f"exported to {os.path.basename(target)}")

    def status_msg(self, msg):
        if hasattr(self, "history_status"):
            self.history_status.setText(msg)
        self.note.setText(msg)

    def export_session(self):
        """Export what is currently on screen."""
        if not self.cards:
            self.note.setText("nothing to export")
            return
        from PySide6.QtWidgets import QFileDialog

        os.makedirs(SESSION_DIR, exist_ok=True)
        default = os.path.join(
            SESSION_DIR, datetime.now().strftime("transcript-%Y%m%d-%H%M.txt"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export transcript", default,
            "Text (*.txt);;Markdown (*.md);;Subtitles (*.srt)")
        if not path:
            return
        write_transcript(path, self.session_payload())
        secure_file(path)
        log(f"exported {len(self.cards)} lines -> {path}")
        self.note.setText(f"exported to {os.path.basename(path)}")

    def build_history(self):
        page = QWidget()
        page.setStyleSheet(f"background:{BG};")
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        bar = QWidget()
        bar.setFixedHeight(46)
        bar.setStyleSheet(f"border-bottom:1px solid {BORDER};")
        h = QHBoxLayout(bar)
        h.setContentsMargins(20, 0, 16, 0)
        t = QLabel("SAVED SESSIONS")
        t.setStyleSheet(
            f"color:{MUTED};font-size:11px;font-weight:700;letter-spacing:1px;")
        h.addWidget(t)
        h.addStretch()
        folder = QPushButton("Open folder")
        folder.setCursor(Qt.PointingHandCursor)
        folder.setFixedHeight(26)
        folder.setStyleSheet(
            f"QPushButton{{background:#fff;border:1px solid {BORDER};"
            f"border-radius:6px;padding:0 10px;font-size:11px;color:{TEXT};}}"
            f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};}}")
        folder.clicked.connect(
            lambda: subprocess.Popen(["xdg-open", SESSION_DIR]))
        h.addWidget(folder)
        v.addWidget(bar)

        self.history_status = QLabel("")
        self.history_status.setStyleSheet(f"color:{MUTED};font-size:11px;")

        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        self.history_list = QVBoxLayout(inner)
        self.history_list.setContentsMargins(16, 14, 16, 14)
        self.history_list.setSpacing(8)
        self.history_list.addStretch()
        self.history_scroll.setWidget(inner)
        v.addWidget(self.history_scroll)
        wrap = QWidget()
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(20, 0, 20, 10)
        wl.addWidget(self.history_status)
        v.addWidget(wrap)
        return page

    def load_history(self):
        while self.history_list.count() > 1:
            w = self.history_list.takeAt(0).widget()
            if w:
                w.setParent(None)

        os.makedirs(SESSION_DIR, exist_ok=True)
        files = sorted((f for f in os.listdir(SESSION_DIR) if f.endswith(".json")),
                       reverse=True)
        if not files:
            empty = QLabel("No saved sessions yet.\nThey appear here "
                           "automatically while you record.")
            empty.setStyleSheet(f"color:{MUTED};font-size:12px;")
            self.history_list.insertWidget(0, empty)
            return

        for name in files:
            path = os.path.join(SESSION_DIR, name)
            try:
                with open(path) as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
            except Exception:
                continue
            self.history_list.insertWidget(
                self.history_list.count() - 1, self.history_row(path, data))

    def history_row(self, path, data):
        row = QFrame()
        row.setObjectName("card")
        row.setStyleSheet(
            f"#card{{background:#fff;border:1px solid {BORDER};"
            "border-radius:9px;}")
        h = QHBoxLayout(row)
        h.setContentsMargins(14, 10, 12, 11)

        lines = data.get("lines", [])
        done = sum(1 for l in lines if l.get("translation"))
        left = QVBoxLayout()
        title = QLabel(f"{data.get('started', '?')}   ·   "
                       f"{os.path.basename(path).replace('session-', '')[:-5]}")
        title.setStyleSheet(f"color:{TEXT};font-size:12px;font-weight:600;")
        left.addWidget(title)
        sub = QLabel(f"{len(lines)} lines · {done} translated · "
                     f"{data.get('target', '')}")
        sub.setStyleSheet(f"color:{MUTED};font-size:11px;")
        left.addWidget(sub)
        h.addLayout(left)
        h.addStretch()

        for label, slot in (("Open", lambda: self.open_session(path)),
                            ("Export", lambda: self.export_file(path)),
                            ("Delete", lambda: self.delete_session(path))):
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(26)
            b.setStyleSheet(
                f"QPushButton{{background:#fff;border:1px solid {BORDER};"
                f"border-radius:6px;padding:0 12px;font-size:11px;color:{TEXT};}}"
                f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};}}")
            b.clicked.connect(slot)
            h.addWidget(b)
        return row

    def open_session(self, path):
        # session files are local, but a truncated or hand-edited one should
        # not take the app down
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("not a session object")
        except Exception as e:
            self.status_msg(f"could not open session: {e}")
            return
        for c in self.cards:
            c.setParent(None)
        self.cards.clear()
        self.source.cards.clear()
        SPEAKER_NAMES.clear()
        for k, v in (data.get("speakers") or {}).items():
            try:
                SPEAKER_NAMES[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        for i, item in enumerate(data.get("lines") or []):
            if not isinstance(item, dict):
                continue
            spk = item.get("speaker")
            card = LineCard(i, str(item.get("text", "")),
                            str(item.get("stamp", "")),
                            speaker=spk if isinstance(spk, int) else None)
            card.retry.connect(self.translate_card)
            card.edit.connect(self.edit_card)
            if item.get("translation"):
                card.set_state("done", item["translation"])
            self.source.add(card)
            self.cards.append(card)
        if data.get("target") in LANGUAGES:
            self.target.picker.setCurrentText(data["target"])
        # viewing an old session: further edits save back to that same file
        self.session_file = path
        self.session_started = data.get("started", self.session_started)
        self.show_live()
        self.update_progress()
        log(f"opened session {os.path.basename(path)}")

    def delete_session(self, path):
        if QMessageBox.question(
                self, "Delete session",
                f"Delete this saved session?\n\n{os.path.basename(path)}"
        ) != QMessageBox.Yes:
            return
        try:
            os.remove(path)
        except OSError as e:
            self.status_msg(f"could not delete: {e}")
            return
        log(f"deleted {os.path.basename(path)}")
        self.status_msg("session deleted")
        self.load_history()

    def build_controls(self):
        box = QFrame()
        box.setStyleSheet(
            f"background:#fff;border:1px solid {BORDER};border-radius:26px;")
        row = QHBoxLayout(box)
        row.setContentsMargins(16, 9, 9, 9)
        row.setSpacing(10)

        self.from_lang = QLabel("English")
        self.from_lang.setStyleSheet(f"color:{TEXT};font-size:12px;border:none;")
        row.addWidget(self.from_lang)

        arrow = QLabel("→")
        arrow.setStyleSheet(f"color:{MUTED};font-size:13px;border:none;")
        row.addWidget(arrow)

        self.to_lang = QLabel("Turkish")
        self.to_lang.setStyleSheet(f"color:{TEXT};font-size:12px;border:none;")
        row.addWidget(self.to_lang)

        self.button = QPushButton("  Start Listening")
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setFixedHeight(36)
        self.button.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#fff;border:none;"
            "border-radius:18px;padding:0 20px;font-size:12px;font-weight:600;}"
            "QPushButton:hover{background:#5b3fd4;}"
            "QPushButton:disabled{background:#c9c4e0;}")
        self.button.clicked.connect(self.toggle)
        row.addWidget(self.button)

        self.source.picker.currentTextChanged.connect(
            lambda t: self.from_lang.setText(t.split(" (")[0]))
        self.target.picker.currentTextChanged.connect(self.to_lang.setText)

        self.note.setText("")
        if DEEPGRAM_ENABLED and not DEEPGRAM_KEY:
            self.button.setEnabled(False)
            self.note.setText("DEEPGRAM_API_KEY missing from .env")
        elif not HAVE_TRANSLATION:
            self.note.setText("no translation key in .env")
        elif not HAVE_TRANSLATION:
            self.note.setText("No translation key in .env - transcript only")
        else:
            self.note.setText(
                f"Translating with {GEMINI_MODEL if USE_GEMINI else TRANSLATE_MODEL}")
        return box

    # -- run control --

    def paint_engine(self):
        self.engine_picker.blockSignals(True)
        self.engine_picker.setCurrentText(
            "Local (Whisper)" if self.local else "API (Deepgram)")
        self.engine_picker.blockSignals(False)
        self.models_btn.setVisible(self.local)

    def engine_chosen(self, label):
        """Top-bar dropdown picks which engine transcribes."""
        if self.listener:                       # never mid-session
            self.paint_engine()
            self.note.setText("stop the session before switching engine")
            return

        want_local = label.startswith("Local")
        if want_local and not installed_models():
            self.note.setText("no local model installed - opening manager")
            self.paint_engine()
            self.open_model_manager()
            return
        if not want_local and not DEEPGRAM_KEY:
            self.note.setText("No DEEPGRAM_API_KEY in .env")
            self.paint_engine()
            return

        self.local = want_local
        self.paint_engine()
        self.refresh_labels()
        self.button.setEnabled(
            bool(installed_models()) if self.local else bool(DEEPGRAM_KEY))
        self.note.setText("" if self.local else "speaker labels on")
        log(f"engine -> {'local whisper' if self.local else 'deepgram'}")

    def paint_mic(self):
        """Green when your voice is going in, grey when muted."""
        on = self.mic_on
        self.mic_btn.setText(("🎙  Mic on" if on else "🔇  Mic off"))
        self.mic_btn.setToolTip(
            "Your microphone is being recorded (Ctrl+M to mute)" if on else
            "You are muted - your voice is not in the transcript (Ctrl+M)")
        fg, bg, edge = (("#0a8f4e", "#e6f7ef", "#cfe8da") if on
                        else (MUTED, "#f4f3f7", BORDER))
        self.mic_btn.setStyleSheet(
            f"QPushButton{{color:{fg};background:{bg};border:1px solid {edge};"
            "border-radius:6px;padding:0 10px;font-size:11px;font-weight:600;}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")

    def toggle_mic(self):
        """Mute / unmute instantly, mid-session, with no restart."""
        self.mic_on = not self.mic_on

        cap = getattr(self.listener, "capture", None) if self.listener else None
        if cap is not None:
            actual = cap.set_mic(self.mic_on)
            if self.mic_on and not actual:
                self.mic_on = False
                self.note.setText(
                    f"microphone unavailable: {cap.mic_error or 'no device'}")
        # remember the choice for the next session too
        global CAPTURE_MIC
        CAPTURE_MIC = self.mic_on
        save_env({"CAPTURE_MIC": "1" if self.mic_on else "0"})
        self.paint_mic()
        if cap is not None:
            self.note.setText("microphone on - you are being transcribed"
                              if self.mic_on else "microphone muted")
        log(f"mic {'on' if self.mic_on else 'muted'}")

    def open_audio_settings(self):
        self.paint_nav("Audio")
        AudioSettings(self).exec()
        self.paint_nav("Live Transcript" if self.pages.currentIndex() == 0
                       else "History")

    def open_key_manager(self):
        self.paint_nav("API Keys")
        KeyManager(self).exec()
        self.paint_nav("Live Transcript" if self.pages.currentIndex() == 0
                       else "History")
        self.refresh_labels()

    def open_model_manager(self):
        dlg = ModelManager(self)
        dlg.exec()
        self.paint_engine()
        self.refresh_labels()
        if self.local:
            self.button.setEnabled(bool(installed_models()))

    def toggle(self):
        if self.listener:
            self.stop()
        else:
            self.start()

    def start(self):
        self.seconds = 0
        # a new recording is a new session file, otherwise it would overwrite
        # whatever was last saved (or a session opened from History)
        if self.session_file and self.cards:
            self.autosave()                     # flush the previous one first
        self.session_file = None
        self.session_started = datetime.now().strftime("%Y-%m-%d %H:%M")
        # a fresh run keeps earlier text so you can compare engines in one
        # session; the divider shows where the switch happened
        if self.cards:
            div = LineCard(len(self.cards),
                           f"— switched to {transcribe_summary(self.local)} —", "")
            div.set_state("done", "")
            div.tools.setVisible(False)
            div.body.setStyleSheet(
                f"color:{MUTED};font-size:11px;font-style:italic;")
            div.out.setVisible(False)
            self.source.add(div)
        self.translate_btn.setEnabled(any(
            c.state in ("new", "failed") for c in self.cards))
        self.source.picker.setEnabled(False)
        self.button.setText("  Stop")
        self.button.setStyleSheet(
            self.button.styleSheet().replace(ACCENT, "#e84e6c"))
        self.live.setText("● LIVE")
        self.live.setStyleSheet(
            "color:#0a8f4e;background:#e6f7ef;border-radius:11px;"
            "padding:4px 11px;font-size:11px;font-weight:700;")
        cls = LocalListener if self.local else Listener
        self.listener = cls(self.source.lang_code())
        self.listener.interim.connect(self.on_interim)
        self.listener.final.connect(self.on_final)
        self.listener.status.connect(self.note.setText)
        self.listener.start()
        self.clock.start(1000)
        self.stop_btn.setEnabled(True)
        self.session_label.setText(
            f"  Session Active\n  Source: {self.source.lang_name().split(' (')[0]}")

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener.wait(2000)
            self.listener = None
        self.clock.stop()
        self.clear_interim()
        self.source.picker.setEnabled(True)
        self.button.setText("  Start Listening")
        self.button.setStyleSheet(
            self.button.styleSheet().replace("#e84e6c", ACCENT))
        self.live.setText("● IDLE")
        self.live.setStyleSheet(
            f"color:{MUTED};background:#f2f2f6;border-radius:11px;"
            "padding:4px 11px;font-size:11px;font-weight:700;")
        self.stop_btn.setEnabled(False)
        self.session_label.setText("  Session Idle\n  Source: English")
        self.autosave()
        if self.cards:
            log(f"session saved: {os.path.basename(self.session_file or '')}")
            self.note.setText(
                f"saved · {len(self.cards)} lines · see History")

    def tick(self):
        self.seconds += 1
        h, rem = divmod(self.seconds, 3600)
        m, s = divmod(rem, 60)
        self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    # -- transcript flow --

    def clear_interim(self):
        if self.interim_card is not None:
            self.interim_card.setParent(None)
            if self.interim_card in self.source.cards:
                self.source.cards.remove(self.interim_card)
            self.interim_card = None

    def on_interim(self, speaker, text):
        if self.interim_card is None:
            self.interim_card = Card(text=text + " …")
            self.interim_card.body.setStyleSheet(
                f"color:{MUTED};font-size:13px;font-style:italic;")
            self.source.add(self.interim_card)
        else:
            self.interim_card.set_text(text + " …")

    def on_final(self, speaker, text, stamp, ended):
        """Every utterance becomes its own card."""
        self.clear_interim()
        text = text.strip()
        if not text:
            return

        # only deepgram tells us who spoke; local whisper has no diarization
        who = speaker if not self.local else None
        if who is not None:
            self.speakers_seen.add(who)
        card = LineCard(len(self.cards), text, stamp, speaker=who)
        card.retry.connect(self.translate_card)
        card.edit.connect(self.edit_card)
        self.source.add(card)
        self.cards.append(card)

        self.translate_btn.setEnabled(True)
        n = len(self.speakers_seen)
        self.note.setText(f"{len(self.cards)} lines"
                          + (f" · {n} speakers" if n > 1 else ""))

    # -- click-to-translate --

    def paint_translator(self):
        """The active translation model is shown by the top-bar pill."""
        self.refresh_labels()

    def translate_now(self):
        """Translate every line that has not been done yet."""
        pending = [c for c in self.cards if c.state in ("new", "failed")]
        if not pending:
            self.note.setText("nothing new to translate")
            return
        log(f"click Translate: {len(pending)} pending lines")
        for card in pending:
            if card.state != "working" and card not in self.queue:
                card.set_state("working")
                self.queue.append(card)
        self.flush_queue()

    def translate_card(self, card):
        """Queue one line. Lines are sent together to save API requests."""
        if card in self.queue:
            return
        if card.state == "working":
            # stuck from an earlier attempt - abandon it and try again
            for jid, (c, _g) in list(self.pairs.items()):
                if c is card:
                    del self.pairs[jid]
        card.set_state("working")
        self.queue.append(card)

        if len(self.queue) >= BATCH_SIZE:
            self.flush_queue()
        else:
            self.batch_timer.start(BATCH_WAIT_MS)

    def check_stalled(self):
        """Mark cards failed if their request never came back."""
        if not self.pairs:
            return
        alive = any(w.isRunning() for w in self.workers)
        if alive:
            return
        stuck = [c for c, _g in self.pairs.values()]
        self.pairs.clear()
        for card in stuck:
            if card.state == "working":
                card.set_state("failed", "no response - click ↻ to retry")
        if stuck:
            log(f"watchdog: released {len(stuck)} stalled card(s)")
            self.update_progress()

    def keep(self, worker):
        """Hold a reference until the thread is truly done.

        Dropping it inside finished() frees the QThread while it is still
        emitting, so done/failed never arrive and cards hang on "working".
        """
        self.workers.append(worker)
        worker.finished.connect(lambda w=worker: QTimer.singleShot(
            0, lambda: self.workers.remove(w) if w in self.workers else None))

    def flush_queue(self):
        """Send everything queued as ONE api request."""
        self.batch_timer.stop()
        if not self.queue:
            return

        batch, self.queue = self.queue[:BATCH_SIZE], self.queue[BATCH_SIZE:]
        gen = self.generation
        ids, texts = [], []
        for card in batch:
            jid = self.next_id
            self.next_id += 1
            self.pairs[jid] = (card, gen)
            ids.append(jid)
            texts.append(card.text)

        self.requests += 1
        log(f"api request #{self.requests}: {len(texts)} lines in one call")
        self.note.setText(f"translating {len(texts)} lines…")

        worker = Translator(ids, texts, self.target.lang_name())
        worker.done.connect(self.on_translated)
        worker.failed.connect(self.on_translate_failed)
        if hasattr(worker, "status"):
            worker.status.connect(self.note.setText)
        self.keep(worker)
        worker.start()

        # anything left over goes out after a short gap, never all at once
        if self.queue:
            QTimer.singleShot(MIN_REQUEST_GAP_MS, self.flush_queue)

    def suggest_reply(self):
        """Draft what to say next, based on the tail of the conversation."""
        spoken = [c.text for c in self.cards if not c.text.startswith("—")]
        if not spoken:
            self.note.setText("nothing said yet")
            return
        recent = spoken[-REPLY_CONTEXT_LINES:]

        self.reply_box.setVisible(True)
        self.reply_text.setText("Thinking…")
        self.reply_btn.setEnabled(False)
        self.note.setText(f"drafting a reply from the last {len(recent)} lines…")

        level = CEFR_LEVELS[self.level_picker.currentText()]
        log(f"reply level: {self.level_picker.currentText()}")
        worker = Replier(recent, self.source.lang_name(), level)
        worker.done.connect(self.on_reply)
        worker.failed.connect(self.on_reply_failed)
        worker.finished.connect(lambda: self.reply_btn.setEnabled(True))
        self.keep(worker)
        worker.start()

    def on_reply(self, text):
        text = text.strip().strip('"')
        self.reply_text.setText(text)
        self.reply_text.setStyleSheet(
            f"color:{TEXT};font-size:12px;line-height:150%;")
        words = len(text.split())
        self.note.setText(f"~{words} words · about {max(1, round(words / 2.5))}s")

    def on_reply_failed(self, err):
        self.reply_text.setText(f"⚠ {err}")
        self.reply_text.setStyleSheet("color:#c2413f;font-size:12px;")

    def edit_card(self, card):
        """Fix a mis-heard line, then offer to translate the corrected text."""
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getMultiLineText(
            self, "Edit line", "Correct the transcript:", card.text)
        if not ok or not text.strip() or text.strip() == card.text:
            return
        card.set_text(text.strip())
        log(f"edited line {card.index}")
        self.translate_card(card)

    def retranslate_all(self):
        """Redo every line - use after changing the target language."""
        if not self.cards:
            return
        log(f"re-translating {len(self.cards)} lines -> {self.target.lang_name()}")
        self.queue.clear()
        self.generation += 1        # anything already in flight is now stale
        for card in self.cards:
            card.set_state("working")
            self.queue.append(card)
        self.flush_queue()

    def on_translated(self, ids, texts):
        for jid, text in zip(ids, texts):
            item = self.pairs.pop(jid, None)
            if not item:
                continue
            card, gen = item
            if gen != self.generation:      # superseded by a re-translate
                continue
            if text:
                card.set_state("done", text)
            else:
                # the model merged or skipped this one - retryable on its own
                card.set_state("failed", "line skipped - click ↻ to retry")
        self.refresh_labels()
        self.update_progress()

    def on_translate_failed(self, ids, err):
        for jid in ids:
            item = self.pairs.pop(jid, None)
            if not item:
                continue
            card, gen = item
            if gen == self.generation:
                card.set_state("failed", err[:90])
        log(f"batch failed: {err[:160]}")
        self.update_progress()

    def update_progress(self):
        done = sum(1 for c in self.cards if c.state == "done")
        bad = sum(1 for c in self.cards if c.state == "failed")
        left = len(self.cards) - done - bad
        bits = [f"{done}/{len(self.cards)} translated"]
        if bad:
            bits.append(f"{bad} failed")
        if self.requests:
            bits.append(f"{self.requests} api calls")
        self.note.setText("  ·  ".join(bits) if self.cards else "")
        self.translate_btn.setEnabled(left > 0 or bad > 0)

    def closeEvent(self, e):
        self.stop()
        if self.cards:
            self.autosave()
        elif self.session_file and os.path.exists(self.session_file):
            # nothing was recorded - don't leave an empty file in History
            try:
                os.remove(self.session_file)
                log("removed empty session file")
            except OSError:
                pass
        e.accept()


app = QApplication(sys.argv)
app.setFont(QFont("Inter" if "Inter" in QFont().families() else "sans", 10))
win = Window()
win.show()
sys.exit(app.exec())
