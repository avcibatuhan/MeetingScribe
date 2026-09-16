"""
Cross-platform audio capture for Linux and Windows.

Two things can be recorded:

  * system output  - what the meeting app is playing (everyone else)
  * microphone     - what you are saying

Either or both. When both are on they are mixed into one stream, so the
transcript contains the whole conversation.

The rest of the app only ever sees 16 kHz mono 16-bit PCM, so it does not
need to know which platform or backend produced it.

Linux    : parec, reading the monitor of the output device (PipeWire/Pulse)
Windows  : WASAPI loopback via soundcard, or pyaudiowpatch as a fallback
"""

import platform
import queue
import shutil
import subprocess
import threading

import numpy as np

RATE = 16000
SAMPLE_BYTES = 2                    # s16le
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


# ----------------------------------------------------------------------
# what backends are actually usable here
# ----------------------------------------------------------------------

def backend_name():
    if IS_LINUX:
        return "parec" if shutil.which("parec") else None
    if IS_WINDOWS:
        for mod in ("soundcard", "pyaudiowpatch"):
            try:
                __import__(mod)
                return mod
            except ImportError:
                continue
    return None


def missing_dependency():
    """A sentence telling the user how to fix an unusable setup, or None."""
    if backend_name():
        return None
    if IS_LINUX:
        return ("PulseAudio tools not found. Install them with:\n"
                "    sudo dnf install pulseaudio-utils      (Fedora)\n"
                "    sudo apt install pulseaudio-utils      (Debian/Ubuntu)")
    if IS_WINDOWS:
        return ("No Windows audio backend. Install one with:\n"
                "    pip install soundcard")
    return f"{platform.system()} is not supported - Linux and Windows only."


# ----------------------------------------------------------------------
# device discovery
# ----------------------------------------------------------------------

def list_outputs():
    """[(id, label)] of things we can record the sound of."""
    if IS_LINUX:
        out = [("default", "Default output")]
        try:
            r = subprocess.run(["pactl", "list", "short", "sinks"],
                               capture_output=True, text=True, timeout=5)
            for line in r.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) > 1:
                    out.append((parts[1], _pretty(parts[1])))
        except Exception:
            pass
        return out

    if IS_WINDOWS:
        out = [("default", "Default output")]
        try:
            import soundcard as sc
            for spk in sc.all_speakers():
                out.append((str(spk.name), str(spk.name)))
        except Exception:
            pass
        return out
    return [("default", "Default output")]


def list_microphones():
    """[(id, label)] of real input devices, monitors excluded."""
    if IS_LINUX:
        out = [("default", "Default microphone")]
        try:
            r = subprocess.run(["pactl", "list", "short", "sources"],
                               capture_output=True, text=True, timeout=5)
            for line in r.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) > 1 and not parts[1].endswith(".monitor"):
                    out.append((parts[1], _pretty(parts[1])))
        except Exception:
            pass
        return out

    if IS_WINDOWS:
        out = [("default", "Default microphone")]
        try:
            import soundcard as sc
            for mic in sc.all_microphones(include_loopback=False):
                out.append((str(mic.name), str(mic.name)))
        except Exception:
            pass
        return out
    return [("default", "Default microphone")]


def _pretty(device_id):
    """Turn alsa_output.pci-0000_00_1f.3.analog-stereo into something readable."""
    name = device_id.split(".")[-1] if "." in device_id else device_id
    return name.replace("_", " ").replace("-", " ").strip() or device_id


# ----------------------------------------------------------------------
# one capture stream
# ----------------------------------------------------------------------

class Stream:
    """A single source of PCM. Call read() until it returns b''."""

    def __init__(self, device="default", loopback=True, frame_ms=100):
        self.device = device
        self.loopback = loopback
        self.frame_bytes = RATE * SAMPLE_BYTES * frame_ms // 1000
        self.frame_ms = frame_ms
        self.proc = None
        self._q = None
        self._thread = None
        self._stop = threading.Event()
        self._start()

    # -- linux --

    def _start_parec(self):
        if self.device == "default":
            sink = subprocess.run(["pactl", "get-default-sink"] if self.loopback
                                  else ["pactl", "get-default-source"],
                                  capture_output=True, text=True).stdout.strip()
            device = sink + ".monitor" if self.loopback else sink
        else:
            device = self.device + ".monitor" if self.loopback else self.device

        self.proc = subprocess.Popen(
            ["parec", "-d", device, "--format=s16le", f"--rate={RATE}",
             "--channels=1", f"--latency-msec={self.frame_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # -- windows --

    def _start_soundcard(self):
        """soundcard hands us float32 frames; convert to s16le in a thread."""
        import soundcard as sc

        self._q = queue.Queue(maxsize=64)

        def pump():
            try:
                if self.loopback:
                    name = (str(sc.default_speaker().name)
                            if self.device == "default" else self.device)
                    dev = sc.get_microphone(name, include_loopback=True)
                else:
                    dev = (sc.default_microphone() if self.device == "default"
                           else sc.get_microphone(self.device))
                frames = RATE * self.frame_ms // 1000
                with dev.recorder(samplerate=RATE, channels=1,
                                  blocksize=frames) as rec:
                    while not self._stop.is_set():
                        data = rec.record(numframes=frames)
                        mono = data[:, 0] if data.ndim > 1 else data
                        pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
                        try:
                            self._q.put(pcm.tobytes(), timeout=1)
                        except queue.Full:
                            pass            # consumer stalled; drop the frame
            except Exception:
                pass
            finally:
                self._q.put(b"")

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()

    def _start(self):
        backend = backend_name()
        if backend == "parec":
            self._start_parec()
        elif backend == "soundcard":
            self._start_soundcard()
        elif backend == "pyaudiowpatch":
            self._start_pyaudiowpatch()
        else:
            raise RuntimeError(missing_dependency() or "no audio backend")

    def _start_pyaudiowpatch(self):
        import pyaudiowpatch as pyaudio

        self._q = queue.Queue(maxsize=64)

        def pump():
            pa = pyaudio.PyAudio()
            stream = None
            try:
                if self.loopback:
                    info = pa.get_default_wasapi_loopback()
                else:
                    info = pa.get_default_input_device_info()
                native = int(info["defaultSampleRate"])
                channels = int(info["maxInputChannels"]) or 1
                frames = native * self.frame_ms // 1000
                stream = pa.open(format=pyaudio.paInt16, channels=channels,
                                 rate=native, input=True,
                                 input_device_index=info["index"],
                                 frames_per_buffer=frames)
                while not self._stop.is_set():
                    raw = stream.read(frames, exception_on_overflow=False)
                    a = np.frombuffer(raw, np.int16)
                    if channels > 1:
                        a = a.reshape(-1, channels).mean(axis=1).astype(np.int16)
                    if native != RATE:      # cheap linear resample to 16k
                        want = int(len(a) * RATE / native)
                        a = np.interp(np.linspace(0, len(a) - 1, want),
                                      np.arange(len(a)), a).astype(np.int16)
                    try:
                        self._q.put(a.tobytes(), timeout=1)
                    except queue.Full:
                        pass
            except Exception:
                pass
            finally:
                if stream:
                    stream.close()
                pa.terminate()
                self._q.put(b"")

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()

    # -- consuming --

    def read(self, nbytes=None):
        """Blocking read. Returns b'' when the stream ends."""
        nbytes = nbytes or self.frame_bytes
        if self.proc:
            return self.proc.stdout.read(nbytes)

        buf = b""
        while len(buf) < nbytes:
            try:
                chunk = self._q.get(timeout=2)
            except queue.Empty:
                return buf
            if chunk == b"":
                return buf
            buf += chunk
        return buf

    def close(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()
            self.proc = None


# ----------------------------------------------------------------------
# system output + microphone, mixed
# ----------------------------------------------------------------------

class Capture:
    """What the app actually uses.

    Both streams stay open for the whole session. Muting only decides whether
    a stream's audio is mixed in, so the user can mute and unmute mid-meeting
    with no restart and no gap in the recording.
    """

    def __init__(self, with_system=True, with_mic=False,
                 output_device="default", mic_device="default",
                 mic_gain=1.0, frame_ms=100, open_mic=True,
                 duck=0.25, duck_threshold=300, duck_hold_ms=600):
        self.frame_bytes = RATE * SAMPLE_BYTES * frame_ms // 1000
        self.mic_gain = mic_gain

        # ducking: drop the meeting audio while you speak so your voice is
        # not buried underneath it
        self.duck = duck
        self.duck_threshold = duck_threshold
        self.duck_frames = max(1, duck_hold_ms // max(frame_ms, 1))
        self._hold = 0

        # live switches - safe to flip from the UI thread at any time
        self.system_on = with_system
        self.mic_on = with_mic

        self.system = Stream(output_device, True, frame_ms)

        # opening the mic can fail (no device, in use); that must not stop
        # the meeting from being recorded
        self.mic = None
        self.mic_error = None
        if open_mic:
            try:
                self.mic = Stream(mic_device, False, frame_ms)
            except Exception as e:
                self.mic_error = str(e)
                self.mic_on = False

    # -- live controls --

    def set_mic(self, on):
        """Mute/unmute yourself. Takes effect on the next frame."""
        self.mic_on = bool(on) and self.mic is not None
        return self.mic_on

    def set_system(self, on):
        self.system_on = bool(on)

    def set_gain(self, gain):
        self.mic_gain = float(gain)

    @property
    def mic_available(self):
        return self.mic is not None

    # -- reading --

    def read(self, nbytes=None):
        nbytes = nbytes or self.frame_bytes

        # always drain both streams, even when muted: an unread stream backs
        # up and would replay stale audio the moment it is unmuted
        sys_data = self.system.read(nbytes)
        mic_data = self.mic.read(nbytes) if self.mic else b""

        if not sys_data and not mic_data:
            return b""

        use_sys = self.system_on and sys_data
        use_mic = self.mic_on and mic_data

        if use_sys and use_mic:
            # hold the duck briefly so it does not flap between words
            if rms(mic_data) * self.mic_gain > self.duck_threshold:
                self._hold = self.duck_frames
            elif self._hold:
                self._hold -= 1
            level = self.duck if self._hold else 1.0
            n = min(len(sys_data), len(mic_data)) & ~1
            if n == 0:
                return sys_data or mic_data
            x = np.frombuffer(sys_data[:n], np.int16).astype(np.float32) * level
            y = np.frombuffer(mic_data[:n], np.int16).astype(np.float32) * self.mic_gain
            return np.clip(x + y, -32768, 32767).astype(np.int16).tobytes()
        if use_mic:
            return scale(mic_data, self.mic_gain)
        if use_sys:
            return sys_data
        # everything muted - hand back silence so timing stays correct
        return b"\x00" * len(sys_data or mic_data)

    def close(self):
        for s in (self.system, self.mic):
            if s:
                s.close()


def scale(data, gain):
    if gain == 1.0 or not data:
        return data
    a = np.frombuffer(data, np.int16).astype(np.int32)
    return np.clip(a * gain, -32768, 32767).astype(np.int16).tobytes()


def rms(pcm):
    if not len(pcm):
        return 0.0
    a = np.frombuffer(pcm, np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a)))


def mix(a, b, gain_b=1.0):
    """Sum two PCM buffers without clipping."""
    n = min(len(a), len(b)) & ~1          # keep it sample-aligned
    if n == 0:
        return a or b
    x = np.frombuffer(a[:n], np.int16).astype(np.int32)
    y = (np.frombuffer(b[:n], np.int16).astype(np.int32) * gain_b).astype(np.int32)
    return np.clip(x + y, -32768, 32767).astype(np.int16).tobytes()


def duck_mix(system, mic, mic_gain=1.0, duck=0.25, threshold=300):
    """Mix the two, quietening the meeting while you are actually speaking.

    Without this your voice is simply added on top of whatever is playing.
    Music at rms 8000 against speech at rms 2000 leaves the speech at an
    eighth of the level, and the transcriber reads it as background noise -
    which is why it only worked when the other audio stopped.
    """
    n = min(len(system), len(mic)) & ~1
    if n == 0:
        return system or mic

    speaking = rms(mic) * mic_gain > threshold
    sys_level = duck if speaking else 1.0

    x = (np.frombuffer(system[:n], np.int16).astype(np.float32) * sys_level)
    y = (np.frombuffer(mic[:n], np.int16).astype(np.float32) * mic_gain)
    return np.clip(x + y, -32768, 32767).astype(np.int16).tobytes()


if __name__ == "__main__":
    print(f"platform : {platform.system()}")
    print(f"backend  : {backend_name() or 'NONE'}")
    problem = missing_dependency()
    if problem:
        print(problem)
    print("\noutputs:")
    for i, label in list_outputs():
        print(f"   {label}")
    print("\nmicrophones:")
    for i, label in list_microphones():
        print(f"   {label}")
