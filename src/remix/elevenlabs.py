"""ElevenLabs as an audio source, next to Gemini.

The takes are saved into folders of the project, and everything after that (fitting each take to its slot, captions,
the check) is shared with every other voice:

    <remix folder>/eleven/NNNN.mp3     one file per script line; NNNN is the line's id
    <project>/eleven/music.mp3         the score, when ElevenLabs makes the music too

Two ways to make them. With an API key in the environment (ELEVENLABS_API_KEY, set by the user in their own
terminal and passed into the container like the Google token) the official `elevenlabs` SDK speaks the lines and
composes the music itself (`fill_voice`, `fill_music`). Without a key the user's signed-in browser is driven by
hand and the takes are dropped into the same folders. `order()` writes the sheet of what to make (`order.md`,
`order.json`); `Takes` is the voice that `voices.generate` calls instead of Gemini and hands back the saved file for
a line as a WAV. Nothing is paid per call in dollars: speech and music come out of the account's ElevenLabs credits,
so the ledger records characters and tracks at $0.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import voices
from .project import Project

KEY_ENV = "ELEVENLABS_API_KEY"  # set by the user in their own environment; never stored in the project or sent anywhere but ElevenLabs
OUTPUT_FORMAT = "mp3_44100_128"
VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75}  # the web app's defaults (the page showed stability 50%, similarity 75%)
MAX_WORKERS = 2  # concurrent requests an account may make varies by plan; two is safe on every plan
MUSIC_MODEL = "music_v2_5"  # the Music page showed v2.5; ELEVENLABS_MUSIC_MODEL overrides

PROVIDERS = ("gemini", "elevenlabs")
DEFAULT_PROVIDER = "gemini"
DEFAULT_VOICE = "Zoe - Witty and Reassuring"  # a British female voice in the Voice Library; the closest to the reference ad
MODEL = "Eleven v4"
EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")
# Gemini's angle-bracket cues become ElevenLabs audio tags (square brackets). A cue with no counterpart is dropped,
# since an unknown tag could be read aloud.
CUES = {
    "sigh": "[sighs]", "sighs": "[sighs]", "chuckle": "[chuckles]", "chuckles": "[chuckles]", "laugh": "[laughs]", "laughter": "[laughs]",
    "giggle": "[giggles]", "breath": "[inhales]", "heavy breath": "[exhales]", "exhales": "[exhales]", "phew": "[exhales]",
    "gasp": "[gasps]", "groan": "[groans]", "cough": "[coughs]", "throat-clearing": "[clears throat]", "yawn": "[yawns]",
    "whispers": "[whispers]", "whispering": "[whispers]", "short pause": "[short pause]", "long pause": "[long pause]",
}


class ElevenLabsError(RuntimeError):
    pass


def provider_of(value: str | None) -> str:
    name = (value or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise ElevenLabsError(f"audio provider must be one of {', '.join(PROVIDERS)}, not '{value}'")
    return name


def speech_text(text: str) -> str:
    """What to type into ElevenLabs for a line's performed text: cues become audio tags, spacing is tidied."""
    swapped = voices.TAG.sub(lambda m: " " + CUES.get(m.group(1).strip().lower(), "") + " ", text)
    return re.sub(r"\s+", " ", swapped).replace(" ,", ",").replace(" .", ".").strip()


def folder(project: Project, name: str) -> Path:
    return voices.folder(project, name) / "eleven"


def find_file(where: Path, stem: str) -> Path | None:
    for ext in EXTENSIONS:
        path = where / f"{stem}{ext}"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def music_file(project: Project) -> Path | None:
    return find_file(project.root / "eleven", "music")


def order(project: Project, name: str, script: dict, music: dict | None = None, write: bool = True) -> dict:
    """What has to be made in ElevenLabs for this script, and which of it is already saved."""
    where = folder(project, name)
    rows = []
    for line in script["lines"]:
        have = find_file(where, f"{line['id']:04d}")
        rows.append({"id": line["id"], "speaker": line["speaker"], "voice": script["voices"][line["speaker"]],
                     "text": speech_text(voices.performed(line)), "file": project.rel(where / f"{line['id']:04d}.mp3"), "have": have is not None})
    result = {"folder": project.rel(where), "model": MODEL, "lines": rows, "missing": [r["id"] for r in rows if not r["have"]]}
    if music is not None:
        have_music = music_file(project)
        result["music"] = {**music, "file": project.rel(project.root / "eleven" / "music.mp3"), "have": have_music is not None}
    if write:
        where.mkdir(parents=True, exist_ok=True)
        (where / "order.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (where / "order.md").write_text(sheet(project, name, result), encoding="utf-8")
    return result


def sheet(project: Project, name: str, result: dict) -> str:
    out = [f"# ElevenLabs order: {project.slug} / {name}", "",
           f"In ElevenLabs, open Text to Speech (model {result['model']}, stability 50%, similarity 75%). For each row: pick the voice, "
           "paste the text exactly as written, generate, download the take, and save it at the path in the row "
           "(relative to the project folder; an .mp3 or .wav both work). Then run the voice step again; it fits each take to its slot.", "",
           "| id | voice | file | text | status |", "|---|---|---|---|---|"]
    for row in result["lines"]:
        out.append(f"| {row['id']} | {row['voice']} | {row['file']} | {row['text']} | {'made' if row['have'] else 'TO MAKE'} |")
    if "music" in result:
        music = result["music"]
        out += ["", "## Music", "", f"About {music.get('seconds', 30):g} seconds, instrumental. Save as `{music['file']}` "
                f"in the project folder ({'made' if music['have'] else 'TO MAKE'}).", "", "Prompt:", "", music.get("prompt", "")]
    return "\n".join(out) + "\n"


class Takes:
    """The voice `voices.generate` calls for ElevenLabs: it returns the saved take for a line instead of speaking."""

    def __init__(self, project: Project, name: str, script: dict):
        self.files: dict[str, Path] = {}
        where = folder(project, name)
        for line in script["lines"]:
            path = find_file(where, f"{line['id']:04d}")
            if path is not None:
                self.files.setdefault(speech_text(voices.performed(line)), path)

    def _file(self, text: str) -> Path:
        path = self.files.get(speech_text(text))
        if path is None:
            raise ElevenLabsError(f"no ElevenLabs take is saved for: {text[:60]}")
        return path

    def fingerprint(self, text: str) -> str:
        """Changes when the saved file does, so a take made again in ElevenLabs replaces the one `generate` kept."""
        return hashlib.sha1(self._file(text).read_bytes()).hexdigest()[:12]

    def __call__(self, text: str, voice: str, direction: str = "") -> tuple[bytes, dict]:
        path = self._file(text)
        try:
            samples = voices._decode(path.read_bytes())
        except Exception as exc:  # ffmpeg could not read it
            raise ElevenLabsError(f"could not read {path.name} as audio: {exc}") from None
        return voices._encode(samples), {"characters": len(speech_text(text))}


# --- the API (the official SDK) ----------------------------------------------------------------------------


def api_key() -> str:
    return os.environ.get(KEY_ENV, "").strip()


def sdk_available() -> bool:
    try:
        import elevenlabs  # noqa: F401
    except ImportError:
        return False
    return True


def api_ready() -> bool:
    """Whether the takes can be made through the API: a key is set and the SDK is installed."""
    return bool(api_key()) and sdk_available()


_cache: dict[str, object] = {}


def client():
    """The SDK's client, made once from the key in the environment."""
    if "client" not in _cache:
        if not api_key():
            raise ElevenLabsError(f"no ElevenLabs API key: set {KEY_ENV} in your own environment (see CLAUDE.md, 'ElevenLabs')")
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError:
            raise ElevenLabsError("the elevenlabs package is not installed in this image: run `docker compose build` (see CLAUDE.md)") from None
        _cache["client"] = ElevenLabs(api_key=api_key())
    return _cache["client"]


def _guard(action):
    """Run one SDK call: wait and retry when ElevenLabs is busy, and turn its errors into ElevenLabsError (the key is never shown)."""
    for attempt in range(3):
        try:
            return action()
        except ElevenLabsError:
            raise
        except Exception as exc:  # the SDK's ApiError carries status_code and body; anything else is a network problem
            status = getattr(exc, "status_code", None)
            if (status in (429, 500, 502, 503) or status is None) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            if status is None:
                raise ElevenLabsError(f"could not reach ElevenLabs: {type(exc).__name__}") from None
            body = getattr(exc, "body", None)
            detail = (body.get("detail") if isinstance(body, dict) else body) or ""
            detail = detail.get("message", detail) if isinstance(detail, dict) else detail
            raise ElevenLabsError(f"ElevenLabs answered HTTP {status}: {str(detail)[:300]}") from None
    raise ElevenLabsError("unreachable")


def tts_model() -> str:
    """The model to speak with: ELEVENLABS_MODEL if set, else the newest text-to-speech model the account lists (v4, then v3)."""
    if os.environ.get("ELEVENLABS_MODEL"):
        return os.environ["ELEVENLABS_MODEL"]
    if "model" not in _cache:
        usable = [m for m in _guard(lambda: client().models.list()) if getattr(m, "can_do_text_to_speech", False)]
        if not usable:
            raise ElevenLabsError("the account lists no text-to-speech model")
        for token in ("v4", "v3"):
            hit = [m for m in usable if token in f"{m.model_id} {m.name}".lower()]
            if hit:
                _cache["model"] = hit[0].model_id
                break
        else:
            _cache["model"] = usable[0].model_id
    return str(_cache["model"])


def find_voice(name: str) -> str:
    """The voice id for a voice named in the plan: an id passes through, a name is looked up among the account's voices."""
    if re.fullmatch(r"[A-Za-z0-9]{20}", name.strip()):
        return name.strip()
    if "voices" not in _cache:
        _cache["voices"] = list(_guard(lambda: client().voices.get_all()).voices)
    wanted = name.strip().lower()
    short = wanted.split(" - ")[0].strip()
    for pick in (lambda v: v.name.lower() == wanted, lambda v: v.name.lower().split(" - ")[0].strip() == short):
        hit = [v for v in _cache["voices"] if pick(v)]
        if hit:
            return hit[0].voice_id
    raise ElevenLabsError(f"no voice called '{name}' in your ElevenLabs account. Add it from the Voice Library (Voices, then Add) "
                          "or put its voice id in the plan's voice.elevenlabs")


def credits_left() -> int | None:
    """Characters the account may still use this period (None if it cannot be read)."""
    try:
        sub = _guard(lambda: client().user.subscription.get())
        return int(sub.character_limit) - int(sub.character_count)
    except (ElevenLabsError, AttributeError, TypeError, ValueError):
        return None


def speak_line(text: str, voice_id: str, model: str | None = None) -> bytes:
    """One line, spoken: MP3 bytes."""
    from elevenlabs import VoiceSettings

    audio = _guard(lambda: b"".join(client().text_to_speech.convert(
        voice_id=voice_id, text=text, model_id=model or tts_model(), output_format=OUTPUT_FORMAT, voice_settings=VoiceSettings(**VOICE_SETTINGS))))
    if len(audio) < 1000:
        raise ElevenLabsError("ElevenLabs returned no audio for a line")
    return audio


def fill_voice(project: Project, name: str, script: dict, ids: list[int] | None = None) -> dict:
    """Speak the lines that have no saved take through the API and save each as NNNN.mp3. Returns what was spoken."""
    order_now = order(project, name, script, write=False)
    todo = [row for row in order_now["lines"] if not row["have"] and (ids is None or row["id"] in ids)]
    if not todo:
        return {"spoken": [], "characters": 0}
    need = sum(len(row["text"]) for row in todo)
    left = credits_left()
    if left is not None and left < need:
        raise ElevenLabsError(f"these lines need about {need} characters of ElevenLabs credit and the account has {left} left")
    model, where = tts_model(), folder(project, name)
    where.mkdir(parents=True, exist_ok=True)
    voice_ids = {voice: find_voice(voice) for voice in {row["voice"] for row in todo}}

    def one(row: dict) -> int:
        (where / f"{row['id']:04d}.mp3").write_bytes(speak_line(row["text"], voice_ids[row["voice"]], model))
        return row["id"]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        spoken = sorted(pool.map(one, todo))
    return {"spoken": spoken, "characters": need}


def fill_music(project: Project, prompt: str, seconds: float) -> Path:
    """Compose the score through the API (instrumental) and save it as eleven/music.mp3."""
    if not prompt:
        raise ElevenLabsError("the plan has no music prompt")
    length = int(min(max(seconds, 3.0), 600.0) * 1000)
    audio = _guard(lambda: b"".join(client().music.compose(
        prompt=prompt[:4000], music_length_ms=length, model_id=os.environ.get("ELEVENLABS_MUSIC_MODEL", MUSIC_MODEL),
        force_instrumental=True, output_format=OUTPUT_FORMAT)))
    if len(audio) < 1000:
        raise ElevenLabsError("ElevenLabs returned no music")
    target = project.root / "eleven" / "music.mp3"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(audio)
    return target


def status() -> dict:
    """Is the API connected? Reads only (the key's presence, the account's credits, the model); nothing is spoken or spent."""
    if not api_key():
        return {"connected": False, "reason": f"{KEY_ENV} is not set in the environment"}
    if not sdk_available():
        return {"connected": False, "reason": "the elevenlabs package is not installed in this image (docker compose build)"}
    try:
        return {"connected": True, "credits_left": credits_left(), "model": tts_model()}
    except ElevenLabsError as exc:
        return {"connected": False, "reason": str(exc)}
