"""Calls to Google's Agent Platform.

Auth is a short-lived access token handed in from the host (`gcloud auth print-access-token`), so
no Google credentials are ever stored in the image or the project folder.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

# v1beta1, not v1: on 2026-11-17 the documented v1 interactions endpoint accepted Lyria requests and
# answered "completed" with no output at all, while v1beta1 returned the audio.
API = "https://aiplatform.googleapis.com/v1beta1"
MUSIC_MODEL = "lyria-3-pro-preview"  # global location only; one track of up to 184 s per request
VIDEO_MODEL = "gemini-omni-1.2-flash"  # video edit: input and output must be the same 3-15 s length
TTS_API = "https://aiplatform.googleapis.com/v1"  # speech works on plain v1 generateContent
TTS_MODEL = "gemini-3.8-flash-tts"  # returns audio/wav and reports text and audio tokens
VOICES = (
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus",
    "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia",
    "Sadaltager", "Sulafat",
)
EXTENSIONS = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav"}


class GoogleError(RuntimeError):
    pass


def settings() -> tuple[str, str]:
    token = os.environ.get("GOOGLE_ACCESS_TOKEN", "").strip()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not token or not project:
        raise GoogleError(
            "GOOGLE_ACCESS_TOKEN and GOOGLE_CLOUD_PROJECT are not set in the container; "
            "CLAUDE.md shows how to pass them from gcloud"
        )
    return token, project


def _call(url: str, token: str, body: dict | None = None, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise GoogleError(f"Google returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise GoogleError(f"could not reach Google: {exc.reason}") from None


def parse_speech(reply: dict) -> tuple[bytes, dict]:
    """(WAV bytes, token usage) from a generateContent reply that asked for audio."""
    parts = ((reply.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    for part in parts:
        blob = part.get("inlineData") or {}
        if blob.get("data"):
            usage = reply.get("usageMetadata") or {}
            return base64.b64decode(blob["data"]), {
                "text_tokens": usage.get("promptTokenCount", 0),
                "audio_tokens": usage.get("candidatesTokenCount", 0),
            }
    raise GoogleError(f"no audio in Google's reply: {json.dumps(reply)[:400]}")


def synthesize(text: str, voice: str, direction: str = "", model: str = TTS_MODEL) -> tuple[bytes, dict]:
    """Speak one line in a stock voice. `direction` is a short delivery note such as 'calm, measured'."""
    token, project = settings()
    url = f"{TTS_API}/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    # The delivery note goes in speechMetadata.style. Written into the text ("Say in a calm voice: ...")
    # this model reads the note aloud, which tripled the length of every line on 2026-11-17.
    part: dict = {"text": text}
    if direction.strip():
        part["speechMetadata"] = {"style": direction.strip()}
    body = {
        "contents": [{"role": "user", "parts": [part]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"voice": voice}},
        },
    }
    for attempt in range(3):
        try:
            return parse_speech(_call(url, token, body, timeout=120))
        except GoogleError as exc:
            if attempt == 2 or not any(code in str(exc) for code in ("HTTP 429", "HTTP 500", "HTTP 503", "reach")):
                raise
            time.sleep(4 * (attempt + 1))
    raise GoogleError("unreachable")


def edit_video(
    video: bytes, prompt: str, resolution: str = "360p", references: list[bytes] | None = None, model: str = VIDEO_MODEL
) -> tuple[bytes, dict]:
    """Restyle or edit one clip of 3 to 15 seconds with Gemini Omni. Returns (MP4 bytes, usage).

    `references` are up to two JPEG images that show how characters should look.
    """
    token, project = settings()
    body = {
        "model": model,
        "store": False,
        "input": [
            {"type": "text", "text": prompt},
            {"type": "video", "mime_type": "video/mp4", "data": base64.b64encode(video).decode()},
            *(
                {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image).decode()}
                for image in (references or [])[:2]
            ),
        ],
        "response_format": [{"type": "video", "delivery": "inline", "resolution": resolution}],
        "generation_config": {"video_config": {"task": "edit"}},
    }
    for attempt in range(4):  # the video model answers 429 when several shots are sent at once
        try:
            reply = _call(f"{API}/projects/{project}/locations/global/interactions", token, body, timeout=900)
            break
        except GoogleError as exc:
            if attempt == 3 or not any(code in str(exc) for code in ("HTTP 429", "HTTP 500", "HTTP 503")):
                raise
            time.sleep(20 * (attempt + 1))
    for part in _parts(reply):
        if part.get("type") == "video" and part.get("data"):
            return base64.b64decode(part["data"]), reply.get("usage") or {}
    raise GoogleError(f"no video in Google's reply (status {reply.get('status')!r}): {json.dumps(reply)[:500]}")


def _parts(reply: dict):
    """Output parts of an Interactions reply: v1beta1 lists them under `outputs`, the docs under `steps`."""
    yield from reply.get("outputs") or []
    for step in reply.get("steps") or []:
        yield from step.get("content") or []


def parse_music(reply: dict) -> tuple[bytes, str, list[str]]:
    """Pull the audio, its MIME type and any text notes out of an Interactions reply."""
    audio, mime, notes = None, "", []
    for part in _parts(reply):
        if part.get("type") == "audio" and part.get("data"):
            audio, mime = base64.b64decode(part["data"]), part.get("mime_type", "audio/mpeg")
        elif part.get("type") == "text" and part.get("text"):
            notes.append(part["text"])
    if audio is None:
        raise GoogleError(
            f"no audio in Google's reply (status {reply.get('status')!r}): {json.dumps(reply)[:500]}"
        )
    return audio, mime, notes


def generate_music(prompt: str, model: str = MUSIC_MODEL, wait_seconds: int = 600) -> tuple[bytes, str, list[str]]:
    """One music track from a text prompt. Returns (audio bytes, MIME type, the model's text notes)."""
    token, project = settings()
    base = f"{API}/projects/{project}/locations/global/interactions"
    reply = _call(
        base,
        token,
        {"model": model, "input": [{"type": "user_input", "content": [{"type": "text", "text": prompt}]}]},
    )
    deadline = time.monotonic() + wait_seconds
    while reply.get("status") in ("in_progress", "queued", "pending") and reply.get("id"):
        if time.monotonic() > deadline:
            raise GoogleError(f"music generation did not finish within {wait_seconds} s (id {reply['id']})")
        time.sleep(5)
        reply = _call(f"{base}/{reply['id']}", token)
    return parse_music(reply)
