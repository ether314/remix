"""Calls to Google's Agent Platform.

Auth is a short-lived access token handed in from the host (`gcloud auth print-access-token`), so
no Google credentials are ever stored in the image or the project folder.
"""
from __future__ import annotations

import base64
import http.client
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
    except (http.client.HTTPException, ConnectionError, TimeoutError) as exc:  # the connection dropped mid-answer
        raise GoogleError(f"could not reach Google: {type(exc).__name__}: {exc}") from None


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
            if attempt == 3 or not any(code in str(exc) for code in ("HTTP 429", "HTTP 500", "HTTP 503", "reach")):
                raise
            time.sleep(20 * (attempt + 1))
    for part in _parts(reply):
        if part.get("type") == "video" and part.get("data"):
            return base64.b64decode(part["data"]), reply.get("usage") or {}
    raise GoogleError(f"no video in Google's reply (status {reply.get('status')!r}): {json.dumps(reply)[:500]}")


VISION_MODEL = "gemini-3.8-flash"  # reads a clip and answers in JSON; billed per token (see prices.yaml)


def watch_video(video: bytes, prompt: str, schema: dict, model: str = VISION_MODEL, fps: int = 10) -> tuple[dict, dict]:
    """Show a short clip to Gemini and get a JSON answer shaped by `schema`. Returns (answer, token usage).

    The clip is sampled at `fps` frames a second (the default of 1 misses a mouth that opens and closes).
    """
    token, project = settings()
    url = f"{TTS_API}/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    clip = {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(video).decode()}}
    last = None
    for sampled in (True, False):  # fall back to the default sampling if the model rejects the setting
        part = {**clip, "videoMetadata": {"fps": fps}} if sampled else clip
        body = {
            "contents": [{"role": "user", "parts": [part, {"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": 0},
        }
        for attempt in range(3):
            try:
                reply = _call(url, token, body, timeout=300)
                break
            except GoogleError as exc:
                last = exc
                if "HTTP 400" in str(exc) and sampled:
                    reply = None
                    break
                if attempt == 2 or not any(code in str(exc) for code in ("HTTP 429", "HTTP 500", "HTTP 503", "reach")):
                    raise
                time.sleep(5 * (attempt + 1))
        else:
            reply = None
        if reply is not None:
            break
    else:
        raise last or GoogleError("Google did not answer")
    parts = ((reply.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    try:
        answer = json.loads(text)
    except ValueError:
        raise GoogleError(f"the model did not answer in JSON: {text[:300]}") from None
    usage = reply.get("usageMetadata") or {}
    return answer, {
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
    }


def generate_json(prompt: str, schema: dict, model: str = VISION_MODEL, system: str = "", temperature: float = 0.8) -> tuple[dict, dict]:
    """Ask Gemini for a JSON answer shaped by `schema` (text only). Returns (answer, token usage).

    Used to draft plans. The answer is checked by the caller; a schema only makes the shape likely.
    """
    token, project = settings()
    url = f"{TTS_API}/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": temperature},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    reply = None
    for attempt in range(3):
        try:
            reply = _call(url, token, body, timeout=300)
            break
        except GoogleError as exc:
            if attempt == 2 or not any(code in str(exc) for code in ("HTTP 429", "HTTP 500", "HTTP 503", "reach")):
                raise
            time.sleep(5 * (attempt + 1))
    parts = ((reply.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    try:
        answer = json.loads(text)
    except ValueError:
        raise GoogleError(f"the model did not answer in JSON: {text[:300]}") from None
    usage = reply.get("usageMetadata") or {}
    return answer, {
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
    }


VEO_API = "https://us-central1-aiplatform.googleapis.com/v1"
VEO_MODELS = {  # model id -> price table prefix (prices.yaml)
    "veo-3.1-generate-001": "veo-3.1",
    "veo-3.1-fast-generate-001": "veo-3.1-fast",
    "veo-3.1-lite-generate-001": "veo-3.1-lite",
}


def veo_price_id(model: str, resolution: str, audio: bool) -> str:
    return f"{VEO_MODELS[model]}.{'video-audio' if audio else 'video'}.{resolution}"


def generate_veo(
    prompt: str, model: str = "veo-3.1-fast-generate-001", aspect: str = "16:9", resolution: str = "1080p",
    seconds: int = 4, negative: str = "", seed: int | None = None, audio: bool = False, wait_seconds: int = 900,
) -> tuple[bytes, dict]:
    """Generate one clip with Veo from a text prompt. Returns (MP4 bytes, what Google reported).

    Veo is billed per second of output (4, 6 or 8 s), at the rate for the model, resolution and
    whether it makes sound. The request is a long-running operation that is polled until it is done.
    """
    token, project = settings()
    base = f"{VEO_API}/projects/{project}/locations/us-central1/publishers/google/models/{model}"
    parameters: dict = {
        "aspectRatio": aspect, "sampleCount": 1, "durationSeconds": seconds, "resolution": resolution,
        "personGeneration": "allow_adult", "generateAudio": audio,
    }
    if negative:
        parameters["negativePrompt"] = negative
    if seed is not None:
        parameters["seed"] = seed
    started = _call(f"{base}:predictLongRunning", token, {"instances": [{"prompt": prompt}], "parameters": parameters})
    name = started.get("name")
    if not name:
        raise GoogleError(f"Veo did not start an operation: {json.dumps(started)[:400]}")
    deadline = time.monotonic() + wait_seconds
    while True:
        time.sleep(10)
        reply = _call(f"{base}:fetchPredictOperation", token, {"operationName": name}, timeout=120)
        if reply.get("done"):
            break
        if time.monotonic() > deadline:
            raise GoogleError(f"Veo did not finish within {wait_seconds} s ({name})")
    if reply.get("error"):
        raise GoogleError(f"Veo failed: {json.dumps(reply['error'])[:500]}")
    response = reply.get("response") or {}
    for video in response.get("videos") or response.get("generatedSamples") or []:
        if video.get("bytesBase64Encoded"):
            return base64.b64decode(video["bytesBase64Encoded"]), {"model": model, "seconds": seconds, "operation": name}
    filtered = response.get("raiMediaFilteredReasons") or response.get("raiMediaFilteredCount")
    raise GoogleError(f"no video in Veo's reply ({'filtered: ' + json.dumps(filtered)[:300] if filtered else json.dumps(response)[:400]})")


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
