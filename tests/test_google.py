"""The Google client's parsing and guards. Nothing here touches the network."""
import base64

import pytest

from remix import google


def reply_with(parts):
    return {"id": "abc", "status": "completed", "outputs": parts}  # the shape v1beta1 really returns


def test_parse_music_also_reads_the_documented_steps_shape():
    encoded = base64.b64encode(b"song").decode()
    reply = {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "audio", "data": encoded}]}]}
    assert google.parse_music(reply)[0] == b"song"


def test_parse_music_returns_audio_mime_and_notes():
    reply = reply_with(
        [
            {"type": "text", "text": "A slow piano piece."},
            {"type": "audio", "mime_type": "audio/mpeg", "data": base64.b64encode(b"ID3!").decode()},
        ]
    )
    audio, mime, notes = google.parse_music(reply)
    assert (audio, mime, notes) == (b"ID3!", "audio/mpeg", ["A slow piano piece."])
    assert google.EXTENSIONS[mime] == ".mp3"


def test_parse_music_reports_a_reply_without_audio():
    with pytest.raises(google.GoogleError, match="no audio"):
        google.parse_music(reply_with([{"type": "text", "text": "I can't make that."}]))


def test_settings_require_a_token_and_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo")
    with pytest.raises(google.GoogleError, match="not set"):
        google.settings()
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "token\r\n")  # gcloud on Windows ends lines with CRLF
    assert google.settings() == ("token", "demo")


def test_generate_music_polls_until_the_interaction_completes(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "token")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo")
    finished = reply_with([{"type": "audio", "mime_type": "audio/mpeg", "data": base64.b64encode(b"song").decode()}])
    replies = [{"id": "abc", "status": "in_progress"}, finished]
    calls = []

    def fake_call(url, token, body=None, timeout=600):
        calls.append((url, body))
        return replies.pop(0)

    monkeypatch.setattr(google, "_call", fake_call)
    monkeypatch.setattr(google.time, "sleep", lambda seconds: None)

    audio, mime, notes = google.generate_music("instrumental piano")

    assert audio == b"song" and notes == []
    base = "https://aiplatform.googleapis.com/v1beta1/projects/demo/locations/global/interactions"
    assert calls[0][0] == base
    assert calls[0][1]["model"] == "lyria-3-pro-preview"
    assert calls[0][1]["input"][0]["content"] == [{"type": "text", "text": "instrumental piano"}]
    assert calls[1] == (f"{base}/abc", None)
