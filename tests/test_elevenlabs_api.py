"""The ElevenLabs SDK path, with a fake client: nothing here reaches the network or needs a real key."""
import json
from types import SimpleNamespace

import pytest

from remix import cost, elevenlabs, make, voices
from test_elevenlabs import ZOE, eleven_plan, made  # noqa: F401  (the `made` fixture)
from test_voices import fake_wav

VOICE_ID = "a" * 20


class FakeClient:
    """Just the parts of the SDK the project uses, recording what it was asked."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], list(fail or [])
        models = [SimpleNamespace(model_id="eleven_v3", name="Eleven v3", can_do_text_to_speech=True),
                  SimpleNamespace(model_id="eleven_v4", name="Eleven v4", can_do_text_to_speech=True),
                  SimpleNamespace(model_id="eleven_sfx", name="Sound effects", can_do_text_to_speech=False)]
        self.models = SimpleNamespace(list=lambda: models)
        self.voices = SimpleNamespace(get_all=lambda: SimpleNamespace(voices=[SimpleNamespace(voice_id=VOICE_ID, name=ZOE)]))
        self.user = SimpleNamespace(subscription=SimpleNamespace(get=lambda: SimpleNamespace(character_limit=50000, character_count=1200)))
        self.text_to_speech = SimpleNamespace(convert=self.convert)
        self.music = SimpleNamespace(compose=self.compose)

    def convert(self, **kwargs):
        self.calls.append(("speech", kwargs))
        if self.fail:
            raise self.fail.pop(0)
        return iter([fake_wav(1.0)])  # the SDK yields bytes in chunks

    def compose(self, **kwargs):
        self.calls.append(("music", kwargs))
        return iter([b"MUSIC" * 400])


class Throttled(Exception):
    def __init__(self, status, body=None):
        self.status_code, self.body = status, body


@pytest.fixture()
def api(monkeypatch):
    pytest.importorskip("elevenlabs")  # speak_line builds the SDK's VoiceSettings
    fake = FakeClient()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ELEVENLABS_MODEL", raising=False)
    monkeypatch.delenv("ELEVENLABS_MUSIC_MODEL", raising=False)
    monkeypatch.setattr(elevenlabs, "client", lambda: fake)
    monkeypatch.setattr(elevenlabs.time, "sleep", lambda s: None)
    elevenlabs._cache.clear()
    yield fake
    elevenlabs._cache.clear()


def test_the_voice_step_speaks_the_missing_lines_through_the_sdk(made, api):
    eleven_plan(made)
    result = make.speak(made)  # no take is saved, a key is set: the SDK makes it
    assert result["lines"] == 1 and result["provider"] == "elevenlabs" and result["usd"] == 0.0
    kind, call = api.calls[0]
    assert kind == "speech" and call["voice_id"] == VOICE_ID and call["model_id"] == "eleven_v4"  # the newest text-to-speech model
    assert call["text"] == "Hello there friend." and call["output_format"] == "mp3_44100_128"
    assert (voices.folder(made, make.REMIX) / "eleven" / "0001.mp3").is_file()
    api.calls.clear()
    assert make.speak(made)["respoken"] == 0 and api.calls == []  # a saved take is never paid for twice


def test_the_music_step_composes_through_the_sdk(made, api):
    eleven_plan(made)
    path = make.score(made)
    kind, call = api.calls[0]
    assert kind == "music" and call["music_length_ms"] == 6000 and call["force_instrumental"] is True and call["model_id"] == "music_v2_5"
    assert call["prompt"].startswith("dry plucked") and path.name == "music.mp3" and path.read_bytes().startswith(b"MUSIC")
    assert [(r["provider"], r["usd"]) for r in cost.entries() if r["remix"] == "music"] == [("elevenlabs", 0.0)]


def test_models_voices_and_credits_are_looked_up_not_assumed(api, monkeypatch):
    assert elevenlabs.tts_model() == "eleven_v4" and elevenlabs.find_voice(ZOE) == VOICE_ID
    assert elevenlabs.find_voice("zoe") == VOICE_ID and elevenlabs.find_voice("b" * 20) == "b" * 20  # a short name and a raw id both work
    with pytest.raises(elevenlabs.ElevenLabsError, match="no voice called 'Nobody'"):
        elevenlabs.find_voice("Nobody")
    assert elevenlabs.credits_left() == 48800
    monkeypatch.setenv("ELEVENLABS_MODEL", "my_model")
    assert elevenlabs.tts_model() == "my_model"
    assert elevenlabs.status() == {"connected": True, "credits_left": 48800, "model": "my_model"}


def test_a_line_that_needs_more_credit_than_the_account_has_is_refused_before_anything_is_spoken(made, api, monkeypatch):
    eleven_plan(made)
    api.user.subscription.get = lambda: SimpleNamespace(character_limit=100, character_count=95)
    with pytest.raises(make.MakeError, match="need about 19 characters of ElevenLabs credit and the account has 5 left"):
        make.speak(made)
    assert api.calls == []


def test_sdk_errors_read_plainly_never_show_the_key_and_throttling_is_retried(api):
    api.fail = [Throttled(429), Throttled(429)]
    assert elevenlabs.speak_line("Hello there friend.", VOICE_ID).startswith(b"RIFF")  # two busy answers, then success
    assert len(api.calls) == 3
    api.fail = [Throttled(401, {"detail": {"message": "Invalid API key"}})]
    with pytest.raises(elevenlabs.ElevenLabsError) as error:
        elevenlabs.speak_line("Hello there friend.", VOICE_ID)
    assert "HTTP 401: Invalid API key" in str(error.value) and "test-key-not-real" not in str(error.value)
    api.fail = [OSError("connection reset")] * 3
    with pytest.raises(elevenlabs.ElevenLabsError, match="could not reach ElevenLabs: OSError"):
        elevenlabs.speak_line("Hello there friend.", VOICE_ID)


def test_without_a_key_nothing_is_attempted_and_the_way_forward_is_named(made, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert not elevenlabs.api_ready() and elevenlabs.status()["connected"] is False
    eleven_plan(made)
    with pytest.raises(make.MakeError, match="or set ELEVENLABS_API_KEY"):
        make.speak(made)
    with pytest.raises(make.MakeError, match="so the API makes it"):
        make.score(made)
    with pytest.raises(elevenlabs.ElevenLabsError, match="no ElevenLabs API key"):
        elevenlabs.client()


def test_the_web_button_runs_when_the_api_is_connected_even_with_no_takes(app_with_api):
    call, started, slug = app_with_api
    assert json.loads(call(f"/api/make/{slug}")[2])["audio"]["api"] is True
    assert call(f"/api/make/{slug}/voice", {})[0] == 200 and started[-1] == ["make", "voice", slug]
    assert call(f"/api/make/{slug}/music", {})[0] == 200 and started[-1] == ["make", "music", slug]


@pytest.fixture()
def app_with_api(monkeypatch, tmp_path):
    """The web app with a make project set to ElevenLabs, and the API reported as connected."""
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from remix import config, web

    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)  # no Google access: only the free ElevenLabs path can run
    started = []
    monkeypatch.setattr(web.JOBS, "table", {})
    monkeypatch.setattr(web.JOBS, "start", lambda label, slug, args, usd=0.0: started.append(args) or {"id": "job1", "label": label, "status": "queued"})
    monkeypatch.setattr(web.elevenlabs, "api_ready", lambda: True)
    project = make.create("Depot web", {"prompt": "an ad", "seconds": 6, "aspects": ["16:9"], "brand": "Acme", "voice": "Kore", "brand_in_hook": True})
    eleven_plan(project)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(base + path, data=data, headers={"X-Remix": "1"} if body is not None else {})
        try:
            with urllib.request.urlopen(request) as reply:
                return reply.status, reply.headers, reply.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    yield call, started, project.slug
    server.shutdown()
