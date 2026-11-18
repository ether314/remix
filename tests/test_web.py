"""The web app's guards and gates, served on a throwaway port. Jobs are stubbed: nothing heavy runs."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from remix import web


@pytest.fixture()
def app(project, monkeypatch):
    started = []
    monkeypatch.setattr(web.JOBS, "table", {})
    monkeypatch.setattr(
        web.JOBS, "start",
        lambda label, slug, args, usd=0.0: started.append(args) or {"id": "job1", "label": label, "status": "queued"},
    )
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None, headers=None, raw=None):
        send = {"X-Remix": "1", **(headers or {})} if (body is not None or raw is not None) else dict(headers or {})
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(base + path, data=data, headers=send)
        try:
            with urllib.request.urlopen(request) as reply:
                return reply.status, reply.headers, reply.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    yield call, project, started
    server.shutdown()


def test_parse_range_and_upload_names():
    assert web.parse_range(None, 100) is None
    assert web.parse_range("bytes=0-9", 100) == (0, 9)
    assert web.parse_range("bytes=90-", 100) == (90, 99)
    assert web.parse_range("bytes=-10", 100) == (90, 99)
    with pytest.raises(web.Refused):
        web.parse_range("bytes=200-300", 100)
    assert web.safe_upload_name("..\\..\\My Clip (1).MP4") == "my-clip-1.mp4"
    with pytest.raises(web.Refused):
        web.safe_upload_name("notes.txt")


def test_state_lists_the_project_and_serves_the_page(app):
    call, project, _ = app
    status, _, body = call("/api/state")
    assert status == 200 and json.loads(body)["projects"][0]["slug"] == project.slug
    status, headers, page = call("/")
    assert status == 200 and headers["Content-Type"].startswith("text/html") and b"<title>Remix</title>" in page


def test_writes_need_the_header_and_a_local_host(app):
    call, project, _ = app
    path = f"/api/project/{project.slug}/rights"
    data = json.dumps({"status": "own"}).encode()
    assert call(path, raw=data, headers={"X-Remix": "0"})[0] == 403
    assert call("/api/state", headers={"Host": "evil.example"})[0] == 403
    assert project.rights() is None
    assert call(path, {"status": "own"})[0] == 200 and project.rights() == "own"


def test_media_supports_ranges_and_stays_inside_the_project(app):
    call, project, _ = app
    status, headers, body = call(f"/media/{project.slug}/audio/original.wav", headers={"Range": "bytes=0-99"})
    assert status == 206 and len(body) == 100 and headers["Content-Range"].startswith("bytes 0-99/")
    assert call(f"/media/{project.slug}/..%2F..%2Fclip.mp4")[0] == 404
    assert call(f"/media/{project.slug}/missing.wav")[0] == 404


def test_paid_music_is_gated(app, monkeypatch):
    call, project, started = app
    path = f"/api/project/{project.slug}/music"
    status, _, body = call(path, {"name": "take 1", "prompt": "instrumental piano"})
    assert status == 409 and "Google access" in json.loads(body)["error"]

    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "token")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo")
    assert call(path, {"name": "take 1", "prompt": "  "})[0] == 400
    monkeypatch.setattr(web.config, "CAP_USD", 0.01)  # make an 8-cent song over the cap
    status, _, body = call(path, {"name": "take 1", "prompt": "instrumental piano"})
    assert status == 409 and "over the $0.01 cap" in json.loads(body)["error"] and started == []

    assert call(path, {"name": "take 1", "prompt": "instrumental piano", "approve_over_cap": True})[0] == 200
    assert started == [["music", project.slug, "--name", "take-1", "--prompt", "instrumental piano", "--yes"]]

    (project.root / "remixes" / "take-1").mkdir(parents=True)
    (project.root / "remixes" / "take-1" / "music.mp3").write_bytes(b"x")
    status, _, body = call(path, {"name": "take 1", "prompt": "again", "approve_over_cap": True})
    assert status == 409 and "already has a track" in json.loads(body)["error"]


def test_mixing_needs_rights_and_a_track_and_gates_need_a_blueprint(app):
    call, project, started = app
    base = f"/api/project/{project.slug}"
    status, _, body = call(base + "/mix", {"name": "take-1"})
    assert status == 409 and "no rights recorded" in json.loads(body)["error"]
    project.set_rights("own")
    assert call(base + "/mix", {"name": "take-1"})[0] == 409  # no track yet
    assert call(base + "/gate", {"gate": "blueprint"})[0] == 409  # nothing to approve
    assert call(base + "/gate", {"gate": "decompose"})[0] == 200 and "decompose" in project.load()["gates"]
    assert call(base + "/approve", {"name": "take-1"})[0] == 409  # not mixed
    status, _, body = call(base + "/voices", {"name": "take-1"})
    assert status == 409 and "Google access" in json.loads(body)["error"]  # paid, so it needs access first
    track = project.root / "remixes" / "take-1" / "music.mp3"
    track.parent.mkdir(parents=True)
    track.write_bytes(b"x")
    status, _, body = call(base + "/mix", {"name": "take-1", "voices": True})
    assert status == 409 and "no AI voices yet" in json.loads(body)["error"]
    assert call("/api/upload?name=notes.txt", raw=b"hello")[0] == 400
    assert started == []


def test_restyle_needs_a_look_a_pilot_and_a_tick_over_the_cap(app, monkeypatch):
    call, project, started = app
    base = f"/api/project/{project.slug}"
    project.set_rights("own")
    shots = [{"index": n, "start": n - 1.0, "end": float(n), "duration": 1.0} for n in (1, 2, 3)]
    (project.root / "timeline.json").write_text(json.dumps({"shots": shots, "sheets": []}))
    look = {"name": "Clay", "audio_from": "take-1", "style": "Claymation.", "characters": {"dave": "A man of thirty.", "cat": " "}}
    status, _, body = call(base + "/look", look)
    assert status == 409 and "pick a mixed take" in json.loads(body)["error"]  # nothing to take the sound from yet

    take = project.root / "remixes" / "take-1"
    take.mkdir(parents=True)
    (take / "output.mp4").write_bytes(b"x")
    (take / "lines.json").write_text(json.dumps({"voices": {}, "directions": {}, "lines": []}))
    assert call(base + "/look", look)[0] == 200
    folder = project.root / "remixes" / "clay"
    saved = json.loads((folder / "look.json").read_text())
    assert saved["audio_from"] == "take-1" and saved["characters"] == {"dave": "A man of thirty."} and (folder / "lines.json").is_file()

    status, _, body = call(base + "/restyle", {"name": "clay", "shots": [1, 2]})
    assert status == 409 and "Google access" in json.loads(body)["error"]
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "token")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo")
    status, _, body = call(base + "/restyle", {"name": "clay", "shots": []})
    assert status == 409 and "pilot" in json.loads(body)["error"] and started == []  # no full run before a pilot
    assert call(base + "/restyle", {"name": "clay", "shots": [2, 1]})[0] == 200
    assert started == [["restyle", project.slug, "--name", "clay", "--workers", "4", "--only", "1,2"]]

    (folder / "shots").mkdir()
    for n in (1, 2):
        (folder / "shots" / f"000{n}_restyled.mp4").write_bytes(b"x")
    status, _, body = call(base + "/restyle", {"name": "clay", "shots": [1, 2]})
    assert status == 409 and "nothing to make" in json.loads(body)["error"]  # piloted shots are never paid for twice
    status, _, body = call(base + "/look", look)
    assert status == 409 and "already has restyled shots" in json.loads(body)["error"]
    monkeypatch.setattr(web.config, "CAP_USD", 0.01)
    status, _, body = call(base + "/restyle", {"name": "clay"})
    assert status == 409 and "over the $0.01 cap" in json.loads(body)["error"]
    assert call(base + "/restyle", {"name": "clay", "approve_over_cap": True})[0] == 200
    assert started[-1] == ["restyle", project.slug, "--name", "clay", "--workers", "4", "--yes"]

    (folder / "remix.json").write_text(json.dumps({"mode": "restyle", "output": "remixes/clay/output.mp4", "pending": [3]}))
    status, _, body = call(base + "/approve", {"name": "clay"})
    assert status == 409 and "still show the original picture" in json.loads(body)["error"]
    assert call(base + "/frames", {"name": "clay"})[0] == 409  # no complete video to confirm yet
    view = json.loads(call(base)[2])
    assert "error" not in view, view
    clay = next(t for t in view["remixes"] if t["name"] == "clay")
    assert clay["restyled_shots"] == [1, 2] and clay["look"]["style"] == "Claymation." and view["restyle"]["usd_per_second"] > 0


def test_make_a_video_from_a_prompt_in_the_web_app(app, monkeypatch):
    call, project, started = app
    status, _, body = call("/api/make/new", {"title": "Depot ad", "prompt": "an ad", "seconds": 6, "aspects": ["16:9"], "brand": "Acme", "voice": "Kore"})
    assert status == 200
    slug = json.loads(body)["slug"]
    assert call("/api/make/new", {"title": "  "})[0] == 400

    state = json.loads(call("/api/state")[2])
    assert [m["slug"] for m in state["makes"]] == [slug] and slug not in [p["slug"] for p in state["projects"]] and "Kore" in state["voices"]
    view = json.loads(call(f"/api/make/{slug}")[2])
    assert view["plan"] is None and view["brief"]["brand"] == "Acme" and view["renders"]["16:9"]["final"] is None

    base = f"/api/make/{slug}"
    status, _, body = call(base + "/plan", {})
    assert status == 409 and "Google access" in json.loads(body)["error"]  # paid steps need Google access first
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "token")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo")
    assert call(base + "/plan", {})[0] == 200 and started[-1] == ["make", "plan", slug, "--draft"]

    assert call(base + "/saveplan", {"plan": "{not json"})[0] == 400
    plan = {
        "title": "Depot ad", "style": "cinematic", "voice": {"name": "Kore"}, "brand": {"name": "Acme", "in_hook": True},
        "shots": {"a": {"seconds": 4, "prompt": "a yard at dawn, {style}."}},
        "scenes": [
            {"id": "one", "kind": "veo", "clip": "a", "seconds": 3.0, "vo": "Hello there friend.", "overlays": [{"type": "headline", "text": "Acme Fleet", "from": 0.1, "to": 2.9}]},
            {"id": "two", "kind": "title", "seconds": 3.0, "overlays": [{"type": "headline", "text": "Call us", "from": 0.1, "to": 2.9}]},
        ],
    }
    reply = json.loads(call(base + "/saveplan", {"plan": json.dumps(plan)})[2])
    assert reply["problems"] == []
    view = json.loads(call(base)[2])
    assert view["problems"] == [] and view["estimates"]["pilot"] == pytest.approx(0.12) and [s["id"] for s in view["shots"]] == ["a"]

    status, _, body = call(base + "/shots", {"tier": "final"})
    assert status == 409 and "pilot the shots first" in json.loads(body)["error"]  # the expensive tier waits for the cheap one
    assert call(base + "/shots", {"tier": "pilot"})[0] == 200 and started[-1] == ["make", "shots", slug, "--tier", "pilot"]
    monkeypatch.setattr(web.config, "CAP_USD", 0.05)
    status, _, body = call(base + "/shots", {"tier": "pilot"})
    assert status == 409 and "over the $0.05 cap" in json.loads(body)["error"]
    assert call(base + "/shots", {"tier": "pilot", "approve_over_cap": True})[0] == 200 and started[-1][-1] == "--yes"
    assert call(base + "/shots", {"tier": "gold"})[0] == 400 and call(base + "/dance", {})[0] == 404
    assert call(base + "/render", {"aspect": "16:9", "preview": True})[0] == 200 and started[-1] == ["make", "render", slug, "--aspect", "16:9", "--preview"]
    assert call("/api/make/nope")[0] == 404


def test_the_audio_provider_can_be_picked_and_elevenlabs_steps_are_free_but_need_the_takes(app):
    from remix import make as make_module, voices
    from test_voices import fake_wav

    call, project, started = app
    reply = call("/api/make/new", {"title": "Depot audio", "prompt": "an ad", "seconds": 6, "aspects": ["16:9"], "brand": "Acme",
                                   "audio_provider": "elevenlabs", "elevenlabs_voice": "Zoe - Witty and Reassuring"})
    slug = json.loads(reply[2])["slug"]
    assert call("/api/make/new", {"title": "Other", "audio_provider": "azure"})[0] == 400
    base = f"/api/make/{slug}"
    assert json.loads(call(base)[2])["audio"]["provider"] == "elevenlabs"
    plan = {
        "title": "Depot audio", "style": "cinematic", "voice": {"name": "Kore", "elevenlabs": "Zoe - Witty and Reassuring"}, "brand": {"name": "Acme", "in_hook": True},
        "music": {"prompt": "plucked underscore"}, "shots": {"a": {"seconds": 4, "prompt": "a yard at dawn, {style}."}},
        "scenes": [
            {"id": "one", "kind": "veo", "clip": "a", "seconds": 3.0, "vo": "Hello there friend.", "overlays": [{"type": "headline", "text": "Acme Fleet", "from": 0.1, "to": 2.9}]},
            {"id": "two", "kind": "title", "seconds": 3.0, "overlays": [{"type": "headline", "text": "Call us", "from": 0.1, "to": 2.9}]},
        ],
    }
    assert json.loads(call(base + "/saveplan", {"plan": json.dumps(plan)})[2])["problems"] == []
    audio = json.loads(call(base)[2])["audio"]
    assert audio["missing"] == [1] and audio["lines"][0]["voice"].startswith("Zoe") and audio["music"]["have"] is False

    # no Google access is set, and none is needed: these steps cost no dollars, but they wait for the takes
    status, _, body = call(base + "/voice", {})
    assert status == 409 and "not made in ElevenLabs yet" in json.loads(body)["error"]
    status, _, body = call(base + "/music", {})
    assert status == 409 and "music is not made in ElevenLabs yet" in json.loads(body)["error"]
    made = make_module.find(slug)
    (voices.folder(made, make_module.REMIX) / "eleven").mkdir(parents=True, exist_ok=True)
    (voices.folder(made, make_module.REMIX) / "eleven" / "0001.wav").write_bytes(fake_wav(1.0))
    (made.root / "eleven").mkdir(exist_ok=True)
    (made.root / "eleven" / "music.mp3").write_bytes(b"MP3")
    assert json.loads(call(base)[2])["audio"]["missing"] == []
    assert call(base + "/voice", {})[0] == 200 and started[-1] == ["make", "voice", slug]
    assert call(base + "/music", {})[0] == 200 and started[-1] == ["make", "music", slug]

    # back to Gemini: paid again, so Google access is needed
    assert json.loads(call(base + "/provider", {"provider": "gemini"})[2]) == {"provider": "gemini"}
    status, _, body = call(base + "/voice", {})
    assert status == 409 and "Google access" in json.loads(body)["error"]
    assert call(base + "/provider", {"provider": "azure"})[0] == 400
