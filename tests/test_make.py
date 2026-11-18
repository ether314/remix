"""Make a video from a prompt: projects, plan checks, paid steps (with fakes), voice script, render and check."""
import json
import subprocess

import pytest

from remix import cost, make, voices
from test_voices import fake_wav


@pytest.fixture()
def made(tmp_path, monkeypatch):
    from remix import config

    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    project = make.create("Depot ad", {"prompt": "an ad", "seconds": 6, "aspects": ["1:1"], "brand": "Acme", "voice": "Kore", "brand_in_hook": True})
    return project


def plan_for(project, **changes):
    plan = {
        "title": "Depot ad", "style": "cinematic", "voice": {"name": "Kore", "direction": "warm"}, "brand": {"name": "Acme", "in_hook": True},
        "characters": {"boss": "a manager in a navy jacket"},
        "shots": {"a": {"seconds": 4, "prompt": "{boss} walks through a yard, {style}."}},
        "scenes": [
            {"id": "one", "kind": "veo", "clip": "a", "seconds": 3.0, "vo": "Hello there friend.", "overlays": [{"type": "headline", "text": "Acme Fleet", "from": 0.1, "to": 2.9}]},
            {"id": "two", "kind": "title", "seconds": 3.0, "overlays": [{"type": "headline", "text": "Call us", "from": 0.1, "to": 1.9}]},
        ],
    }
    plan.update(changes)
    return plan


def test_a_project_has_a_manifest_so_the_existing_tools_work_on_it(made):
    assert make.is_make(made) and make.find("make-depot") == made and make.find("Depot ad") == made
    assert made.load()["source"]["duration"] == 6 and made.rights() == "own"
    with pytest.raises(make.MakeError, match="already exists"):
        make.create("Depot ad", {})
    with pytest.raises(make.MakeError, match="between 6 seconds"):
        make.create("Too short", {"seconds": 2})


def test_the_plan_check_names_the_usual_mistakes(made):
    plan = plan_for(made)
    assert make.save_plan(made, plan) == []
    bad = plan_for(made)
    bad["scenes"][0].update(clip="nope", vo="one two three four five six seven eight nine ten eleven twelve")
    bad["scenes"][1]["overlays"].append({"type": "sparkle"})
    bad["scenes"][1]["id"] = "one"
    bad["shots"]["a"]["seconds"] = 5
    bad["brand"] = {"name": "Zed", "in_hook": True}
    wrong = " | ".join(make.save_plan(made, bad))
    for expected in ("every scene needs its own id", "clip 'nope' is not defined", "words a second", "unknown overlay type 'sparkle'", "must be 4, 6 or 8",
                     "neither shown nor named in the first 5 seconds"):
        assert expected in wrong
    assert make.clip_length(3.2) == 4 and make.clip_length(5) == 6 and make.clip_length(8) == 8
    with pytest.raises(make.MakeError, match="at most 8"):
        make.clip_length(9)
    assert "a manager in a navy jacket walks through a yard, cinematic." in make.shot_prompt(plan, "a") and make.shot_prompt(plan, "a").endswith(make.NO_TEXT)


def test_shots_are_priced_kept_and_checked(made, monkeypatch):
    make.save_plan(made, plan_for(made))
    assert cost.total(make.estimate_shots(plan_for(made), "pilot")) == pytest.approx(0.12) and cost.total(make.estimate_shots(plan_for(made), "final")) == pytest.approx(0.40)
    asked = []

    def veo(prompt, model, aspect, resolution, seconds, negative):
        asked.append((model, resolution, seconds, aspect))
        return b"FAKE-MP4", {}

    def watch(clip, question, schema):
        return {"visible_text": ["STOP"], "has_watermark_or_logo": False, "artifacts": [], "summary": "a yard"}, {"input_tokens": 3000, "output_tokens": 40}

    result = make.generate_shots(made, "pilot", watch=watch, make=veo)
    assert result["made"] == ["a"] and asked == [("veo-3.1-lite-generate-001", "720p", 4, "16:9")]
    assert (made.root / "shots" / "a.pilot.mp4").is_file() and make.tiers_present(made, "a") == ["pilot"]
    assert "generated text in the picture (STOP)" in make.qa_problems(made, plan_for(made))[0]
    rows = cost.entries()
    assert [r["what"] for r in rows if r["project"] == made.slug][:1] == ["Veo clip a"] and sum(r["usd"] for r in rows) > 0.12
    assert make.generate_shots(made, "pilot", watch=watch, make=veo)["made"] == [] and len(asked) == 1  # kept clips are not paid for twice
    monkeypatch.setattr(make.config, "CAP_USD", 0.05)
    with pytest.raises(make.MakeError, match="over the \\$0.05 cap"):
        make.generate_shots(made, "final", watch=watch, make=veo)


def test_the_voice_over_becomes_a_timed_script_with_captions_that_can_opt_out(made):
    plan = plan_for(made)
    plan["scenes"][0]["captions_off"] = True
    plan["scenes"][1].update(vo="And call us today.", vo_at=0.3, speaker="cfo")
    plan["voices"] = {"cfo": "Orus"}
    make.save_plan(made, plan)
    path = make.write_script(made, plan)
    script = json.loads(path.read_text(encoding="utf-8"))
    assert [l["text"] for l in script["lines"]] == ["Hello there friend.", "And call us today."]
    assert script["lines"][0]["start"] == pytest.approx(0.25) and script["lines"][1]["start"] == pytest.approx(3.3)
    assert script["voices"] == {"narrator": "Kore", "cfo": "Orus"} and script["lines"][0]["captions"] is False
    result = make.speak(made, synth=lambda text, voice, direction="": (fake_wav(1.0), {"text_tokens": 3, "audio_tokens": 25}))
    assert result["lines"] == 2 and (voices.folder(made, make.REMIX) / "voice.wav").is_file()
    voices.write_captions(made, make.REMIX, size=(1080, 1080), stem="captions_1x1")
    srt = (voices.folder(made, make.REMIX) / "captions_1x1.srt").read_text(encoding="utf-8")
    assert "call us today" in srt.lower() and "Hello there" not in srt


def test_sound_effects_and_the_music_choice(made, monkeypatch):
    for kind in ("ping", "ding", "tick", "thud", "whoosh", "riser"):
        clip = make.sfx(kind)
        assert 0 < len(clip) < 48000 * 2 and abs(clip).max() <= 1.0
    with pytest.raises(make.MakeError, match="unknown sound effect"):
        make.sfx("kazoo")
    plan = plan_for(made)
    plan["music"] = {"prompt": "instrumental piano"}
    make.save_plan(made, plan)
    seen = []

    def generate(prompt, model):
        seen.append(model)
        return b"MP3", "audio/mpeg", ["notes"]

    path = make.score(made, generate=generate)
    assert path.name == "music.mp3" and seen == ["lyria-3-clip-preview"]  # a video of 30 s or less needs only a clip
    assert [r["what"] for r in cost.entries() if r["remix"] == "music"] == ["music clip"]


def test_a_plan_renders_to_a_checked_video(made, tmp_path):
    plan = plan_for(made)
    plan["scenes"][0].update(vo="Hello there friend.", vo_at=0.2)
    plan["scenes"][0]["sfx"] = [{"at": 0.1, "kind": "tick"}]
    make.save_plan(made, plan)
    clip = made.root / "shots" / "a.final.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    make.speak(made, synth=lambda text, voice, direction="": (fake_wav(1.4), {"text_tokens": 3, "audio_tokens": 25}))
    final = make.render(made, "1:1")
    facts = make.verify.stream_facts(final)
    assert abs(facts["video"]["seconds"] - 6.0) < 0.1 and abs(facts["audio"]["seconds"] - 6.0) < 0.2
    assert (made.root / "out" / "1x1" / "captions.srt").is_file()
    again = final.stat().st_mtime
    make.render(made, "1:1")  # unchanged scenes are not drawn twice
    assert not any(p.stat().st_mtime > again for p in (made.root / "work" / "1x1").glob("0*.mp4"))

    heard = [{"word": " Hello", "start": 0.2, "end": 0.5}, {"word": " there", "start": 0.55, "end": 0.9}, {"word": " friend.", "start": 0.95, "end": 1.4}]
    result = make.check(made, "1:1", listen=lambda path: heard)
    assert result["loudness"]["lufs"] is not None and result["seconds"]["plan"] == 6.0 and (made.root / result["sheet"]).is_file()
    assert any("frames have not been read" in p["problem"] for p in result["problems"])
    ok = make.check(made, "1:1", frames_ok=True, listen=lambda path: heard)
    assert not [p for p in ok["problems"] if "heard" in p["problem"] or "long" in p["problem"] or "captions" in p["problem"]], ok["problems"]

    with pytest.raises(make.MakeError, match="needs its shot"):
        (made.root / "shots" / "a.final.mp4").unlink()
        make.render(made, "1:1")


def test_a_clip_with_sound_is_priced_made_and_checked_as_spoken_on_camera(made):
    plan = plan_for(made)
    plan["shots"]["a"]["audio"] = True
    plan["scenes"][0].pop("vo")
    plan["scenes"][0]["on_camera"] = {"speaker": "mechanic", "text": "Truck twelve is back again."}
    assert make.save_plan(made, plan) == []
    assert cost.total(make.estimate_shots(plan, "final")) == pytest.approx(4 * 0.12)  # sound costs more than a silent clip
    asked = {}

    def veo(prompt, model, aspect, resolution, seconds, negative, audio=False):
        asked["audio"] = audio
        return b"FAKE", {}

    make.generate_shots(made, "pilot", watch=lambda *a: ({"visible_text": [], "has_watermark_or_logo": False, "artifacts": [], "summary": ""}, {}), make=veo)
    assert asked == {"audio": True}
    silent = plan_for(made)
    silent["scenes"][0]["on_camera"] = {"text": "Hello."}
    assert any("needs a veo scene whose shot has" in p for p in make.validate(made, silent))

    heard = [{"word": w, "start": 0.4 + i * 0.3, "end": 0.6 + i * 0.3} for i, w in enumerate(["Truck", "12", "is", "back", "again."])]
    assert make.on_camera_problems(made, plan, made.root / "x.mp4", lambda path: heard) == []
    wrong = make.on_camera_problems(made, plan, made.root / "x.mp4", lambda path: heard[:2])
    assert "not heard as written" in wrong[0]["problem"]


def test_only_one_render_of_a_project_runs_at_a_time(made):
    make.save_plan(made, plan_for(made))
    (made.root / "work").mkdir(exist_ok=True)
    (made.root / "work" / "render.lock").write_text("busy")
    with pytest.raises(make.MakeError, match="another render"):
        make.render(made, "1:1")
