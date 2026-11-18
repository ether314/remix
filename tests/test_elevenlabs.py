"""ElevenLabs as an audio provider: takes made in the user's account are ordered, fitted and recorded like any other voice."""
import json

import pytest

from remix import cost, elevenlabs, make, voices
from test_make import made, plan_for  # noqa: F401  (the `made` fixture)
from test_voices import fake_wav

ZOE = "Zoe - Witty and Reassuring"


def eleven_plan(project, **changes):
    plan = plan_for(project, **changes)
    plan["voice"] = {"name": "Kore", "direction": "dry", "elevenlabs": ZOE}
    plan["music"] = {"prompt": "dry plucked underscore"}
    make.save_plan(project, plan)
    make.set_audio_provider(project, "elevenlabs")
    return plan


def test_gemini_cues_become_elevenlabs_audio_tags_and_unknown_ones_are_dropped():
    assert elevenlabs.speech_text("<sigh> Well, <short pause> okay.") == "[sighs] Well, [short pause] okay."
    assert elevenlabs.speech_text("Hi <pff> there.") == "Hi there."
    assert elevenlabs.speech_text("Plain words.") == "Plain words."


def test_the_provider_is_a_brief_setting_and_unknown_ones_are_refused(made):
    assert make.audio_provider(made) == "gemini"
    assert make.set_audio_provider(made, "ElevenLabs") == "elevenlabs" and make.audio_provider(made) == "elevenlabs"
    with pytest.raises(make.MakeError, match="must be one of"):
        make.set_audio_provider(made, "azure")
    plan = plan_for(made)  # no ElevenLabs voice is named
    assert any("names no ElevenLabs voice" in p for p in make.save_plan(made, plan))
    plan["voice"]["elevenlabs"] = ZOE
    assert make.save_plan(made, plan) == []


def test_an_elevenlabs_project_asks_for_takes_then_fits_them_at_zero_dollars(made):
    eleven_plan(made)
    order = make.audio_order(made)
    assert order["provider"] == "elevenlabs" and order["missing"] == [1] and order["lines"][0]["voice"] == ZOE
    assert order["music"]["have"] is False and (voices.folder(made, make.REMIX) / "eleven" / "order.md").is_file()
    with pytest.raises(make.MakeError, match="not made in ElevenLabs yet"):
        make.speak(made)

    take = voices.folder(made, make.REMIX) / "eleven" / "0001.wav"
    take.write_bytes(fake_wav(1.0))
    result = make.speak(made)
    assert result["lines"] == 1 and result["provider"] == "elevenlabs" and result["usd"] == 0.0 and result["characters"] == len("Hello there friend.")
    assert (voices.folder(made, make.REMIX) / "voice.wav").is_file()
    script = json.loads((voices.folder(made, make.REMIX) / "lines.json").read_text(encoding="utf-8"))
    assert script["audio"] == "elevenlabs" and script["voices"] == {"narrator": ZOE}
    rows = [r for r in cost.entries() if r["project"] == made.slug]
    assert [(r["provider"], r["usd"]) for r in rows] == [("elevenlabs", 0.0)]

    assert make.speak(made)["respoken"] == 0  # the same file: the kept take stands
    take.write_bytes(fake_wav(1.2))  # made again in ElevenLabs: the new file replaces it
    assert make.speak(made)["respoken"] == 1


def test_the_score_made_in_elevenlabs_is_brought_in(made):
    eleven_plan(made)
    with pytest.raises(make.MakeError, match="music is not made in ElevenLabs yet"):
        make.score(made)
    (made.root / "music.wav").write_bytes(b"OLD")
    (made.root / "eleven").mkdir()
    (made.root / "eleven" / "music.mp3").write_bytes(b"MP3DATA")
    path = make.score(made)
    assert path.name == "music.mp3" and path.read_bytes() == b"MP3DATA" and not (made.root / "music.wav").exists()  # one score only
    assert [(r["provider"], r["usd"]) for r in cost.entries() if r["remix"] == "music"] == [("elevenlabs", 0.0)]


def test_gemini_projects_are_unchanged(made):
    make.save_plan(made, plan_for(made))
    assert make.audio_order(made) == {"provider": "gemini"}
    result = make.speak(made, synth=lambda text, voice, direction="": (fake_wav(1.0), {"text_tokens": 3, "audio_tokens": 25}))
    assert result["provider"] == "gemini" and result["usd"] > 0
