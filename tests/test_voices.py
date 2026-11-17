"""Line lists, slot fitting, voice assembly (with a fake speech model) and captions."""
import io
import json

import numpy as np
import pytest
import soundfile as sf

from remix import cost, remixes, voices


def fake_wav(seconds: float) -> bytes:
    buffer = io.BytesIO()
    tone = 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * voices.RATE)) / voices.RATE)
    padded = np.concatenate([np.zeros(2400), tone, np.zeros(2400)]).astype(np.float32)  # a model's lead-in and tail
    sf.write(buffer, padded, voices.RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_script(project, name, lines):
    folder = voices.folder(project, name)
    folder.mkdir(parents=True)
    script = {"voices": {"host": "Charon", "guest": "Kore"}, "directions": {"host": "", "guest": "calm"}, "lines": lines}
    (folder / "lines.json").write_text(json.dumps(script), encoding="utf-8")


def test_fit_speed_and_cue_splitting():
    assert voices.fit_speed(1.0, 2.0) == voices.MIN_SPEED  # a short take is slowed a little toward the original's length
    assert voices.fit_speed(2.2, 2.0) == pytest.approx(1.1)
    assert voices.fit_speed(5.0, 2.0) == voices.MAX_SPEED and voices.fit_speed(1.0, 1.01) == pytest.approx(0.99, abs=0.01)
    cues = voices.split_cue(10.0, 14.0, "one two three four five six seven eight nine ten eleven twelve", max_chars=20)
    assert [c["text"] for c in cues] == ["one two three four", "five six seven eight", "nine ten eleven", "twelve"]
    assert cues[0]["start"] == 10.0 and cues[-1]["end"] == pytest.approx(14.0)
    assert voices.split_cue(0, 1, "short") == [{"start": 0, "end": 1.0, "text": "short"}]


def test_build_lines_takes_speakers_from_the_blueprint(project):
    (project.root / "transcript").mkdir()
    segments = [
        {"start": 0.2, "end": 1.0, "text": "Hello.", "words": []},
        {"start": 1.6, "end": 2.4, "text": "Hi there.", "words": []},
        {"start": 2.5, "end": 2.9, "text": "Bye.", "words": []},
    ]
    (project.root / "transcript" / "transcript.json").write_text(json.dumps({"segments": segments}))
    shots = [{"start": 0.0, "end": 1.5, "speaker": "host"}, {"start": 1.5, "end": 2.45, "speaker": "guest"},
             {"start": 2.45, "end": 3.0, "speaker": ""}]
    (project.root / "blueprint.json").write_text(json.dumps({"shots": shots}))

    script = voices.build_lines(project)

    assert [line["speaker"] for line in script["lines"]] == ["host", "guest", "guest"]  # silent shot keeps the last speaker
    assert script["voices"] == {"host": "Charon", "guest": "Kore"}
    voices.write_lines(project, "dub")
    with pytest.raises(voices.VoiceError, match="already exists"):
        voices.write_lines(project, "dub")


def test_a_segment_with_two_speakers_is_split_at_the_pause_not_the_cut(project):
    def word(text, start, end):
        return {"word": text, "start": start, "end": end}

    # The second speaker starts talking at 1.9, a beat before the picture cuts to them at 2.0.
    words = [word("How", 1.0, 1.2), word("so?", 1.2, 1.5), word("Like", 1.9, 2.05), word("this.", 2.1, 2.4)]
    (project.root / "transcript").mkdir()
    (project.root / "transcript" / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 1.0, "end": 2.4, "text": "How so? Like this.", "words": words}]})
    )
    shots = [{"start": 0.0, "end": 2.0, "speaker": "host"}, {"start": 2.0, "end": 3.0, "speaker": "guest"}]
    (project.root / "blueprint.json").write_text(json.dumps({"shots": shots}))

    lines = voices.build_lines(project)["lines"]

    assert [(line["speaker"], line["text"], line["start"], line["end"]) for line in lines] == [
        ("host", "How so?", 1.0, 1.5),
        ("guest", "Like this.", 1.9, 2.4),
    ]


def test_generate_places_lines_fits_long_ones_and_logs_the_cost(project):
    write_script(project, "dub", [
        {"id": 1, "start": 0.2, "end": 1.0, "speaker": "host", "text": "Fits easily."},
        {"id": 2, "start": 1.2, "end": 1.8, "speaker": "guest", "text": "Too long for its slot."},
        {"id": 3, "start": 2.0, "end": 2.6, "speaker": "host", "text": "Last."},
    ])
    asked = []

    def synth(text, voice, direction=""):
        asked.append((text, voice, direction))
        return fake_wav({"Fits easily.": 0.5, "Too long for its slot.": 0.9, "Last.": 0.3}[text]), {"text_tokens": 5, "audio_tokens": 20}

    result = voices.generate(project, "dub", synth=synth, workers=1, rerolls=0)

    assert asked[1] == ("Too long for its slot.", "Kore", "calm")
    first, second, third = result["report"]
    assert first["speed"] == voices.MIN_SPEED and first["words_start"] == 0.2 and first["overrun"] == 0  # words start on the slot
    assert second["speed"] == voices.SHORT_SPEED and second["seconds"] < 0.9 and second["end_error"] > 0  # a short line may be tightened to 1.25x
    assert result["sped_up"] == 1 and result["slowed"] == 2 and result["tokens"] == {"text_tokens": 15, "audio_tokens": 60}
    assert result["suspect"] == 0 and result["file"] == "voice.wav"
    assert voices.too_long("Thank you.", 5.2) and not voices.too_long("Thank you.", 0.9)

    samples, rate = sf.read(str(voices.folder(project, "dub") / "voice.wav"))
    assert rate == voices.RATE and abs(len(samples) / rate - project.load()["source"]["duration"]) < 0.1
    quiet, voiced = np.abs(samples[: int(0.1 * rate)]).max(), np.abs(samples[int(0.3 * rate) : int(0.6 * rate)]).max()
    assert quiet < 0.01 < voiced  # silence before the first line, sound inside it

    rows = cost.entries()
    assert [(row["price_id"], row["quantity"]) for row in rows] == [("flash-tts.audio-out", 60), ("flash-tts.text-in", 15)]
    assert result["usd"] == pytest.approx(sum(row["usd"] for row in rows))


def test_unknown_voices_are_refused_before_any_call(project):
    write_script(project, "dub", [{"id": 1, "start": 0, "end": 1, "speaker": "host", "text": "Hi."}])
    path = voices.folder(project, "dub") / "lines.json"
    script = json.loads(path.read_text())
    script["voices"]["host"] = "Morgan"
    path.write_text(json.dumps(script))
    with pytest.raises(voices.VoiceError, match="unknown voice"):
        voices.generate(project, "dub", synth=lambda *a: pytest.fail("should not be called"))


def test_captions_and_mixing_with_new_voices(project, tmp_path, tone):
    write_script(project, "dub", [
        {"id": 1, "start": 0.2, "end": 1.4, "speaker": "host", "text": "Hello there."},
        {"id": 2, "start": 1.6, "end": 2.8, "speaker": "guest", "text": "Hi {you}."},
    ])
    result = voices.write_captions(project, "dub")
    folder = voices.folder(project, "dub")
    assert result["cues"] == 2
    assert "00:00:00,200 --> 00:00:01,400\nHello there." in (folder / "captions.srt").read_text()
    ass = (folder / "captions.ass").read_text()
    assert "Style: host," in ass and "Style: guest," in ass
    assert "Dialogue: 0,0:00:01.60,0:00:02.80,guest,Hi (you)." in ass

    voices.generate(project, "dub", synth=lambda text, voice, direction="": (fake_wav(0.5), {"text_tokens": 1, "audio_tokens": 1}))
    project.set_rights("own")
    record = remixes.audio_remix(project, "dub", tone(tmp_path / "bed.wav", 220, 3), voice=folder / "voice.wav")
    assert record["voices"] == "new"
    burned = voices.burn_captions(project, "dub", project.root / record["output"])
    assert burned.is_file() and burned.stat().st_size > 1000
