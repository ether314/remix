"""Sync enforced by code: one script behind the voices, captions and picture, and measured offsets."""
import json

import numpy as np
import pytest
import soundfile as sf

from remix import dialogue, mouths, remixes, restyle, verify, voices
from test_voices import fake_wav, write_script

LINES = [
    {"id": 1, "start": 0.2, "end": 1.2, "speaker": "host", "text": "Hello there."},
    {"id": 2, "start": 1.6, "end": 2.6, "speaker": "guest", "text": "Good morning."},
]


def synth(text, voice, direction=""):
    return fake_wav(0.5), {"text_tokens": 1, "audio_tokens": 1}


def rewrite(project, name, text):
    path = voices.folder(project, name) / "lines.json"
    script = json.loads(path.read_text())
    script["lines"][1]["text"] = text
    path.write_text(json.dumps(script))
    return script["lines"]


def test_the_voice_track_is_found_in_the_mix_and_its_offset_measured(tmp_path):
    rate, seconds = 8000, 6
    clock = np.arange(rate * seconds) / rate
    voice = np.zeros(rate * seconds)
    for start, length in ((0.5, 0.4), (1.3, 0.7), (2.9, 0.3), (3.6, 0.9), (5.0, 0.5)):
        voice[int(start * rate) : int((start + length) * rate)] = 0.3
    voice *= np.sin(2 * np.pi * 300 * clock)
    music = 0.05 * np.sin(2 * np.pi * 110 * clock)
    sf.write(str(tmp_path / "voice.wav"), voice, rate)
    sf.write(str(tmp_path / "on_time.wav"), voice + music, rate)
    sf.write(str(tmp_path / "late.wav"), np.concatenate([np.zeros(rate // 4), voice])[: len(voice)] + music, rate)
    sf.write(str(tmp_path / "other.wav"), music, rate)

    late, found = verify.voice_offset(tmp_path / "on_time.wav", tmp_path / "voice.wav")
    assert late == 0.0 and found > 0.9
    late, found = verify.voice_offset(tmp_path / "late.wav", tmp_path / "voice.wav")
    assert late == pytest.approx(0.25, abs=0.011) and found > 0.9
    assert verify.voice_offset(tmp_path / "other.wav", tmp_path / "voice.wav")[1] < verify.FOUND


def test_voices_spoken_from_an_older_script_are_refused_and_flagged(project, tmp_path, tone):
    write_script(project, "dub", LINES)
    voices.generate(project, "dub", synth=synth)
    project.set_rights("own")
    folder, bed = voices.folder(project, "dub"), tone(tmp_path / "bed.wav", 220, 3)
    record = remixes.audio_remix(project, "dub", bed, voice=folder / "voice.wav")
    assert record["script"] == voices.script_hash(LINES)
    assert verify.script_problems(project, folder, record, LINES) == []

    changed = rewrite(project, "dub", "Good evening.")
    found = [p["problem"] for p in verify.script_problems(project, folder, record, changed)]
    assert any("lines.json changed after this video was made" in p for p in found)
    assert any("voices were spoken from a different version" in p for p in found)
    with pytest.raises(remixes.RemixError, match="different version of lines.json"):
        remixes.audio_remix(project, "dub", bed, voice=folder / "voice.wav")


def test_captions_must_sit_in_their_lines_time_slots(project):
    write_script(project, "dub", LINES)
    voices.write_captions(project, "dub")
    folder = voices.folder(project, "dub")
    assert verify.caption_problems(folder, LINES) == []
    moved = [dict(line, start=line["start"] + 1, end=line["end"] + 1) for line in LINES]
    assert "other times" in verify.caption_problems(folder, moved)[0]["problem"]


def test_restyle_keeps_one_script_behind_voices_captions_and_picture(project, tmp_path, tone):
    write_script(project, "dub", LINES)
    voices.generate(project, "dub", synth=synth)
    project.set_rights("own")
    remixes.audio_remix(project, "dub", tone(tmp_path / "bed.wav", 220, 3, 0.2), voice=voices.folder(project, "dub") / "voice.wav")
    write_script(project, "clay", LINES)
    folder = voices.folder(project, "clay")
    (folder / "look.json").write_text(json.dumps({"style": "Clay.", "characters": {}, "keep": "Keep it."}))
    shots = [{"index": 1, "start": 0.0, "end": 1.5, "duration": 1.5}, {"index": 2, "start": 1.5, "end": 3.0, "duration": 1.5}]
    (project.root / "timeline.json").write_text(json.dumps({"shots": shots}))
    (project.root / "blueprint.json").write_text(
        json.dumps({"characters": [], "shots": [{"index": 1, "description": "a"}, {"index": 2, "description": "b"}]})
    )
    prompts = []

    def edit(video, prompt, resolution, references):
        prompts.append(prompt)
        return video, {}

    record = restyle.full(project, "clay", "dub", edit=edit)
    assert len(prompts) == 2 and "Hello there." in prompts[0] + prompts[1]
    assert record["frames"] == 90 and record["script"] == voices.script_hash(LINES)
    restyle.full(project, "clay", "dub", edit=edit)
    assert len(prompts) == 2  # finished shots are kept while their lines stay the same

    problems, measured = verify.timing_problems(project, folder, record, folder / "output.mp4")
    assert problems == [] and abs(measured["voice_offset_seconds"]) <= 0.02 and measured["frames"]["made"] == 90
    assert verify.script_problems(project, folder, record, LINES) == []
    assert len(verify.frame_sheets(project, folder, folder / "output.mp4", LINES)) == 1

    changed = rewrite(project, "clay", "Good evening.")
    with pytest.raises(restyle.RestyleError, match="different script"):  # the audio still says the old line
        restyle.full(project, "clay", "dub", edit=edit)
    found = [p["problem"] for p in verify.script_problems(project, folder, record, changed)]
    assert any("generated for a different line" in p and "[2]" in p for p in found)

    rewrite(project, "dub", "Good evening.")
    again = restyle.full(project, "clay", "dub", edit=edit)
    assert len(prompts) == 3 and "Good evening." in prompts[2] and again["regenerated_for_script"] == [2]
    assert (folder / "shots" / "0002_restyled_stale.mp4").is_file()
    assert verify.script_problems(project, folder, again, changed) != []  # the voices were never respoken


def test_vocal_cues_reach_the_voice_model_and_stay_out_of_the_captions(project):
    lines = [
        {"id": 1, "start": 0.2, "end": 1.2, "speaker": "host", "text": "Hello there.", "say": "<sigh> Hello, uh, there.", "style": "tired"},
        {"id": 2, "start": 1.6, "end": 2.6, "speaker": "guest", "text": "Good morning."},
    ]
    write_script(project, "dub", lines)
    sent = []

    def speak(text, voice, direction=""):
        sent.append((text, direction))
        return fake_wav(0.5), {"text_tokens": 1, "audio_tokens": 1}

    result = voices.generate(project, "dub", synth=speak)
    assert ("<sigh> Hello, uh, there.", "tired") in sent and ("Good morning.", "calm") in sent
    assert result["with_cues"] == 1 and result["report"][0]["cues"] == ["sigh"]
    voices.write_captions(project, "dub")
    shown = (voices.folder(project, "dub") / "captions.srt").read_text()
    assert "Hello there." in shown and "sigh" not in shown and "uh" not in shown
    plain = [{k: v for k, v in line.items() if k not in ("say", "style")} for line in lines]
    assert voices.script_hash(lines) == voices.script_hash(plain)  # cues change the performance, not the script


def test_say_texts_that_change_the_words_or_use_unknown_cues_are_refused(project):
    base = {"id": 1, "start": 0, "end": 1, "speaker": "host", "text": "Hello there."}
    assert voices.performance_problems([dict(base, say="<chuckle> Hello... <short pause> well, there.")]) == []
    assert "would be read aloud" in voices.performance_problems([dict(base, say="<applause> Hello there.")])[0]
    assert "captions and voice would differ" in voices.performance_problems([dict(base, say="Goodbye there.")])[0]
    assert "leaves out 'there'" in voices.performance_problems([dict(base, say="<sigh> Hello.")])[0]
    write_script(project, "dub", [dict(base, say="Goodbye there.")])
    with pytest.raises(voices.VoiceError, match="cannot be spoken as written"):
        voices.generate(project, "dub", synth=lambda *a: pytest.fail("should not be called"))


def test_a_vocal_cue_read_aloud_is_caught():
    lines = [{"id": 1, "start": 0.2, "end": 1.2, "speaker": "host", "text": "Hello there.", "say": "<sigh> Hello there."}]
    heard = [{"word": " Sigh.", "start": 0.2}, {"word": " Hello", "start": 0.4}, {"word": " there.", "start": 0.7}]
    assert "read a vocal cue aloud" in verify.audio_problems(lines, heard)[0]["problem"]
    assert verify.audio_problems(lines, heard[1:]) == []


def speech_stem(project, bursts, seconds=6.0):
    """A stand-in speech stem: a voice-like tone during each (start, end) burst, silence elsewhere."""
    rate = 8000
    clock = np.arange(int(rate * seconds)) / rate
    wave = np.zeros(len(clock))
    for start, end in bursts:
        wave[int(start * rate) : int(end * rate)] = 0.3
    sf.write(str(project.root / "audio" / "speech.wav"), wave * np.sin(2 * np.pi * 300 * clock), rate)


def test_lines_are_anchored_to_where_the_original_speech_really_is(project):
    speech_stem(project, [(0.5, 1.3), (2.0, 2.4), (2.9, 3.6)])
    # Stamps as a transcriber gives them: each start early, the last two lines close together.
    lines = [
        {"id": 1, "start": 0.2, "end": 1.3, "speaker": "host", "text": "Hello there."},
        {"id": 2, "start": 1.6, "end": 2.45, "speaker": "guest", "text": "Good morning."},
        {"id": 3, "start": 2.5, "end": 3.5, "speaker": "host", "text": "Good day."},
    ]
    found = voices.utterances(project, lines)
    assert [(u["start"], u["end"]) for u in found] == [(0.5, 1.3), (2.0, 2.4), (2.9, 3.6)] and all(u["found"] for u in found)
    write_script(project, "dub", lines)
    moved = voices.anchor(project, "dub")
    assert [m["start"] for m in moved] == [0.3, 0.4, 0.4]
    saved = voices.load_lines(project, "dub")["lines"]
    assert (saved[1]["start"], saved[1]["end"]) == (2.0, 2.4) and voices.anchor(project, "dub") == []  # a second pass changes nothing
    assert voices.utterances(project, [{"id": 9, "start": 4.5, "end": 5.0, "speaker": "host", "text": "x"}])[0]["found"] is False


def test_a_take_that_does_not_fit_is_spoken_again_at_another_pace(project):
    speech_stem(project, [(0.5, 1.5)], seconds=3.0)
    write_script(project, "dub", [{"id": 1, "start": 0.5, "end": 1.5, "speaker": "host", "text": "Hello there."}])
    styles = []

    def speak(text, voice, direction=""):
        styles.append(direction)
        return fake_wav(1.0 if "quickly" in direction else 1.8), {"text_tokens": 1, "audio_tokens": 1}

    result = voices.generate(project, "dub", synth=speak, workers=1)
    row = result["report"][0]
    assert styles == ["", "speaking quickly"] and row["retaken"] and result["retaken"] == 1
    assert row["words_start"] == 0.5 and abs(row["end_error"]) <= 0.1 and 1.0 <= row["speed"] <= voices.MAX_SPEED
    assert (voices.folder(project, "dub") / "takes" / "0001_try1.wav").is_file()  # every paid take is kept
    assert voices.generate(project, "dub", synth=lambda *a: pytest.fail("the chosen take is reused"), workers=1)["respoken"] == 0


def test_a_leading_cue_plays_before_the_line_so_the_first_word_lands_on_the_mouth(project):
    speech_stem(project, [(1.5, 2.3)], seconds=4.0)
    line = {"id": 1, "start": 1.5, "end": 2.3, "speaker": "host", "text": "Hello there.", "say": "<sigh> Hello there."}
    write_script(project, "dub", [line])
    rate = voices.RATE
    tone = lambda seconds: 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * rate)) / rate)
    take = np.concatenate([tone(0.4), np.zeros(int(0.25 * rate)), tone(0.8)]).astype(np.float32)  # a sigh, a pause, the words

    def speak(text, voice, direction=""):
        import io
        buffer = io.BytesIO()
        sf.write(buffer, take, rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), {"text_tokens": 1, "audio_tokens": 1}

    row = voices.generate(project, "dub", synth=speak, workers=1)["report"][0]
    assert row["begin"] == pytest.approx(1.5 - 0.65, abs=0.06) and row["words_start"] == 1.5 and abs(row["end_error"]) < 0.1


def test_a_voice_away_from_the_original_utterance_fails_the_sync_gate(project):
    speech_stem(project, [(0.5, 1.5), (2.5, 3.5)], seconds=4.0)
    lines = [
        {"id": 1, "start": 0.5, "end": 1.5, "speaker": "host", "text": "Hello there."},
        {"id": 2, "start": 2.5, "end": 3.5, "speaker": "guest", "text": "Good morning."},
    ]
    folder = voices.folder(project, "dub")
    folder.mkdir(parents=True)
    rate = 8000
    clock = np.arange(rate * 4) / rate

    def voice(bursts):
        wave = np.zeros(len(clock))
        for start, end in bursts:
            wave[int(start * rate) : int(end * rate)] = 0.3
        sf.write(str(folder / "voice.wav"), wave * np.sin(2 * np.pi * 200 * clock), rate)

    record = {"voices": "new", "mode": "audio"}
    voice([(0.55, 1.45), (2.6, 3.4)])
    problems, measured = verify.sync_problems(project, folder, record, lines)
    assert problems == [] and measured["lines_timed"] == 2 and measured["worst_start_error"] == pytest.approx(0.1, abs=0.02)
    voice([(0.0, 0.9), (2.5, 4.0)])  # the first line half a second early, the second running on after the mouth stops
    found = [(p["line"], p["problem"]) for p in verify.sync_problems(project, folder, record, lines)[0]]
    assert any(n == 1 and "starts -0.50 s" in text for n, text in found) and any(n == 2 and "stops +0.50 s" in text for n, text in found)
    assert verify.sync_problems(project, folder, {"voices": "original", "mode": "audio"}, lines) == ([], {})


def test_each_shot_prompt_names_the_speaker_the_seconds_and_who_stays_silent():
    look = {"style": "Clay.", "characters": {"dave": "Dave is thirty.", "mantis": "A green grasshopper."},
            "names": {"dave": "Dave", "mantis": "the grasshopper"}, "keep": "Keep it."}
    shot = {"index": 4, "start": 10.0, "end": 14.0, "description": "Two-shot at the desk."}
    lines = [
        {"id": 1, "start": 9.0, "end": 11.0, "speaker": "mantis", "text": "Where does that road end?"},
        {"id": 2, "start": 12.0, "end": 13.5, "speaker": "dave", "text": "Nowhere."},
        {"id": 3, "start": 13.95, "end": 15.0, "speaker": "mantis", "text": "Too short to count."},
    ]
    heard = restyle.heard_in(shot, lines)
    assert [(h["speaker"], h["from"], h["to"]) for h in heard] == [("mantis", 0.0, 1.0), ("dave", 2.0, 3.5)]  # 0.05 s of line 3 does not count
    both = restyle.shot_prompt(look, shot, ["dave", "mantis"], heard, ("dave", "mantis"))
    assert 'From 0.0 to 1.0 seconds the grasshopper says: "Where does that road end?"' in both
    assert 'From 2.0 to 3.5 seconds Dave says: "Nowhere."' in both and "Only the grasshopper and Dave moves a mouth" in both
    assert "Dave is clearly TALKING" in both and "never holds a fixed smile" in both  # a speaker must visibly talk for the whole line
    only_dave = restyle.shot_prompt(look, shot, ["dave"], heard[:1], ("dave", "mantis"))
    assert "the grasshopper is heard from off screen" in only_dave and "Nobody on screen speaks" in only_dave
    assert "Dave does not speak in this shot and only listens, mouth closed." in only_dave
    assert restyle.heard_in(shot, lines, stretch=2.0)[1]["from"] == 4.0  # a short shot slowed for the model


def test_the_dialogue_gate_stays_closed_until_script_and_picture_agree(project):
    speech_stem(project, [(0.5, 1.3), (2.0, 2.8)], seconds=4.0)
    stamped = [
        {"id": 1, "start": 0.2, "end": 1.3, "speaker": "host", "text": "Hello there."},
        {"id": 2, "start": 1.6, "end": 2.8, "speaker": "guest", "text": "Good morning."},
    ]
    write_script(project, "dub", stamped)
    shots = [{"index": 1, "start": 0.0, "end": 1.6, "duration": 1.6}, {"index": 2, "start": 1.6, "end": 4.0, "duration": 2.4}]
    (project.root / "timeline.json").write_text(json.dumps({"shots": shots}))
    (project.root / "blueprint.json").write_text(json.dumps({"characters": [{"id": "host", "shots": [1]}], "shots": []}))

    closed = dialogue.problems(project, "dub")
    assert any(p.startswith("script:") and "--anchor" in p for p in closed)  # slots still on transcript stamps
    assert any(p.startswith("picture:") and "[2]" in p for p in closed)  # nobody has said who shot 2 shows
    voices.anchor(project, "dub")
    assert [p for p in dialogue.problems(project, "dub") if p.startswith("script:")] == []
    (project.root / "blueprint.json").write_text(json.dumps({"characters": [{"id": "host", "shots": [1, 2]}], "shots": []}))
    assert dialogue.problems(project, "dub") == []

    shot = dialogue.plan(project, "dub")["shots"][1]
    assert shot["on_screen"] == ["host"] and shot["off_screen"] == ["guest"] and shot["silent"] == ["host"]  # the guest is heard over the host
    assert "| 2 |" in dialogue.render(project, "dub").read_text()
    (project.root / "blueprint.json").write_text(json.dumps({"characters": [{"id": "host", "shots": [1]}], "no_one_on_screen": [2], "shots": []}))
    assert dialogue.problems(project, "dub") == [] and dialogue.plan(project, "dub")["shots"][1]["on_screen"] == []


def test_the_mouth_check_holds_a_clip_against_who_is_heard_when():
    want = {"watch": ["dave", "mantis"], "heard": [{"speaker": "mantis", "from": 1.0, "to": 4.0}]}
    good = {"characters": [{"id": "mantis", "mouth_visible": True, "mouth_moving": [[1.1, 3.9]]},
                           {"id": "dave", "mouth_visible": True, "mouth_moving": []}]}
    assert mouths.judge(want, good) == []
    smiling = {"characters": [{"id": "mantis", "mouth_visible": True, "mouth_moving": [[1.0, 1.8]]},
                              {"id": "dave", "mouth_visible": True, "mouth_moving": []}]}
    assert "moves for only 0.8 s" in mouths.judge(want, smiling)[0]
    wrong = {"characters": [{"id": "mantis", "mouth_visible": True, "mouth_moving": []},
                            {"id": "dave", "mouth_visible": True, "mouth_moving": [[0.5, 4.0]]}]}
    found = mouths.judge(want, wrong)
    assert any("mantis speaks" in p for p in found) and any("dave's mouth moves for 3.0 s while mantis is the one heard" in p for p in found)
    assert "cuts to a different scene at 2.9 s" in mouths.judge(want, {**good, "scene_changes": [2.9]})[0]
    assert mouths.judge(want, {**good, "scene_changes": [0.1]}) == []
    assert mouths.judge(want, {**good, "scene_changes": [3.1]}, length=3.2) == [] and mouths.trim_point({"scene_changes": [3.1]}, 3.2) == 3.0
    assert mouths.trim_point({"scene_changes": [1.5]}, 3.2) is None and mouths.trim_point({}, 3.2) is None
    hidden = {"characters": [{"id": "dave", "mouth_visible": False, "mouth_moving": []}, {"id": "mantis", "mouth_visible": True, "mouth_moving": [[1, 4]]}]}
    assert mouths.judge(want, hidden) == []  # a mouth that cannot be seen is not held against the shot
    short = {"watch": ["dave"], "heard": [{"speaker": "dave", "from": 2.0, "to": 2.4}]}
    assert mouths.judge(short, {"characters": [{"id": "dave", "mouth_visible": True, "mouth_moving": [[2.1, 2.3]]}]}) == []


def test_a_shot_that_fails_the_mouth_check_is_retaken_and_the_best_take_is_kept(project, tmp_path, tone):
    speech_stem(project, [(0.5, 1.5), (2.0, 2.8)], seconds=3.0)
    write_script(project, "dub", [
        {"id": 1, "start": 0.5, "end": 1.5, "speaker": "host", "text": "Hello there."},
        {"id": 2, "start": 2.0, "end": 2.8, "speaker": "guest", "text": "Good morning."},
    ])
    voices.generate(project, "dub", synth=synth, workers=1)
    project.set_rights("own")
    remixes.audio_remix(project, "dub", tone(tmp_path / "bed.wav", 220, 3, 0.2), voice=voices.folder(project, "dub") / "voice.wav")
    write_script(project, "clay", [
        {"id": 1, "start": 0.5, "end": 1.5, "speaker": "host", "text": "Hello there."},
        {"id": 2, "start": 2.0, "end": 2.8, "speaker": "guest", "text": "Good morning."},
    ])
    folder = voices.folder(project, "clay")
    (folder / "look.json").write_text(json.dumps({"style": "Clay.", "characters": {}, "keep": "Keep it.", "audio_from": "dub", "names": {}}))
    (project.root / "timeline.json").write_text(json.dumps({"shots": [{"index": 1, "start": 0.0, "end": 3.0, "duration": 3.0}]}))
    (project.root / "blueprint.json").write_text(json.dumps({"characters": [{"id": "host", "shots": [1]}, {"id": "guest", "shots": [1]}],
                                                              "shots": [{"index": 1, "description": "a"}]}))
    asked, answers = [], iter([
        {"characters": [{"id": "host", "mouth_visible": True, "mouth_moving": []}, {"id": "guest", "mouth_visible": True, "mouth_moving": []}]},
        {"characters": [{"id": "host", "mouth_visible": True, "mouth_moving": [[0.5, 1.5]]}, {"id": "guest", "mouth_visible": True, "mouth_moving": [[2.0, 2.8]]}]},
    ])

    def edit(video, prompt, resolution, references):
        asked.append(prompt)
        return video + bytes([len(asked)]), {}  # every take is a different file, as real ones are

    def watch(clip, text, schema):
        return next(answers), {"input_tokens": 1000, "output_tokens": 50}

    record = restyle.full(project, "clay", edit=edit, mouth_check=True, watch=watch)
    assert len(asked) == 2 and "The last attempt at this shot was wrong" in asked[1] and "host speaks from" in asked[1]
    assert record["mouth_failed"] == {} and mouths.problems(project, "clay") == []
    saved = json.loads((folder / "shots" / "0001_mouths.json").read_text())
    assert saved["ok"] and saved["usd"] > 0
    assert len(restyle.pending_shots(project, "clay")) == 0  # the check does not make the shot outdated
