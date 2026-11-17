"""The script check: audio, picture and captions against a remix's lines."""
import json

from remix import verify, voices

LINES = [
    {"id": 1, "start": 0.2, "end": 1.2, "speaker": "host", "text": "Hello there, Dave."},
    {"id": 2, "start": 1.6, "end": 2.6, "speaker": "guest", "text": "Touché."},
]
HEARD = [{"word": " Hello", "start": 0.25}, {"word": " there,", "start": 0.5}, {"word": " Dave.", "start": 0.8},
         {"word": " Touche.", "start": 1.65}]


def make_remix(project, lines, mode="audio"):
    folder = voices.folder(project, "take")
    folder.mkdir(parents=True)
    (folder / "lines.json").write_text(json.dumps({"voices": {"host": "Charon", "guest": "Kore"}, "directions": {}, "lines": lines}))
    (folder / "remix.json").write_text(json.dumps({"mode": mode, "voices": "new", "output": "source.mp4"}))
    (project.root / "audio" / "speech.wav").unlink(missing_ok=True)  # a test tone is not speech to anchor lines to
    (project.root / "transcript").mkdir(exist_ok=True)
    (project.root / "transcript" / "transcript.json").write_text(json.dumps({"text": "Hello there, Dave. Touche.", "segments": []}))
    return folder


def test_a_matching_remix_passes_and_only_a_fresh_check_counts(project):
    folder = make_remix(project, LINES)
    result = verify.run(project, "take", listen=lambda path: HEARD)
    assert result["passed"] and result["lines"] == 2 and verify.passed(folder)
    (folder / "remix.json").write_text((folder / "remix.json").read_text() + " ")  # the video was remade
    import os, time
    os.utime(folder / "remix.json", (time.time() + 5, time.time() + 5))
    assert not verify.passed(folder)


def test_rewritten_lines_fail_on_audio_picture_and_captions(project):
    changed = [dict(LINES[0], text="Goodbye forever, Dave."), LINES[1]]
    folder = make_remix(project, changed)
    (folder / "captions.srt").write_text("1\n00:00:00,200 --> 00:00:01,200\nHello there, Dave.\n")
    result = verify.run(project, "take", listen=lambda path: HEARD)
    problems = [p["problem"] for p in result["problems"]]
    assert not result["passed"] and not verify.passed(folder)
    assert "the audio does not say this line" in problems[0]
    assert any("picture is the original, but 1 line(s) were rewritten" in p for p in problems)
    assert any("captions.srt does not match" in p for p in problems)


def test_late_lines_and_unread_frames_fail(project):
    make_remix(project, LINES, mode="restyle")
    late = [dict(HEARD[0], start=1.3), dict(HEARD[1], start=1.4), dict(HEARD[2], start=1.5), HEARD[3]]
    problems = [p["problem"] for p in verify.run(project, "take", listen=lambda path: late)["problems"]]
    assert any("off its slot" in p for p in problems) and any("frames have not been read" in p for p in problems)
    assert verify.run(project, "take", frames_ok=True, listen=lambda path: HEARD)["passed"]
