"""Caption line width: wide renders hold longer lines, and the check expects exactly what the render wrote."""
from remix import verify, voices
from remix.steps.transcript import to_srt

LINE = {"id": 1, "start": 1.0, "end": 5.0, "speaker": "narrator", "text": "Replace, hold, or remarket. With numbers to back the call."}  # 58 characters


def test_wide_frames_hold_longer_caption_lines():
    assert voices.caption_chars((1920, 1080)) == 60
    assert voices.caption_chars((1080, 1080)) == 42 and voices.caption_chars((1080, 1920)) == 42
    assert voices.caption_chars(None) == 42  # captions for a source video keep the old width, so earlier ones stay valid
    assert len(voices.split_cue(1.0, 5.0, LINE["text"], 42)) == 2  # a stub cue: "call."
    assert len(voices.split_cue(1.0, 5.0, LINE["text"], voices.caption_chars((1920, 1080)))) == 1


def test_the_check_expects_what_the_render_wrote(tmp_path):
    wide = voices.split_cue(LINE["start"], LINE["end"], LINE["text"], voices.caption_chars((1920, 1080)))
    (tmp_path / "captions_wide.srt").write_text(to_srt(wide), encoding="utf-8")
    assert verify.caption_problems(tmp_path, [LINE], "captions_wide.srt", (1920, 1080)) == []
    # the same file judged as if it were made for another frame is caught, not silently accepted
    assert verify.caption_problems(tmp_path, [LINE], "captions_wide.srt", (1080, 1080))
