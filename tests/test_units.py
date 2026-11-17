import numpy as np
import pytest

from remix.media import parse_rate
from remix.overview import spoken_in
from remix.project import slugify
from remix.steps.shots import fmt_time, keyframe_times, sheet_layout
from remix.steps.transcript import (
    SPEECH_FLOOR_DBFS,
    caption_cues,
    format_srt_time,
    join_words,
    stem_level,
    to_script,
    to_srt,
)


def test_parse_rate():
    assert parse_rate("30/1") == 30.0
    assert abs(parse_rate("30000/1001") - 29.97) < 0.01
    assert parse_rate("0/0") is None
    assert parse_rate(None) is None


def test_slugify():
    assert slugify("My Video (final) v2!") == "my-video-final-v2"
    assert slugify("???") == "video"
    assert len(slugify("x" * 100)) == 40


def test_keyframe_times_scale_with_shot_length():
    assert keyframe_times(0, 3) == [1.5]
    assert keyframe_times(10, 22) == [13.0, 19.0]
    assert len(keyframe_times(0, 600, max_frames=8)) == 8
    assert keyframe_times(5, 5) == [5.0]


def test_fmt_time():
    assert fmt_time(0) == "0:00.0"
    assert fmt_time(72.44) == "1:12.4"
    assert fmt_time(59.96) == "1:00.0"


def test_sheet_layout_by_orientation():
    assert sheet_layout(1920, 1080) == (4, 480, 270)
    assert sheet_layout(1080, 1920) == (6, 320, 569)


def test_srt_formatting():
    assert format_srt_time(3661.5) == "01:01:01,500"
    srt = to_srt([{"start": 0.5, "end": 2.0, "text": "Hello there."}])
    assert srt == "1\n00:00:00,500 --> 00:00:02,000\nHello there.\n"


def test_caption_cues_break_on_sentences_pauses_and_length():
    words = [
        {"start": 0.0, "end": 0.4, "word": "Hello"},
        {"start": 0.4, "end": 0.8, "word": "there."},
        {"start": 0.9, "end": 1.2, "word": "Next"},
        {"start": 3.0, "end": 3.4, "word": "pause"},
    ]
    cues = caption_cues(words)
    assert [cue["text"] for cue in cues] == ["Hello there.", "Next", "pause"]
    assert (cues[0]["start"], cues[0]["end"]) == (0.0, 0.8)

    steady = [{"start": float(i), "end": i + 0.5, "word": "wordy"} for i in range(10)]
    assert [len(cue["text"]) for cue in caption_cues(steady)] == [41, 17]
    assert caption_cues([]) == []


def test_glued_words_attach_to_the_previous_word():
    words = [
        {"start": 0.0, "end": 0.3, "word": "self"},
        {"start": 2.0, "end": 2.4, "word": "-test.", "glue": True},
        {"start": 2.5, "end": 2.8, "word": "Done"},
    ]
    assert join_words(words) == "self-test. Done"
    # A glued word never starts a new cue, even after a long gap.
    assert [cue["text"] for cue in caption_cues(words)] == ["self-test.", "Done"]


def test_stem_level_measures_the_requested_span():
    rate = 100
    samples = np.concatenate([np.full(rate, 0.1, dtype=np.float32), np.zeros(rate, dtype=np.float32)])
    assert stem_level(samples, 0.0, 1.0, rate) == pytest.approx(-20.0, abs=0.01)
    assert stem_level(samples, 1.0, 2.0, rate) < SPEECH_FLOOR_DBFS
    assert stem_level(samples, 5.0, 6.0, rate) == -120.0  # past the end of the audio


def test_script_breaks_paragraphs_on_long_pauses():
    segments = [
        {"start": 0, "end": 2, "text": "One."},
        {"start": 2.2, "end": 4, "text": "Two."},
        {"start": 7, "end": 8, "text": "Three."},
    ]
    assert to_script(segments) == "One. Two.\n\nThree.\n"
    assert to_script([]) == ""


def test_spoken_in_assigns_words_by_midpoint():
    words = [
        {"start": 0.0, "end": 0.4, "word": "a"},
        {"start": 0.9, "end": 1.3, "word": "b"},
        {"start": 1.4, "end": 1.8, "word": "c"},
    ]
    assert spoken_in(0, 1, words) == "a"
    assert spoken_in(1, 2, words) == "b c"
