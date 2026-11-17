import numpy as np
import pytest

from remix import blueprint
from remix.audio_features import ANALYSIS_RATE, active_ratio, estimate_tempo

TIMELINE = {
    "shots": [
        {"index": 1, "start": 0.0, "end": 2.0, "duration": 2.0, "text": "Hi."},
        {"index": 2, "start": 2.0, "end": 5.0, "duration": 3.0, "text": ""},
    ]
}


def filled() -> dict:
    return {
        "schema": 1,
        "slug": "demo",
        "summary": "A demo.",
        "format": "short",
        "audience": "everyone",
        "themes": ["testing"],
        "tone": ["plain"],
        "structure": [
            {"beat": "hook", "start": 0.0, "end": 2.0, "what": "Greeting"},
            {"beat": "payoff", "start": 2.0, "end": 5.0, "what": "Bars"},
        ],
        "visual_style": {
            "look": "test patterns",
            "palette": ["red"],
            "lighting": "flat",
            "camera": "static",
            "editing": "hard cuts",
            "on_screen_text": "none",
        },
        "audio_style": {"voice": "one narrator", "music": "not listened to", "sound_design": "none"},
        "characters": [{"id": "host", "description": "narrator, never on screen", "shots": [1]}],
        "shots": [
            {
                "index": 1, "start": 0.0, "end": 2.0, "spoken": "Hi.", "speaker": "host", "role": "hook",
                "description": "Color bars.", "camera": "static", "on_screen_text": [],
            },
            {
                "index": 2, "start": 2.0, "end": 5.0, "spoken": "", "speaker": "", "role": "payoff",
                "description": "More bars.", "camera": "static", "on_screen_text": ["END"],
            },
        ],
        "remix_notes": {"essential": ["the greeting"], "swappable": ["the bars"], "risks": []},
        "metrics": {
            "duration": 5.0,
            "aspect": "16:9",
            "shot_count": 2,
            "cuts_per_minute": 12.0,
            "shot_seconds": {"median": 2.5, "shortest": 2.0, "longest": 3.0},
        },
    }


def click_track(bpm: float, seconds: float = 24.0) -> np.ndarray:
    rate = ANALYSIS_RATE
    samples = np.zeros(int(rate * seconds), dtype=np.float32)
    ticks = np.arange(int(rate * 0.03))
    burst = (np.sin(2 * np.pi * 1000 * ticks / rate) * np.exp(-ticks / (rate * 0.008))).astype(np.float32)
    time = 0.0
    while time + 0.05 < seconds:
        start = int(time * rate)
        samples[start : start + len(burst)] += burst
        time += 60 / bpm
    return samples


@pytest.mark.parametrize("bpm", [90, 120, 140])
def test_estimate_tempo_finds_a_click_track(bpm):
    found = estimate_tempo(click_track(bpm))
    assert found is not None and abs(found["bpm"] - bpm) < 2


def test_estimate_tempo_returns_none_without_a_beat():
    assert estimate_tempo(np.zeros(ANALYSIS_RATE * 20, dtype=np.float32)) is None


def test_active_ratio_counts_windows_above_the_floor():
    rate = 1000
    loud = np.full(rate * 2, 0.1, dtype=np.float32)
    quiet = np.zeros(rate * 2, dtype=np.float32)
    assert active_ratio(np.concatenate([loud, quiet]), rate) == 0.5


def test_aspect_names_and_spoken_seconds():
    assert blueprint.aspect_name(1080, 1920) == "9:16"
    assert blueprint.aspect_name(1280, 720) == "16:9"
    assert blueprint.aspect_name(1000, 300) == "10:3"
    words = [{"start": 0.0, "end": 0.5}, {"start": 0.6, "end": 1.0}, {"start": 3.0, "end": 3.5}]
    assert blueprint.spoken_seconds(words) == pytest.approx(1.5)
    assert blueprint.spoken_seconds([]) == 0.0


def test_complete_blueprint_has_no_problems():
    assert blueprint.problems(filled(), TIMELINE, 5.0) == []


def test_problems_name_what_is_wrong():
    broken = filled()
    broken["summary"] = " "
    broken["shots"][1]["role"] = "finale"
    broken["structure"][1]["start"] = 1.0
    broken["characters"][0]["shots"] = [9]
    broken["shots"][1]["speaker"] = "stranger"
    found = blueprint.problems(broken, TIMELINE, 5.0)
    assert "shot 2 speaker 'stranger' is not a character id" in found
    assert "summary is empty" in found
    assert any(p.startswith("shot 2 role must be one of") for p in found)
    assert "structure[2] starts before the previous beat ends" in found
    assert "characters[1].shots lists a shot that does not exist" in found


def test_normalize_restores_the_measured_shot_fields():
    edited = filled()
    edited["shots"][0] = {"index": 1, "start": 9.9, "role": "hook", "description": "Color bars.", "on_screen_text": []}
    blueprint.normalize(edited, TIMELINE)
    assert edited["shots"][0] == {
        "index": 1, "start": 0.0, "end": 2.0, "spoken": "Hi.",
        "role": "hook", "description": "Color bars.", "on_screen_text": [],
    }


def test_shot_list_must_match_the_timeline():
    short = filled()
    short["shots"].pop()
    assert blueprint.problems(short, TIMELINE, 5.0) == [
        "shots must list indexes 1..2 in order, exactly as in timeline.json"
    ]


def test_render_lists_every_shot_and_the_measurements():
    text = blueprint.render(filled())
    assert "| 1 | 0:00.0-0:02.0 | hook | Color bars. | static | - | host | Hi. |" in text
    assert "| 2 | 0:02.0-0:05.0 | payoff | More bars. | static | END | - | - |" in text
    assert "2 shots, 12.0 cuts per minute" in text
    assert "- **host** (shots 1): narrator, never on screen" in text
