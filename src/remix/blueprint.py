"""The blueprint: a structured description of what a video is and how it is built.

`init` writes a skeleton holding everything that can be measured; Claude fills in the rest after
reading the contact sheets; `check` validates the result and renders blueprint.md.
"""
from __future__ import annotations

import json
import statistics
from math import gcd

from .media import read_pcm_f32
from .project import Project
from .steps.shots import fmt_time

SCHEMA = 1
ROLES = (
    "hook", "title", "setup", "point", "proof", "demo", "b-roll", "transition", "payoff", "cta", "outro", "other",
)
COMMON_ASPECTS = {
    (16, 9): "16:9", (9, 16): "9:16", (1, 1): "1:1", (4, 5): "4:5", (5, 4): "5:4", (4, 3): "4:3", (3, 4): "3:4",
    (21, 9): "21:9",
}
VISUAL_KEYS = ("look", "palette", "lighting", "camera", "editing", "on_screen_text")
AUDIO_KEYS = ("voice", "music", "sound_design")


class BlueprintError(RuntimeError):
    pass


def aspect_name(width: int, height: int) -> str:
    ratio = width / height
    for (w, h), name in COMMON_ASPECTS.items():
        if abs(ratio - w / h) / (w / h) < 0.02:
            return name
    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def spoken_seconds(words: list[dict], bridge: float = 0.35) -> float:
    """Total time someone is talking, counting pauses shorter than `bridge` as speech."""
    total, start, end = 0.0, None, 0.0
    for word in words:
        if start is None:
            start, end = word["start"], word["end"]
        elif word["start"] - end <= bridge:
            end = max(end, word["end"])
        else:
            total += end - start
            start, end = word["start"], word["end"]
    if start is not None:
        total += end - start
    return total


def _load(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _timeline(project: Project) -> dict:
    path = project.root / "timeline.json"
    if not path.is_file():
        raise BlueprintError("timeline.json is missing: run `remix decompose` on this project first")
    return _load(path)


def compute_metrics(project: Project) -> dict:
    """Everything about pacing, speech and background audio that can be measured without judgment."""
    import numpy as np

    from . import audio_features

    root = project.root
    info = project.load()["source"]
    duration = info["duration"]
    lengths = [shot["duration"] for shot in _timeline(project)["shots"]]
    metrics: dict = {
        "duration": duration,
        "aspect": aspect_name(info["width"], info["height"]),
        "shot_count": len(lengths),
        "cuts_per_minute": round((len(lengths) - 1) / duration * 60, 1) if duration else None,
        "shot_seconds": {
            "median": round(statistics.median(lengths), 2),
            "shortest": min(lengths),
            "longest": max(lengths),
        },
    }

    transcript_path = root / "transcript" / "transcript.json"
    if transcript_path.is_file():
        words = [w for seg in _load(transcript_path)["segments"] for w in seg["words"]]
        talking = spoken_seconds(words)
        count = sum(1 for w in words if not w.get("glue"))
        metrics["speech"] = {
            "words": count,
            "share_of_runtime": round(talking / duration, 2) if duration else None,
            "words_per_minute": round(count / talking * 60) if talking > 0 else None,
            "first_word_at": words[0]["start"] if words else None,
        }

    background = root / "audio" / "background.wav"
    if background.is_file():
        rate = audio_features.ANALYSIS_RATE
        samples = np.frombuffer(read_pcm_f32(background, rate), dtype=np.float32)
        metrics["background_audio"] = {
            "share_of_runtime": round(audio_features.active_ratio(samples, rate), 2),
            "tempo": audio_features.estimate_tempo(samples, rate),
        }
    return metrics


def init(project: Project, force: bool = False) -> dict:
    """Write blueprint.json with measurements and shot timings filled in and the rest left empty."""
    path = project.root / "blueprint.json"
    if path.is_file() and not force:
        raise BlueprintError("blueprint.json already exists; pass --force to start over")
    blueprint = {
        "schema": SCHEMA,
        "slug": project.slug,
        "summary": "",
        "format": "",
        "audience": "",
        "themes": [],
        "tone": [],
        "structure": [],
        "visual_style": {key: [] if key == "palette" else "" for key in VISUAL_KEYS},
        "audio_style": {key: "" for key in AUDIO_KEYS},
        "characters": [],
        "shots": [
            {
                "index": shot["index"],
                "start": shot["start"],
                "end": shot["end"],
                "spoken": shot["text"],
                "speaker": "",
                "role": "",
                "description": "",
                "camera": "",
                "on_screen_text": [],
            }
            for shot in _timeline(project)["shots"]
        ],
        "remix_notes": {"essential": [], "swappable": [], "risks": []},
        "metrics": compute_metrics(project),
    }
    path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")
    return blueprint


def _empty(value) -> bool:
    return not value or (isinstance(value, str) and not value.strip())


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def problems(blueprint: dict, timeline: dict, duration: float) -> list[str]:
    """Every reason this blueprint is not finished yet. An empty list means it is complete."""
    found: list[str] = []

    def need(value, label: str) -> None:
        if _empty(value):
            found.append(f"{label} is empty")

    for key in ("summary", "format", "audience", "themes", "tone", "structure"):
        need(blueprint.get(key), key)
    for key in VISUAL_KEYS:
        need((blueprint.get("visual_style") or {}).get(key), f"visual_style.{key}")
    for key in AUDIO_KEYS:
        need((blueprint.get("audio_style") or {}).get(key), f"audio_style.{key}")
    notes = blueprint.get("remix_notes") or {}
    need(notes.get("essential"), "remix_notes.essential")
    need(notes.get("swappable"), "remix_notes.swappable")
    if not isinstance(notes.get("risks"), list):
        found.append("remix_notes.risks must be a list (empty when there are none)")

    previous_end = 0.0
    for i, beat in enumerate(blueprint.get("structure") or [], 1):
        label = f"structure[{i}]"
        if not isinstance(beat, dict):
            found.append(f"{label} must be an object with beat, start, end, what")
            continue
        need(beat.get("beat"), f"{label}.beat")
        need(beat.get("what"), f"{label}.what")
        start, end = beat.get("start"), beat.get("end")
        if not (_number(start) and _number(end) and 0 <= start < end <= duration + 0.5):
            found.append(f"{label} needs 0 <= start < end <= {duration}")
        elif start < previous_end - 0.01:
            found.append(f"{label} starts before the previous beat ends")
        else:
            previous_end = end

    characters = blueprint.get("characters") or []
    character_ids = {c.get("id") for c in characters if isinstance(c, dict)}
    expected = timeline["shots"]
    shots = blueprint.get("shots") or []
    if [s.get("index") if isinstance(s, dict) else None for s in shots] != [s["index"] for s in expected]:
        found.append(f"shots must list indexes 1..{len(expected)} in order, exactly as in timeline.json")
    else:
        for shot in shots:
            label = f"shot {shot['index']}"
            need(shot.get("description"), f"{label} description")
            if shot.get("role") not in ROLES:
                found.append(f"{label} role must be one of: {', '.join(ROLES)}")
            if not isinstance(shot.get("on_screen_text"), list):
                found.append(f"{label} on_screen_text must be a list (empty when there is none)")
            if shot.get("speaker") and shot["speaker"] not in character_ids:
                found.append(f"{label} speaker '{shot['speaker']}' is not a character id")

    known = {s["index"] for s in expected}
    for i, character in enumerate(characters, 1):
        label = f"characters[{i}]"
        if not isinstance(character, dict):
            found.append(f"{label} must be an object with id, description, shots")
            continue
        need(character.get("id"), f"{label}.id")
        need(character.get("description"), f"{label}.description")
        if not set(character.get("shots") or []) <= known:
            found.append(f"{label}.shots lists a shot that does not exist")
    return found


def normalize(blueprint: dict, timeline: dict) -> None:
    """Restore the shot fields that code owns: timings and spoken words always come from the timeline.

    This lets a blueprint be written with just `index` plus the authored fields for each shot.
    """
    measured = ("index", "start", "end", "spoken")
    blueprint["shots"] = [
        {
            "index": reference["index"],
            "start": reference["start"],
            "end": reference["end"],
            "spoken": reference["text"],
            **{key: value for key, value in shot.items() if key not in measured},
        }
        for shot, reference in zip(blueprint["shots"], timeline["shots"])
    ]


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ") or "-"


def render(blueprint: dict) -> str:
    """blueprint.md: the same content laid out for a person to read."""
    visual, audio, notes = blueprint["visual_style"], blueprint["audio_style"], blueprint["remix_notes"]
    metrics = blueprint["metrics"]
    lines = [
        f"# Blueprint: {blueprint['slug']}",
        "",
        blueprint["summary"],
        "",
        f"- **Format:** {blueprint['format']}",
        f"- **Audience:** {blueprint['audience']}",
        f"- **Themes:** {', '.join(blueprint['themes'])}",
        f"- **Tone:** {', '.join(blueprint['tone'])}",
        "",
        "## Structure",
        "",
        "| Beat | Time | What happens |",
        "|---|---|---|",
    ]
    for beat in blueprint["structure"]:
        lines.append(
            f"| {_cell(beat['beat'])} | {fmt_time(beat['start'])}-{fmt_time(beat['end'])} | {_cell(beat['what'])} |"
        )

    lines += ["", "## Visual style", ""]
    for key in VISUAL_KEYS:
        value = ", ".join(visual[key]) if isinstance(visual[key], list) else visual[key]
        lines.append(f"- **{key.replace('_', ' ').capitalize()}:** {value}")
    lines += ["", "## Audio style", ""]
    for key in AUDIO_KEYS:
        lines.append(f"- **{key.replace('_', ' ').capitalize()}:** {audio[key]}")

    if blueprint.get("characters"):
        lines += ["", "## Characters", ""]
        for character in blueprint["characters"]:
            shots = ", ".join(str(n) for n in character.get("shots") or []) or "-"
            lines.append(f"- **{character['id']}** (shots {shots}): {character['description']}")

    lines += [
        "",
        "## Shots",
        "",
        "| # | Time | Role | What is on screen | Camera | On-screen text | Speaker | Spoken |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for shot in blueprint["shots"]:
        lines.append(
            f"| {shot['index']} | {fmt_time(shot['start'])}-{fmt_time(shot['end'])} | {shot['role']} "
            f"| {_cell(shot['description'])} | {_cell(shot.get('camera', ''))} "
            f"| {_cell(' / '.join(shot['on_screen_text']))} | {_cell(shot.get('speaker', ''))} "
            f"| {_cell(shot.get('spoken', ''))} |"
        )

    lines += ["", "## Remix notes", ""]
    for key, title in (("essential", "Keep"), ("swappable", "Free to change"), ("risks", "Risks")):
        lines.append(f"**{title}**")
        lines += [f"- {item}" for item in notes[key]] or ["- none noted"]
        lines.append("")

    lines += ["## Measured", ""]
    lengths = metrics["shot_seconds"]
    lines.append(
        f"- {fmt_time(metrics['duration'])} long, {metrics['aspect']}, {metrics['shot_count']} shots, "
        f"{metrics['cuts_per_minute']} cuts per minute "
        f"(shots {lengths['shortest']}-{lengths['longest']} s, median {lengths['median']} s)"
    )
    speech = metrics.get("speech")
    if speech:
        lines.append(
            f"- Speech: {speech['words']} words, talking for {speech['share_of_runtime']:.0%} of the runtime, "
            f"{speech['words_per_minute']} words per minute while talking"
        )
    background = metrics.get("background_audio")
    if background:
        tempo = background["tempo"]
        beat = f"about {tempo['bpm']} BPM (estimate)" if tempo else "no steady beat found"
        lines.append(f"- Background audio: present for {background['share_of_runtime']:.0%} of the runtime, {beat}")
    return "\n".join(lines) + "\n"


def check(project: Project) -> list[str]:
    """Validate blueprint.json. When it is complete, refresh its metrics and render blueprint.md."""
    path = project.root / "blueprint.json"
    if not path.is_file():
        raise BlueprintError("blueprint.json is missing: run `remix blueprint init` first")
    try:
        blueprint = _load(path)
    except json.JSONDecodeError as exc:
        return [f"blueprint.json is not valid JSON: {exc}"]

    timeline = _timeline(project)
    found = problems(blueprint, timeline, project.load()["source"]["duration"])
    if found:
        return found

    normalize(blueprint, timeline)
    blueprint["metrics"] = compute_metrics(project)
    path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = project.root / "blueprint.md"
    markdown.write_text(render(blueprint), encoding="utf-8")
    project.record("blueprint", "done", outputs=[project.rel(path), project.rel(markdown)])
    return []
