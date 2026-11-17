"""The dialogue gate: one map of who says what, when, over which shot, checked before anything is paid for.

Script, voice and picture go wrong together when they disagree about the dialogue: a line timed
from transcript stamps instead of the real speech, a voice that outlasts the mouth, a shot whose
prompt never said who is talking. This module builds the single map all three are made from and
refuses to let a run go ahead while the map has holes:

- script:  every line has a speaker with a voice, sits on its original utterance, and overlaps no other line
- voice:   the voices were spoken from this script and each line lands on its utterance
- picture: for every shot a line is heard over, someone has stated who is on screen (possibly nobody),
           so the shot's prompt can name the speaker and tell everyone else to keep still

`remix voices` checks the script stage, `remix restyle` all three, and `remix verify` the script
and picture stages again on the finished video.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import voices
from .project import Project, slugify

ANCHOR_START, ANCHOR_END = 0.15, 0.25  # seconds a line's slot may sit from its measured utterance
STAGES = ("script", "voice", "picture")


def on_screen(project: Project, folder: Path) -> dict[int, set[str]]:
    """Who each shot shows, where someone has said so. A shot missing here has not been described.

    Sources: the blueprint's per-character shot lists, its `no_one_on_screen` list for shots that
    show no character at all, and `"characters"` in the remix's look.json corrections.
    """
    shown: dict[int, set[str]] = {}
    blueprint, look = project.root / "blueprint.json", folder / "look.json"
    if blueprint.is_file():
        data = json.loads(blueprint.read_text(encoding="utf-8"))
        for number in data.get("no_one_on_screen") or []:
            shown[int(number)] = set()
        for character in data.get("characters", []):
            for number in character.get("shots") or []:
                shown.setdefault(number, set()).add(character["id"])
    if look.is_file():
        for number, extra in (json.loads(look.read_text(encoding="utf-8")).get("shots") or {}).items():
            if "characters" in extra:
                shown[int(number)] = set(extra["characters"])
    return shown


def _sound(project: Project, folder: Path) -> Path:
    """The folder holding the voices this remix uses: its own, or the take its look.json names."""
    look = folder / "look.json"
    source = json.loads(look.read_text(encoding="utf-8")).get("audio_from") if look.is_file() else None
    return project.root / "remixes" / slugify(source) if source else folder


def plan(project: Project, name: str) -> dict:
    """The dialogue map: per line its speaker, times and shots; per shot who speaks and who stays silent."""
    from .restyle import heard_in

    folder = voices.folder(project, name)
    script = voices.load_lines(project, name)
    lines = script["lines"]
    timeline = project.root / "timeline.json"
    shots = json.loads(timeline.read_text(encoding="utf-8"))["shots"] if timeline.is_file() else []
    real, shown, talkers = voices.utterances(project, lines), on_screen(project, folder), {line["speaker"] for line in lines}
    report = _sound(project, folder) / "voice.json"
    placed = {row["id"]: row for row in json.loads(report.read_text(encoding="utf-8")).get("report", [])} if report.is_file() else {}
    rows = []
    for line, utterance in zip(lines, real):
        row = placed.get(line["id"], {})
        rows.append({
            "id": line["id"], "speaker": line["speaker"], "text": line["text"],
            "slot": [line["start"], line["end"]],
            "utterance": [utterance["start"], utterance["end"]] if utterance["found"] else None,
            "heard": [row.get("words_start", line["start"]), row.get("words_end", line["end"])],
        })
    heard_lines = [{**line, "start": row["heard"][0], "end": row["heard"][1]} for line, row in zip(lines, rows)]
    per_shot = []
    for shot in shots:
        heard = heard_in(shot, heard_lines)
        present = shown.get(shot["index"])
        speaking = sorted({h["speaker"] for h in heard if present is not None and h["speaker"] in present})
        per_shot.append({
            "index": shot["index"], "start": shot["start"], "end": shot["end"],
            "heard": heard,
            "on_screen": sorted(present) if present is not None else None,  # None: nobody has said
            "speaks_on_screen": speaking,
            "off_screen": sorted({h["speaker"] for h in heard} - set(speaking)),
            "silent": sorted((present or set()) & talkers - set(speaking)),
        })
    return {"name": slugify(name), "voices": script["voices"], "lines": rows, "shots": per_shot}


def problems(project: Project, name: str, stages: tuple[str, ...] = STAGES) -> list[str]:
    """What is still wrong with the dialogue map, by stage. Empty means the gate is open."""
    folder = voices.folder(project, name)
    script = voices.load_lines(project, name)
    lines, found = script["lines"], []
    if "script" in stages:
        for line in lines:
            if line["speaker"] not in script["voices"]:
                found.append(f"script: line {line['id']} is spoken by '{line['speaker']}', who has no voice in lines.json")
        loose = []
        for line, utterance in zip(lines, voices.utterances(project, lines)):
            if utterance["found"] and (abs(utterance["start"] - line["start"]) > ANCHOR_START or abs(utterance["end"] - line["end"]) > ANCHOR_END):
                loose.append(line["id"])
        if loose:
            found.append(f"script: {len(loose)} line(s) do not sit on the original speech (first: {loose[:6]}): run `remix lines {project.slug} --name {slugify(name)} --anchor`")
        for one, two in zip(lines, lines[1:]):
            if one["end"] > two["start"] + 0.05:
                found.append(f"script: lines {one['id']} and {two['id']} overlap in time")
    if "voice" in stages:
        sound = _sound(project, folder)
        if (sound / "voice.json").is_file() and (sound / "voice.wav").is_file():
            from . import verify

            spoken_from = json.loads((sound / "voice.json").read_text(encoding="utf-8")).get("script")
            if spoken_from and spoken_from != voices.script_hash(lines):
                found.append("voice: the voices were spoken from a different version of the script: rerun `remix voices`")
            record = {"voices": "new", "audio_from": sound.name if sound != folder else None}
            for item in verify.sync_problems(project, folder, record, lines)[0]:
                found.append(f"voice: line {item.get('line')}: {item['problem']}")
    if "picture" in stages:
        unknown = [shot["index"] for shot in plan(project, name)["shots"] if shot["heard"] and shot["on_screen"] is None]
        if unknown:
            found.append(
                f"picture: lines are heard over shot(s) {unknown}, but nobody has said who is on screen there, so their prompts cannot "
                'name the speaker: add the shots to the characters in blueprint.json, or to "no_one_on_screen" if no character is visible'
            )
    return found


def render(project: Project, name: str) -> Path:
    """dialogue.md next to the remix: the map as a table, for a person or an agent to read before a run."""
    data, folder = plan(project, name), voices.folder(project, name)
    out = [f"# Dialogue map: {data['name']}", "", "| Shot | Time | On screen | Heard | Mouth closed |", "|---|---|---|---|---|"]
    for shot in data["shots"]:
        heard = "; ".join(f"{h['speaker']} {h['from']}-{h['to']} s: {h['text'][:40]}" for h in shot["heard"]) or "(nobody speaks)"
        seen = "not stated" if shot["on_screen"] is None else ", ".join(shot["on_screen"]) or "nobody"
        out.append(f"| {shot['index']} | {shot['start']:.1f}-{shot['end']:.1f} | {seen} | {heard} | {', '.join(shot['silent']) or '-'} |")
    wrong = problems(project, name)
    out += ["", "## Problems", ""] + ([f"- {item}" for item in wrong] or ["None: the gate is open."])
    path = folder / "dialogue.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
