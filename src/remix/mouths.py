"""The mouth check: in every shot, does the right character's mouth move, and only theirs?

A video model animates whoever is on screen, so a shot can come back with the listener talking
or the speaker smiling through their line. Still frames miss it and no audio measurement can see
it. This module shows each restyled clip to a video-understanding model (Gemini, which takes the
clip itself and answers with timestamps), asks when each character's mouth is moving, and holds
the answer against the dialogue map:

- the speaker's mouth must be seen moving for at least half of each line they speak in the shot;
- nobody else's mouth may move for more than half a second while another character is heard;
- no mouth may move for more than a second while nobody is heard at all.

A character whose mouth cannot be seen (back to the camera, hands only, too dark) is left out.
Results are kept next to each clip as `NNNN_mouths.json`, keyed by the clip and what was expected
of it, so a clip is paid for once. `remix restyle` retakes a shot that fails, and `remix verify`
fails a video that still has one.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from . import cost, google, voices
from .project import Project, slugify

SPEAKS = 0.5  # share of a line during which its speaker's mouth must be seen moving
SHORT = 0.6  # a line heard for less than this in a shot only needs some movement near it
SILENT = 0.5  # seconds a listener's mouth may move while someone else is heard
IDLE = 1.0  # seconds any mouth may move while nobody is heard
MIN_CLIP = 3.05  # shots shorter than this were slowed to this length for the video model (see restyle.py)

VERSION = "v2"  # bump when the question or the rules change, so kept results are asked again

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scene_changes": {"type": "ARRAY", "items": {"type": "NUMBER"}},
        "characters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "mouth_visible": {"type": "BOOLEAN"},
                    "mouth_moving": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "NUMBER"}}},
                },
                "required": ["id", "mouth_visible", "mouth_moving"],
            },
        }
    },
    "required": ["characters", "scene_changes"],
}


def expected(shot_plan: dict, talkers: set[str], stretch: float = 1.0) -> dict:
    """What the dialogue map says about one shot: who is watched, and when each of them is heard, in clip seconds."""
    present = shot_plan["on_screen"] or []
    heard = [
        {"speaker": line["speaker"], "from": round(line["from"] * stretch, 2), "to": round(line["to"] * stretch, 2)}
        for line in shot_plan["heard"]
    ]
    return {"watch": sorted(set(present) & talkers), "heard": heard}


def question(look: dict, want: dict) -> str:
    """The instruction for the watching model: who to look at, and what to report about each."""
    names, looks = look.get("names") or {}, look.get("characters") or {}
    who = "\n".join(f'- id "{c}": {names.get(c, c)}. {looks.get(c, "")}'.strip() for c in want["watch"])
    return (
        "Watch this short clip closely, frame by frame. It is stop-motion style animation with no sound.\n"
        "For each character listed below, report when its MOUTH is visibly moving as in speech: opening and closing, "
        "changing shape. A mouth held still does not count, whether it is closed, smiling, or held open. Blinks, head "
        "turns, gestures and body movement do not count.\n"
        f"Characters:\n{who}\n"
        "For each one give: mouth_visible (false if the character is not in the picture, or its mouth cannot be seen "
        "because it faces away, is hidden, or is too dark or small to judge), and mouth_moving as a list of "
        "[start_seconds, end_seconds] spans measured from the start of the clip, empty if the mouth never moves. "
        "Be exact about the seconds. Use the ids as given.\n"
        "Also report scene_changes: the seconds at which the picture cuts or jumps to a different scene, room or "
        "view that is not a continuation of the same shot (empty if the clip is one continuous shot)."
    )


def record_path(clip: Path) -> Path:
    return clip.with_name(clip.name.split("_")[0] + "_mouths.json")


def _overlap(spans: list, start: float, end: float) -> float:
    return sum(max(min(b, end) - max(a, start), 0.0) for a, b in spans)


TAIL = 0.6  # a scene change this close to the clip's end is a stray last frame that is trimmed off, not a retake


def trim_point(seen: dict, length: float | None) -> float | None:
    """Where to cut a clip that ends on a flash of another scene, or None when it does not."""
    late = [m for m in seen.get("scene_changes") or [] if isinstance(m, (int, float)) and length and m > max(length - TAIL, 0.25)]
    return round(max(min(late) - 0.1, 1.0), 2) if late else None


def judge(want: dict, seen: dict, length: float | None = None) -> list[str]:
    """What is wrong in a clip, given who should be talking when and whose mouth was seen moving when."""
    moving = {c["id"]: [(float(a), float(b)) for a, b in (c.get("mouth_moving") or []) if len((a, b)) == 2 and b > a]
              for c in seen.get("characters", []) if c.get("mouth_visible")}
    wrong, heard = [], want["heard"]
    for moment in seen.get("scene_changes") or []:
        if isinstance(moment, (int, float)) and moment > 0.25 and not (length and moment > length - TAIL):
            wrong.append(f"the picture cuts to a different scene at {moment:.1f} s")
    length = max([line["to"] for line in heard] + [b for spans in moving.values() for _, b in spans] + [0.0])
    for who in want["watch"]:
        if who not in moving:
            continue  # the mouth cannot be seen in this shot, so there is nothing to hold against it
        mine, others = [l for l in heard if l["speaker"] == who], [l for l in heard if l["speaker"] != who]
        for line in mine:
            span = line["to"] - line["from"]
            if span < SHORT:
                if _overlap(moving[who], line["from"] - 0.3, line["to"] + 0.3) <= 0:
                    wrong.append(f"{who} speaks from {line['from']} to {line['to']} s but its mouth does not move")
            elif _overlap(moving[who], line["from"], line["to"]) < SPEAKS * span:
                seen_for = _overlap(moving[who], line["from"], line["to"])
                wrong.append(f"{who} speaks from {line['from']} to {line['to']} s but its mouth moves for only {seen_for:.1f} s of that")
        # Time when someone else is heard and this character is not: its mouth has to stay still.
        over = sum(_overlap(moving[who], l["from"], l["to"]) for l in others) - sum(
            _overlap(moving[who], max(l["from"], m["from"]), min(l["to"], m["to"])) for l in others for m in mine
        )
        if over > SILENT:
            wrong.append(f"{who}'s mouth moves for {over:.1f} s while {others[0]['speaker']} is the one heard")
        talk = sum(_overlap(moving[who], l["from"], l["to"]) for l in heard)
        idle = _overlap(moving[who], 0.0, length) - min(talk, _overlap(moving[who], 0.0, length))
        if idle > IDLE + (0.3 * len(mine)):
            wrong.append(f"{who}'s mouth moves for {idle:.1f} s while nobody is heard")
    return wrong


def _key(clip: Path, want: dict) -> str:
    return "-".join((VERSION, hashlib.sha1(clip.read_bytes()).hexdigest()[:16], hashlib.sha1(json.dumps(want, sort_keys=True).encode()).hexdigest()[:8]))


def _stretch(shot: dict) -> float:
    return max(MIN_CLIP / max(shot["end"] - shot["start"], 0.01), 1.0)


def check_clip(project: Project, name: str, shot_plan: dict, look: dict, talkers: set[str], clip: Path,
               watch=None, force: bool = False) -> dict:
    """Check one clip against the dialogue map and keep the result beside it. Paid, a fraction of a cent, once per clip."""
    want = expected(shot_plan, talkers, _stretch(shot_plan))
    path = record_path(clip)
    key = _key(clip, want)
    if not want["watch"]:
        result = {"key": key, "ok": True, "skipped": "no speaking character is on screen", "problems": []}
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if path.is_file() and not force:
        kept = json.loads(path.read_text(encoding="utf-8"))
        if kept.get("key") == key:
            return {**kept, "cached": True}
    seen, usage = (watch or google.watch_video)(clip.read_bytes(), question(look, want), SCHEMA)
    table, today = cost.load_table(), date.today()
    lines = [
        table.line("clip shown to the model that checks mouths", "gemini-flash.input", usage.get("input_tokens", 0), today),
        table.line("its answer", "gemini-flash.text-out", usage.get("output_tokens", 0), today),
    ]
    for line in lines:
        cost.record(project.slug, slugify(name), line, note=f"mouth check, shot {shot_plan['index']}")
    from .media import audio_seconds

    length = audio_seconds(clip)
    wrong = judge(want, seen, length)
    result = {"key": key, "ok": not wrong, "problems": wrong, "expected": want, "seen": seen.get("characters", []),
              "scene_changes": seen.get("scene_changes", []), "trim_to": trim_point(seen, length),
              "usd": round(cost.total(lines), 5)}
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _setup(project: Project, name: str):
    from . import dialogue

    folder = voices.folder(project, name)
    look = json.loads((folder / "look.json").read_text(encoding="utf-8"))
    plan = dialogue.plan(project, name)
    talkers = set(plan["voices"])
    return folder, look, plan, talkers


def check(project: Project, name: str, only: tuple[int, ...] | None = None, watch=None, force: bool = False, workers: int = 4) -> dict:
    """Check every restyled shot (or the ones named) and say which fail."""
    folder, look, plan, talkers = _setup(project, name)
    todo = [s for s in plan["shots"] if (only is None or s["index"] in only) and (folder / "shots" / f"{s['index']:04d}_restyled.mp4").is_file()]

    def one(shot_plan: dict):
        clip = folder / "shots" / f"{shot_plan['index']:04d}_restyled.mp4"
        return shot_plan["index"], check_clip(project, name, shot_plan, look, talkers, clip, watch, force)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(pool.map(one, todo))
    return {
        "checked": len(results),
        "watched": sum(1 for r in results.values() if not r.get("skipped")),
        "failed": {n: r["problems"] for n, r in sorted(results.items()) if not r["ok"]},
        "usd": round(sum(r.get("usd", 0) for r in results.values() if not r.get("cached")), 4),  # only what was paid this time
    }


def problems(project: Project, name: str) -> list[str]:
    """For the final check: shots whose current clip has not passed the mouth check. Makes no paid call."""
    folder, look, plan, talkers = _setup(project, name)
    wrong = []
    for shot_plan in plan["shots"]:
        clip = folder / "shots" / f"{shot_plan['index']:04d}_restyled.mp4"
        if not clip.is_file():
            continue
        want = expected(shot_plan, talkers, _stretch(shot_plan))
        if not want["watch"]:
            continue
        record = clip.with_name(f"{shot_plan['index']:04d}_mouths.json")
        key = _key(clip, want)
        kept = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
        if kept.get("key") != key:
            wrong.append(f"shot {shot_plan['index']}: nobody has checked whose mouth moves in its current picture: run `remix mouths`")
        elif not kept.get("ok") and not kept.get("accepted"):
            wrong.append(f"shot {shot_plan['index']}: {kept['problems'][0]}")
    return wrong
