"""Restyle pilot: run a few shots through Gemini Omni's video edit and lay frames side by side.

A pilot comes before any full restyle: it costs cents, and shows whether a style holds up on this
video's shots and whether the model keeps composition, motion and on-screen text.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import cost, google
from .media import ffmpeg
from .project import Project, slugify

CLIP_SECONDS = (3.0, 15.0)
TILE_WIDTH = 360


class RestyleError(RuntimeError):
    pass


def chosen_shots(project: Project, numbers: list[int]) -> list[dict]:
    timeline = json.loads((project.root / "timeline.json").read_text(encoding="utf-8"))
    by_index = {shot["index"]: shot for shot in timeline["shots"]}
    shots = []
    for number in numbers:
        shot = by_index.get(number)
        if shot is None:
            raise RestyleError(f"there is no shot {number}")
        if not CLIP_SECONDS[0] <= shot["duration"] <= CLIP_SECONDS[1]:
            raise RestyleError(f"shot {number} is {shot['duration']:.1f} s long; the model takes clips of 3 to 15 s")
        shots.append(shot)
    return shots


def estimate(shots: list[dict], resolution: str) -> list[cost.Line]:
    seconds = sum(shot["duration"] for shot in shots)
    return cost.omni_video(cost.load_table(), date.today(), seconds, resolution, source_seconds=seconds)


def _frame(clip: Path, seconds: float, out: Path) -> Path:
    ffmpeg("-ss", f"{seconds:.3f}", "-i", str(clip), "-frames:v", "1", "-update", "1", "-q:v", "3", str(out))
    return out


def comparison(folder: Path, results: list[dict]) -> Path:
    """One row per shot: the source frame on the left, the restyled frame on the right."""
    from PIL import Image, ImageDraw

    from .media import audio_seconds

    # Every restyled shot in the folder, including ones from earlier pilot runs under this name.
    results = [
        {"shot": int(clip.name.split("_")[1]), "seconds": audio_seconds(clip), "source": clip.name.replace("restyled", "source"),
         "restyled": clip.name}
        for clip in sorted(folder.glob("shot_*_restyled.mp4"))
        if (folder / clip.name.replace("restyled", "source")).is_file()
    ] or results
    rows = []
    for result in results:
        middle = result["seconds"] / 2
        pair = [
            Image.open(_frame(folder / result[key], middle, folder / f"_{key}_{result['shot']}.jpg")).convert("RGB")
            for key in ("source", "restyled")
        ]
        rows.append([img.resize((TILE_WIDTH, round(TILE_WIDTH * img.height / img.width))) for img in pair])
    height = sum(max(a.height, b.height) for a, b in rows) + 8 * (len(rows) + 1)
    sheet = Image.new("RGB", (TILE_WIDTH * 2 + 24, height), (18, 18, 18))
    draw, y = ImageDraw.Draw(sheet), 8
    for (before, after), result in zip(rows, results):
        sheet.paste(before, (8, y))
        sheet.paste(after, (TILE_WIDTH + 16, y))
        draw.text((12, y + 4), f"shot {result['shot']} source", fill=(255, 255, 255))
        draw.text((TILE_WIDTH + 20, y + 4), "restyled", fill=(255, 255, 255))
        y += max(before.height, after.height) + 8
    for scrap in folder.glob("_*.jpg"):
        scrap.unlink()
    path = folder / "comparison.jpg"
    sheet.save(path, quality=88)
    return path


FPS = 30
MIN_CLIP = 3.05  # the model refuses clips under 3 s, so shorter shots are slowed down to this and sped back up
DEFAULT_KEEP = (
    "Remove every subtitle and caption from the picture, but keep diagram labels, code and text on screens exactly "
    "as they are. Keep the same composition, camera, poses, motion and timing. Show only the characters named here "
    "and add no one."
)


def billed_usd(shots: list[dict], resolution: str = "360p") -> float:
    """What restyling these shots costs at most: the price table's seconds for each clip, in and out."""
    table = cost.load_table()
    seconds = cost.omni_seconds([shot["duration"] for shot in shots], table)
    return cost.total(cost.omni_video(table, date.today(), seconds, resolution, source_seconds=seconds)) if shots else 0.0


def _correction(look: dict, number: int) -> str:
    return json.dumps((look.get("shots") or {}).get(str(number)) or {}, sort_keys=True)


def _outdated(clips: Path, number: int, spoken: str, correction: str, script_time: float) -> bool:
    """Whether a finished shot was made for another line, or another correction, than it has now."""
    from .verify import words_of

    restyled, note, asked = (clips / f"{number:04d}_{name}" for name in ("restyled.mp4", "spoken.txt", "asked.json"))
    if (asked.read_text(encoding="utf-8") if asked.is_file() else "{}") != correction:
        return True
    if not note.is_file():  # made before these notes existed: current only if the script is older than the clip
        return script_time > restyled.stat().st_mtime
    return words_of(note.read_text(encoding="utf-8")) != words_of(spoken)


def pending_shots(project: Project, name: str, only: tuple[int, ...] | None = None) -> list[dict]:
    """The shots a run would pay for: no finished clip yet, or one made for a line or correction that has changed."""
    from . import voices

    folder = project.root / "remixes" / slugify(name)
    look = json.loads((folder / "look.json").read_text(encoding="utf-8"))
    lines = json.loads((folder / "lines.json").read_text(encoding="utf-8"))["lines"]
    script_time, clips, todo = (folder / "lines.json").stat().st_mtime, folder / "shots", []
    for shot in json.loads((project.root / "timeline.json").read_text(encoding="utf-8"))["shots"]:
        number = shot["index"]
        if only is not None and number not in only:
            continue
        if not (clips / f"{number:04d}_restyled.mp4").is_file() or _outdated(
            clips, number, voices.spoken_in(shot, lines), _correction(look, number), script_time
        ):
            todo.append(shot)
    return todo


def heard_in(shot: dict, lines: list[dict], stretch: float = 1.0) -> list[dict]:
    """Every line audible during a shot, with its speaker and when it is heard inside the clip sent to the model."""
    heard = []
    for line in lines:
        start, end = max(line["start"], shot["start"]), min(line["end"], shot["end"])
        # A line counts for a shot when a real part of it is heard there; a few frames of a line
        # that starts just before the cut belong to the next shot.
        if end - start >= 0.3 or (end > start and end - start >= 0.6 * (line["end"] - line["start"])):
            heard.append({"speaker": line["speaker"], "text": line["text"],
                          "from": round((start - shot["start"]) * stretch, 1), "to": round((end - shot["start"]) * stretch, 1)})
    return heard


def shot_prompt(look: dict, shot: dict, present: list[str], heard: list[dict], voices_of: tuple[str, ...] = ()) -> str:
    """The edit instruction for one shot: the style, the characters in it, what happens, and who speaks when.

    The model animates whoever is on screen unless told otherwise, so every prompt names the
    speaker of each line, the seconds it is heard, and which characters keep their mouths closed.
    """
    # look.json may correct single shots: "shots": {"26": {"description": "...", "note": "..."}}.
    # `description` replaces the blueprint's; `note` is added after it.
    extra = (look.get("shots") or {}).get(str(shot.get("index"))) or {}
    description = extra.get("description") or shot.get("description", "")
    for old, new in (look.get("rename") or {}).items():
        description = description.replace(old, new)
    names = look.get("names") or {}
    parts = [look["style"]] + [look["characters"][c] for c in present if c in look["characters"]]
    if description:
        parts.append(f"In this shot: {description}")
    talking = []
    for line in heard:
        who = names.get(line["speaker"], line["speaker"])
        if line["speaker"] in present:
            talking.append(line["speaker"])
            # Naming the seconds alone gave a mouth that moved for the first words and then froze in a smile.
            parts.append(
                f'From {line["from"]} to {line["to"]} seconds {who} says: "{line["text"]}" '
                f"For that whole span, from its first moment to its last, {who}'s mouth keeps opening and closing as in speech."
            )
        else:
            parts.append(f'From {line["from"]} to {line["to"]} seconds {who} is heard from off screen saying: "{line["text"]}"')
    if talking:
        speakers = " and ".join(dict.fromkeys(names.get(c, c) for c in talking))
        parts.append(f"Only {speakers} moves a mouth, and only during those seconds; at every other moment all mouths are closed.")
    else:
        parts.append("Nobody on screen speaks in this shot: every mouth stays closed the whole time.")
    for silent in present:
        if silent in voices_of and silent not in talking:
            who = names.get(silent, silent)
            parts.append(f"{who[:1].upper()}{who[1:]} does not speak in this shot and only listens, mouth closed.")
    if extra.get("note"):
        parts.append(extra["note"])
    return " ".join(parts + [look["keep"]])


def _run(args: list[str], cwd: Path | None = None) -> None:
    import subprocess

    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RestyleError("ffmpeg failed: " + proc.stderr.strip()[-400:])


def full(
    project: Project, name: str, audio_from: str | None = None, resolution: str = "360p", workers: int = 3,
    dry_run: bool = False, edit=google.edit_video, only: tuple[int, ...] | None = None,
) -> dict:
    """Restyle every shot, put the remix's audio under the new picture, and burn in its captions.

    Reads `look.json` (style and character designs) and `lines.json` from the remix folder. Finished
    shots are kept, so a rerun only pays for the ones that are missing or failed. With `only`, just
    those shots are made (a pilot); the others keep whatever picture they have, and the later full
    run reuses the piloted clips.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from . import voices
    from .media import audio_seconds
    from .verify import stream_facts, words_of

    project.require_reusable()
    folder = project.root / "remixes" / slugify(name)
    look = json.loads((folder / "look.json").read_text(encoding="utf-8"))
    lines = json.loads((folder / "lines.json").read_text(encoding="utf-8"))["lines"]
    audio_from = audio_from or look.get("audio_from")
    if not audio_from:
        raise RestyleError('say which remix supplies the audio: --audio-from, or "audio_from" in look.json')
    if not (project.root / "remixes" / slugify(audio_from) / "output.mp4").is_file():
        raise RestyleError(f"`{slugify(audio_from)}` has no finished video to take the audio from: mix it first")
    # One script for everything: the audio going under this picture must have been voiced from the
    # same lines that the shot prompts and the burned-in captions are built from.
    theirs = project.root / "remixes" / slugify(audio_from) / "lines.json"
    if theirs.is_file():
        other = json.loads(theirs.read_text(encoding="utf-8"))["lines"]
        if voices.script_hash(other) != voices.script_hash(lines):
            differing = [line["id"] for line in lines if line not in other][:8]
            raise RestyleError(
                f"the audio of `{slugify(audio_from)}` was made from a different script than this remix's lines.json "
                f"(first differing lines: {differing}); the voices, captions and picture must share one script"
            )
    # The prompts say when each line is HEARD, so they use where the voice was actually placed.
    placed = project.root / "remixes" / slugify(audio_from) / "voice.json"
    report = json.loads(placed.read_text(encoding="utf-8")) if placed.is_file() else {}
    times = {row["id"]: row for row in report.get("report", []) if "words_start" in row} if report.get("script") == voices.script_hash(lines) else {}
    heard_lines = [
        {**line, "start": times[line["id"]]["words_start"], "end": times[line["id"]]["words_end"]} if line["id"] in times else line
        for line in lines
    ]
    script_time = (folder / "lines.json").stat().st_mtime
    stale: list[int] = []
    blueprint =json.loads((project.root / "blueprint.json").read_text(encoding="utf-8"))
    info = project.load()["source"]
    width, height = (360, round(360 * info["height"] / info["width"] / 2) * 2) if info["height"] >= info["width"] else (
        round(360 * info["width"] / info["height"] / 2) * 2, 360)
    clips = folder / "shots"
    clips.mkdir(parents=True, exist_ok=True)
    references = []
    if look.get("reference_clip") and not dry_run:
        clip = project.root / look["reference_clip"]
        references = [_frame(clip, audio_seconds(clip) / 2, folder / "reference.jpg").read_bytes()]
    table, today, lock, failures, spent = cost.load_table(), date.today(), threading.Lock(), {}, []

    def work(shot: dict) -> Path:
        number, seconds = shot["index"], shot["duration"]
        if seconds > CLIP_SECONDS[1]:
            raise RestyleError(f"shot {number} is longer than 15 s; splitting long shots is not built")
        source, restyled, final = (clips / f"{number:04d}_{kind}.mp4" for kind in ("source", "restyled", "final"))
        stretch = max(MIN_CLIP / seconds, 1.0)
        if not source.is_file():
            _run(["-ss", f"{shot['start']:.3f}", "-t", f"{seconds:.3f}", "-i", str(project.source), "-an",
                  "-vf", f"setpts=PTS*{stretch:.5f},fps={FPS},scale={width}:{height},setsar=1",
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(source)])
        # Each restyled shot remembers the line it was generated for. If the script has changed
        # under it since, the clip is set aside and the shot is generated again for the new line.
        # The same goes for a correction written for the shot in look.json.
        spoken, note = voices.spoken_in(shot, lines), clips / f"{number:04d}_spoken.txt"
        asked, correction = clips / f"{number:04d}_asked.json", _correction(look, number)
        chosen = only is None or number in only
        if restyled.is_file() and not dry_run and chosen:
            if not note.is_file() and script_time <= restyled.stat().st_mtime:
                note.write_text(spoken, encoding="utf-8")  # made before these notes existed, from this script
            if _outdated(clips, number, spoken, correction, script_time):
                restyled.replace(clips / f"{number:04d}_restyled_stale.mp4")
                with lock:
                    stale.append(number)
        if not restyled.is_file() and not dry_run and chosen:
            correction_for = (look.get("shots") or {}).get(str(number)) or {}
            # A correction may list who is on screen when the blueprint missed someone (a dim figure in a diagram shot).
            present = correction_for.get("characters") or [c["id"] for c in blueprint["characters"] if number in (c.get("shots") or [])]
            described = next(s for s in blueprint["shots"] if s["index"] == number)
            speakers = tuple(dict.fromkeys(line["speaker"] for line in lines))
            prompt = shot_prompt(look, described, present, heard_in(shot, heard_lines, stretch), speakers)
            (clips / f"{number:04d}_prompt.txt").write_text(prompt, encoding="utf-8")
            try:
                video, usage = edit(source.read_bytes(), prompt, resolution, references)
                restyled.write_bytes(video)
                note.write_text(spoken, encoding="utf-8")
                asked.write_text(correction, encoding="utf-8")
                with lock:
                    for line in (
                        table.line(f"video at {resolution}", "omni-1.2-flash.video-out", usage.get("total_output_tokens", 0), today),
                        table.line("source video and prompt sent to the model", "omni-1.2-flash.input", usage.get("total_input_tokens", 0), today),
                    ):
                        cost.record(project.slug, folder.name, line, note=f"restyle shot {number}")
                        spent.append(line.usd)
            except google.GoogleError as exc:
                with lock:
                    failures[number] = str(exc)[:200]
        picture = restyled if restyled.is_file() else source
        frames = max(round(shot["end"] * FPS) - round(shot["start"] * FPS), 1)
        factor = (frames / FPS) / audio_seconds(picture)
        _run(["-i", str(picture), "-an",
              "-vf", f"setpts=PTS*{factor:.6f},fps={FPS},tpad=stop_mode=clone:stop_duration=1,scale={width}:{height},setsar=1",
              "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(final)])
        return final

    shots = json.loads((project.root / "timeline.json").read_text(encoding="utf-8"))["shots"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        finals = list(pool.map(work, shots))

    (folder / "shots.txt").write_text("".join(f"file 'shots/{clip.name}'\n" for clip in finals), encoding="utf-8")
    _run(["-f", "concat", "-safe", "0", "-i", "shots.txt", "-c", "copy", "picture.mp4"], cwd=folder)
    # Every cut must land on the same frame as in the source, or the voices drift against the picture.
    expected = sum(max(round(s["end"] * FPS) - round(s["start"] * FPS), 1) for s in shots)
    made = stream_facts(folder / "picture.mp4").get("video", {}).get("frames")
    if made is not None and made != expected:
        raise RestyleError(f"the new picture has {made} frames where the shots add up to {expected}; it would drift out of sync")
    voices.write_captions(project, name)
    sound = (project.root / "remixes" / slugify(audio_from) / "output.mp4").resolve()
    _run(["-i", "picture.mp4", "-i", str(sound), "-vf", "subtitles=captions.ass", "-map", "0:v:0", "-map", "1:a:0",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
          "-shortest", "output.mp4"], cwd=folder)
    record = {
        "name": folder.name, "mode": "restyle", "voices": "new", "resolution": resolution, "output": project.rel(folder / "output.mp4"),
        "shots": len(shots), "restyled": sum(1 for s in shots if (clips / f"{s['index']:04d}_restyled.mp4").is_file()),
        "failed": failures, "usd_this_run": round(sum(spent), 4), "audio_from": slugify(audio_from), "dry_run": dry_run,
        "duration": audio_seconds(folder / "output.mp4"),
        "script": voices.script_hash(lines), "fps": FPS, "frames": expected, "regenerated_for_script": sorted(stale),
        # Shots that still show the source picture. While any are left this is a pilot, not a finished remix.
        "pending": [s["index"] for s in shots if not (clips / f"{s['index']:04d}_restyled.mp4").is_file()] if not dry_run else [],
    }
    (folder / "remix.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def pilot(project: Project, name: str, numbers: list[int], style: str, resolution: str = "360p", edit=google.edit_video) -> dict:
    project.require_reusable()
    shots = chosen_shots(project, numbers)
    folder = project.root / "remixes" / slugify(name)
    folder.mkdir(parents=True, exist_ok=True)
    table, today, results = cost.load_table(), date.today(), []
    for shot in shots:
        number, seconds = shot["index"], shot["duration"]
        source = folder / f"shot_{number:04d}_source.mp4"
        ffmpeg(
            "-ss", f"{shot['start']:.3f}", "-i", str(project.source), "-t", f"{seconds:.3f}", "-an",
            "-vf", "scale='if(gt(iw,ih),-2,360)':'if(gt(iw,ih),360,-2)'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(source),
        )
        video, usage = edit(source.read_bytes(), style, resolution)
        restyled = folder / f"shot_{number:04d}_restyled.mp4"
        restyled.write_bytes(video)
        lines = cost.omni_video(table, today, seconds, resolution, source_seconds=seconds)
        if usage.get("total_output_tokens") and usage.get("total_input_tokens"):  # bill what Google reports
            lines = [
                table.line(f"video at {resolution}", "omni-1.2-flash.video-out", usage["total_output_tokens"], today),
                table.line("source video and prompt sent to the model", "omni-1.2-flash.input", usage["total_input_tokens"], today),
            ]
        for line in lines:
            cost.record(project.slug, folder.name, line, estimated_usd=line.usd,
                        note=f"restyle pilot, shot {number}; reported usage {json.dumps(usage)[:300]}")
        results.append({"shot": number, "seconds": seconds, "source": source.name, "restyled": restyled.name,
                        "usd": cost.total(lines), "usage": usage})
    sheet = comparison(folder, results)
    record = {"mode": "restyle pilot", "style": style, "resolution": resolution, "shots": results,
              "usd": round(sum(r["usd"] for r in results), 4), "comparison": sheet.name}
    (folder / "restyle_pilot.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
