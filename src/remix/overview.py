"""Merge shots and transcript into timeline.json and a short overview.md to read first."""
from __future__ import annotations

import json

from .project import Project
from .steps.shots import fmt_time
from .steps.transcript import join_words


def spoken_in(start: float, end: float, words: list[dict]) -> str:
    """The words whose midpoint falls inside [start, end)."""
    return join_words([w for w in words if start <= (w["start"] + w["end"]) / 2 < end])


def build(project: Project) -> dict:
    root = project.root
    shots_path = root / "shots" / "shots.json"
    if not shots_path.is_file():
        return {"status": "skipped", "reason": "shots have not been detected yet"}

    manifest = project.load()
    info, steps = manifest["source"], manifest["steps"]
    shots_doc = json.loads(shots_path.read_text(encoding="utf-8"))
    transcript_path = root / "transcript" / "transcript.json"
    transcript = (
        json.loads(transcript_path.read_text(encoding="utf-8")) if transcript_path.is_file() else None
    )

    words: list[dict] = []
    for seg in (transcript or {}).get("segments", []):
        words.extend(seg["words"] or [{"start": seg["start"], "end": seg["end"], "word": seg["text"]}])

    shots = []
    for shot in shots_doc["shots"]:
        last = shot is shots_doc["shots"][-1]
        end = float("inf") if last else shot["end"]
        shots.append({**shot, "text": spoken_in(shot["start"], end, words)})

    timeline_path = root / "timeline.json"
    timeline_path.write_text(
        json.dumps(
            {
                "slug": project.slug,
                "source": {k: info[k] for k in ("original_name", "duration", "width", "height", "fps")},
                "language": transcript["language"] if transcript else None,
                "shots": shots,
                "sheets": shots_doc["sheets"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# {info['original_name']}",
        "",
        f"- Project: `{project.slug}`",
        f"- Picture: {fmt_time(info['duration'])} long, {info['width']}x{info['height']}, "
        f"{info['fps']} fps, {info['video_codec']}",
    ]
    stems = steps.get("stems", {})
    if not info["has_audio"]:
        lines.append("- Audio: none")
    elif stems.get("status") == "done":
        lines.append(
            f"- Audio: speech stem {stems['speech_dbfs']} dBFS, background stem {stems['background_dbfs']} dBFS "
            "(RMS; a speech stem below about -50 means no real speech)"
        )
    else:
        lines.append("- Audio: not separated yet")
    if transcript:
        count = sum(len(seg["words"]) for seg in transcript["segments"])
        lines.append(
            f"- Speech: {transcript['language']} "
            f"({transcript['language_probability']:.0%} confidence), {count} words"
        )
        ignored = transcript.get("dropped_as_silence") or []
        if ignored:
            lines.append(
                f"- Note: {len(ignored)} transcript segment(s) had no voice energy in the speech stem and were "
                "dropped as invented text; they are listed under `dropped_as_silence` in transcript.json"
            )
    keyframes = sum(len(shot["keyframes"]) for shot in shots)
    lines.append(f"- Shots: {len(shots)}, with {keyframes} keyframes")
    lines.append("- Contact sheets: " + ", ".join(f"`{s['file']}`" for s in shots_doc["sheets"]))
    if info["vfr_suspected"]:
        lines.append("- Note: variable frame rate source; shot times come from a constant-rate proxy")
    if info["hdr"]:
        lines.append("- Note: HDR source; keyframes are not tone-mapped and may look washed out")

    lines += ["", "## Shots", "", "| # | Start | End | Length | Spoken |", "|---|---|---|---|---|"]
    for shot in shots:
        spoken = shot["text"].replace("|", "\\|") or "-"
        lines.append(
            f"| {shot['index']} | {fmt_time(shot['start'])} | {fmt_time(shot['end'])} "
            f"| {shot['duration']:.1f}s | {spoken} |"
        )

    if transcript:
        script_path = root / "transcript" / "script.txt"
        script = script_path.read_text(encoding="utf-8").strip() if script_path.is_file() else ""
        lines += ["", "## Script", "", script or "(no speech found)"]

    overview_path = root / "overview.md"
    overview_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "done", "outputs": [project.rel(overview_path), project.rel(timeline_path)]}
