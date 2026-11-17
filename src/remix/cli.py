"""remix command line: ingest a video and decompose it into stems, script and shots."""
from __future__ import annotations

import json
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Optional

import typer

from . import __version__, config, cost, dialogue, google, overview, remixes, restyle, verify, voices
from . import blueprint as bp
from .media import MediaError
from .project import RIGHTS, Project, ProjectError, find, ingest, list_projects, resolve_file, slugify

app = typer.Typer(add_completion=False, no_args_is_help=True)
blueprint_app = typer.Typer(
    no_args_is_help=True, help="Create and validate a video's blueprint: themes, style and per-shot descriptions."
)
cost_app = typer.Typer(no_args_is_help=True, help="Prices, per-video estimates and the spend ledger.")
app.add_typer(blueprint_app, name="blueprint")
app.add_typer(cost_app, name="cost")

STEPS = ("audio", "stems", "transcript", "shots", "overview")
REQUIRES = {"stems": ("audio",), "transcript": ("audio",), "overview": ("shots",)}


def _version(value: bool) -> None:
    if value:
        typer.echo(f"remix {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """Decompose videos into stems, script and shots."""


def _open(target: str) -> Project:
    path = resolve_file(target)
    if path is None:
        return find(target)
    project, created = ingest(path)
    typer.echo(f"{'ingested' if created else 'already ingested'}: projects/{project.slug}")
    return project


def _run_step(project: Project, name: str, opts: dict) -> dict:
    if name == "audio":
        from .steps import audio

        return audio.run(project)
    if name == "stems":
        from .steps import stems

        return stems.run(project, keep_stems=opts["keep_stems"])
    if name == "transcript":
        from .steps import transcript

        return transcript.run(project, model_name=opts["whisper_model"], language=opts["language"])
    if name == "shots":
        from .steps import shots

        return shots.run(
            project,
            detector=opts["detector"],
            threshold=opts["threshold"],
            max_keyframes=opts["max_keyframes"],
        )
    return overview.build(project)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _detail(name: str, step: dict) -> str:
    if step.get("status") == "skipped":
        return step.get("reason", "")
    if step.get("status") == "failed":
        return step.get("error", "")
    if name == "stems":
        return f"speech {step['speech_dbfs']} dBFS, background {step['background_dbfs']} dBFS ({step['device']})"
    if name == "transcript":
        dropped = step.get("dropped_as_silence", 0)
        return (
            f"{step['language']}, {_count(step['words'], 'word')} in "
            f"{_count(step['segments'], 'segment')} ({step['model']})"
            + (f", {_count(dropped, 'segment')} dropped as silence" if dropped else "")
        )
    if name == "shots":
        return ", ".join(
            (
                _count(step["shots"], "shot"),
                _count(step["keyframes"], "keyframe"),
                _count(step["sheets"], "contact sheet"),
            )
        )
    return ""


@app.command()
def decompose(
    target: str = typer.Argument(
        ..., help="A video file (a path, or a name inside inbox/) or an existing project slug."
    ),
    only: Optional[str] = typer.Option(None, help="Comma-separated steps to run: " + ", ".join(STEPS)),
    force: bool = typer.Option(False, "--force", help="Re-run steps that already finished."),
    whisper_model: str = typer.Option(config.WHISPER_MODEL, help="faster-whisper model name."),
    language: Optional[str] = typer.Option(None, help="Spoken language code; detected when omitted."),
    keep_stems: bool = typer.Option(False, "--keep-stems", help="Also save the four raw Demucs stems."),
    detector: str = typer.Option("adaptive", help="Shot detector: adaptive or content."),
    threshold: Optional[float] = typer.Option(None, help="Detector threshold; lower finds more cuts."),
    max_keyframes: int = typer.Option(8, help="Most keyframes sampled from a single shot."),
) -> None:
    """Split a video into speech and background stems, a timed transcript, and shots with keyframes."""
    selected = [s.strip() for s in only.split(",") if s.strip()] if only else list(STEPS)
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        typer.echo(f"error: unknown step(s): {', '.join(unknown)}. Choose from: {', '.join(STEPS)}", err=True)
        raise typer.Exit(2)
    if detector not in ("adaptive", "content"):
        typer.echo("error: --detector must be 'adaptive' or 'content'", err=True)
        raise typer.Exit(2)

    try:
        project = _open(target)
    except (ProjectError, MediaError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)

    # Unfinished prerequisites of the selected steps run too, in pipeline order.
    needed = {
        dep for step in selected for dep in REQUIRES.get(step, ()) if project.step(dep).get("status") != "done"
    }
    plan = [step for step in STEPS if step in selected or step in needed]
    opts = {
        "keep_stems": keep_stems,
        "whisper_model": whisper_model,
        "language": language,
        "detector": detector,
        "threshold": threshold,
        "max_keyframes": max_keyframes,
    }

    rows = []
    for name in plan:
        previous = project.step(name)
        if name != "overview" and previous.get("status") == "done" and not (force and name in selected):
            rows.append((name, "cached", previous))
            continue
        typer.echo(f"[{name}] running...")
        started = time.perf_counter()
        try:
            outcome = _run_step(project, name, opts)
        except Exception as exc:  # record the failure in the manifest, then stop
            if not isinstance(exc, MediaError):
                traceback.print_exc()
            project.record(name, "failed", seconds=round(time.perf_counter() - started, 1), error=str(exc)[:500])
            typer.echo(f"error: {name} failed: {exc}", err=True)
            raise typer.Exit(1)
        status = outcome.pop("status")
        project.record(name, status, seconds=round(time.perf_counter() - started, 1), **outcome)
        rows.append((name, status, project.step(name)))

    typer.echo(f"\nproject: projects/{project.slug}")
    for name, status, step in rows:
        seconds = f"{step.get('seconds', 0):.1f}s"
        typer.echo(f"  {name:<11}{status:<9}{seconds:>8}  {_detail(name, step)}".rstrip())
    typer.echo(f"read first: projects/{project.slug}/overview.md, then shots/sheets/*.jpg")


@app.command("list")
def list_cmd() -> None:
    """List projects and how far each one has been decomposed."""
    projects = list_projects()
    if not projects:
        typer.echo("no projects yet: put a video in inbox/ and run `remix decompose <file>`")
        return
    for project in projects:
        manifest = project.load()
        source = manifest["source"]
        steps = " ".join(
            f"{name}={manifest['steps'].get(name, {}).get('status', '-')}" for name in (*STEPS, "blueprint")
        )
        typer.echo(
            f"{project.slug}  {source['duration']:.0f}s  {source['width']}x{source['height']}  "
            f"rights={project.rights() or '-'}  {steps}"
        )


def _project(target: str) -> Project:
    try:
        return find(target)
    except ProjectError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


@app.command()
def info(target: str = typer.Argument(..., help="Project slug or a unique prefix of it.")) -> None:
    """Print a project's manifest."""
    typer.echo(json.dumps(_project(target).load(), indent=2))


def _day(value: Optional[str]) -> date:
    try:
        return date.fromisoformat(value) if value else date.today()
    except ValueError:
        typer.echo("error: give the date as YYYY-MM-DD", err=True)
        raise typer.Exit(2)


@app.command()
def rights(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    status: str = typer.Argument(..., help="Whose video this is: " + ", ".join(RIGHTS)),
    note: str = typer.Option("", help="Optional note, such as who confirmed it."),
) -> None:
    """Record whether a video's own footage, music and voices may be reused in outputs."""
    project = _project(target)
    try:
        project.set_rights(status, note)
    except ProjectError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    typer.echo(f"{project.slug}: rights={status}")


UNVERIFIED = 3  # exit code: the video was made, but it did not pass its script and sync check


def _check(project: Project, name: str, frames_ok: bool = False) -> bool:
    """Run the script and sync check on a remix and print what it found.

    Every command that makes or remakes a remix video ends here, so no video leaves the tool
    unchecked (the user's standing rule).
    """
    try:
        result = verify.run(project, name, frames_ok=frames_ok)
    except (voices.VoiceError, MediaError, FileNotFoundError) as exc:
        typer.echo(f"error: the script check could not run: {exc}", err=True)
        return False
    measured = result.get("measured") or {}
    if "voice_offset_seconds" in measured:
        typer.echo(f"  voices sit {measured['voice_offset_seconds']:+.2f} s from the picture (track found at {measured['voice_found']:.2f})")
    mouths = measured.get("against_original_speech") or {}
    if mouths.get("lines_timed"):
        typer.echo(
            f"  against the original speech (when the mouths move), {mouths['lines_timed']} lines: starts off by "
            f"{mouths['mean_start_error']:.2f} s on average (worst {mouths['worst_start_error']:+.2f}), "
            f"ends by {mouths['mean_end_error']:.2f} s (worst {mouths['worst_end_error']:+.2f})"
        )
    if result["sheets"]:
        typer.echo(f"  {_count(len(result['sheets']), 'frame sheet')} to read in projects/{project.slug}/remixes/{slugify(name)}/check/")
    if result["passed"]:
        typer.echo(f"verified: {_count(result['lines'], 'line')} match the video")
        return True
    typer.echo(f"NOT verified: {_count(len(result['problems']), 'problem')}")
    for problem in result["problems"]:
        where = f"line {problem['line']}: " if "line" in problem else ""
        typer.echo(f"  - {where}{problem['problem']}" + (f" (lines {problem['lines']})" if "lines" in problem else ""))
    return False


def _gate(project: Project, name: str, stages: tuple[str, ...]) -> None:
    """Stop before a paid step while the dialogue map has holes (see dialogue.py)."""
    wrong = dialogue.problems(project, name, stages)
    if wrong:
        typer.echo("error: the dialogue gate is closed, so nothing was generated:", err=True)
        for item in wrong[:8]:
            typer.echo(f"  - {item}", err=True)
        raise typer.Exit(1)


@app.command("dialogue")
def dialogue_cmd(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Short name of the remix."),
) -> None:
    """The dialogue gate: who says what, when, over which shot, and what is still missing. Writes dialogue.md."""
    project = _project(target)
    try:
        path = dialogue.render(project, name)
        data, wrong = dialogue.plan(project, name), dialogue.problems(project, name)
    except (voices.VoiceError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    talking = [shot for shot in data["shots"] if shot["heard"]]
    typer.echo(f"wrote projects/{project.slug}/{project.rel(path)}: {_count(len(data['lines']), 'line')} over {_count(len(talking), 'shot')}")
    typer.echo(f"  {sum(1 for s in talking if s['off_screen'])} shot(s) carry a line whose speaker is off screen; "
               f"{sum(1 for s in talking if s['silent'])} show someone who must stay silent")
    if not wrong:
        typer.echo("dialogue gate open: script, voice and picture agree")
        return
    typer.echo(f"dialogue gate CLOSED: {_count(len(wrong), 'problem')}")
    for item in wrong:
        typer.echo(f"  - {item}")
    raise typer.Exit(1)


@app.command("audio-remix")
def audio_remix_cmd(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Short name for this remix."),
    music: str = typer.Option(..., help="Music file to put under the picture (a path under the data folder)."),
    music_gain: float = typer.Option(0.0, help="Trim in dB on top of matching the original background level."),
    no_speech: bool = typer.Option(False, "--no-speech", help="Leave the original voices out."),
    new_voices: bool = typer.Option(False, "--new-voices", help="Use this remix's voice.wav instead of the original voices."),
    defer_check: bool = typer.Option(False, "--defer-check", help="Do not run the check now. The video stays unverified until `remix verify` passes."),
) -> None:
    """Replace the background audio with new music, keeping the original picture."""
    project = _project(target)
    voice = voices.folder(project, name) / "voice.wav" if new_voices else None
    try:
        record = remixes.audio_remix(
            project, name, Path(music), music_gain_db=music_gain, keep_speech=not no_speech, voice=voice
        )
    except (ProjectError, remixes.RemixError, MediaError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote projects/{project.slug}/{record['output']}")
    typer.echo(
        f"  music set {record['music_gain_db']:+.1f} dB to sit at the old background level, "
        f"{record['music_fit']}, voices: {record['voices']}, {record['duration']:.1f}s"
    )
    if defer_check:
        typer.echo("NOT verified: the check was deferred. Run `remix verify` before this video is handed over or approved.")
        raise typer.Exit(UNVERIFIED)
    if not _check(project, name):
        raise typer.Exit(UNVERIFIED)


@app.command()
def music(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Short name of the remix this music belongs to."),
    prompt: str = typer.Option(..., help="What the music should be. Say 'instrumental' to exclude vocals."),
    yes: bool = typer.Option(False, "--yes", help="Run even if the estimate is above the auto-run cap."),
) -> None:
    """Generate a music track for a remix with Google Lyria (a paid call, logged in the ledger)."""
    project = _project(target)
    today = date.today()
    lines = cost.music(cost.load_table(), today, project.load()["source"]["duration"])
    estimate = cost.total(lines)
    typer.echo(f"estimate: ${estimate:.2f} ({lines[0].what} on {lines[0].model})")
    if not cost.within_cap(estimate) and not yes:
        typer.echo(f"error: that is above the ${config.CAP_USD:g} auto-run cap; pass --yes to run it", err=True)
        raise typer.Exit(1)

    folder = project.root / "remixes" / slugify(name)
    try:
        audio, mime, notes = google.generate_music(prompt)
    except google.GoogleError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    folder.mkdir(parents=True, exist_ok=True)
    track = folder / f"music{google.EXTENSIONS.get(mime, '.mp3')}"
    track.write_bytes(audio)
    (folder / "music_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    if notes:
        (folder / "music_notes.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    for line in lines:
        cost.record(project.slug, folder.name, line, estimated_usd=line.usd)
    typer.echo(f"wrote projects/{project.slug}/{project.rel(track)} ({len(audio) / 1e6:.1f} MB, {mime})")
    typer.echo(f"logged ${estimate:.2f} to the ledger")


@app.command()
def lines(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Short name of the remix."),
    force: bool = typer.Option(False, "--force", help="Rebuild lines.json from the transcript."),
    anchor: bool = typer.Option(False, "--anchor", help="Move each line's time slot onto its original utterance, measured on the speech stem."),
) -> None:
    """Write a remix's lines.json: the script as timed lines with speakers and a voice per speaker."""
    project = _project(target)
    try:
        if anchor:
            moved = voices.anchor(project, name)
            typer.echo(f"{_count(len(moved), 'line')} moved onto the original speech")
            for row in moved:
                typer.echo(f"  line {row['id']}: start {row['start']:+.2f} s, end {row['end']:+.2f} s")
            return
        path = voices.write_lines(project, name, force=force)
    except voices.VoiceError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    script = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(f"wrote projects/{project.slug}/{project.rel(path)}: {_count(len(script['lines']), 'line')}")
    typer.echo("  voices: " + ", ".join(f"{speaker} = {voice}" for speaker, voice in script["voices"].items()))
    typer.echo("edit the text, voices or directions there, then run `remix voices`")


@app.command("voices")
def voices_cmd(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Short name of the remix."),
    yes: bool = typer.Option(False, "--yes", help="Run even if the estimate is above the auto-run cap."),
    pilot: int = typer.Option(0, help="Speak only the first N lines into voice_pilot.wav, as a cheap check."),
    plain: bool = typer.Option(False, "--plain", help="Allow a full run on a script with few or no vocal cues."),
    redo: str = typer.Option("", help="Line ids to speak again, such as 23,44. Other lines keep their takes unless they changed."),
) -> None:
    """Speak a remix's lines with stock AI voices (a paid step, logged in the ledger)."""
    project = _project(target)
    try:
        script = voices.load_lines(project, name)
        chosen = script["lines"][:pilot] if pilot else script["lines"]
        cued = sum(1 for line in chosen if voices.cues(line))
        typer.echo(f"{cued} of {_count(len(chosen), 'line')} carry vocal cues (sighs, chuckles, breaths, pauses)")
        if not pilot:
            _gate(project, name, ("script",))
        if not pilot and not plain and cued < voices.PERFORMED * len(chosen):
            typer.echo(
                "error: this script would be spoken flat. The standing rule is expressive, natural voices: give at least "
                f"{voices.PERFORMED:.0%} of the lines a `say` text with vocal cues in lines.json (CLAUDE.md, 'Expressive voices'), "
                "or pass --plain to speak it as it is",
                err=True,
            )
            raise typer.Exit(1)
        tight = voices.tight_lines(chosen, project.load()["source"]["duration"])
        if tight:
            typer.echo(f"  warning: lines {tight} have little room for their cues before the next line and may run over")
        planned = cost.total(voices.estimate({**script, "lines": chosen}, date.today()))
        typer.echo(f"estimate: ${planned:.3f} for {_count(len(chosen), 'line')}")
        if not cost.within_cap(planned) and not yes:
            typer.echo(f"error: that is above the ${config.CAP_USD:g} auto-run cap; pass --yes to run it", err=True)
            raise typer.Exit(1)
        result = voices.generate(project, name, pilot=pilot, redo=tuple(int(n) for n in redo.replace(" ", "").split(",") if n))
    except (voices.VoiceError, google.GoogleError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote projects/{project.slug}/remixes/{slugify(name)}/{result['file']}")
    typer.echo(
        f"  {_count(result['lines'], 'line')} ({result['respoken']} spoken now, the rest kept), {result['sped_up']} sped up, "
        f"{voices.fit_note(result['overrun'], result['lines'])}; logged ${result['usd']:.4f} "
        f"(estimated ${result['estimated_usd']:.4f})"
    )
    if result["suspect"]:
        typer.echo(
            f"  warning: {_count(result['suspect'], 'line')} took far longer than the words should. "
            "The model may be saying something extra; transcribe the result before using it."
        )


@app.command()
def captions(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Short name of the remix."),
    burn: bool = typer.Option(False, "--burn", help="Also render them into a copy of the remix's output video."),
) -> None:
    """Write captions for a remix from its lines (SRT, and ASS with one color per speaker)."""
    project = _project(target)
    try:
        result = voices.write_captions(project, name)
        typer.echo(f"wrote {_count(result['cues'], 'cue')}: " + ", ".join(result["outputs"]))
        if burn:
            record = json.loads((voices.folder(project, name) / "remix.json").read_text(encoding="utf-8"))
            burned = voices.burn_captions(project, name, project.root / record["output"])
            typer.echo(f"burned into projects/{project.slug}/{project.rel(burned)}")
    except (voices.VoiceError, FileNotFoundError, MediaError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    if burn and not _check(project, name):
        raise typer.Exit(UNVERIFIED)


@app.command("verify")
def verify_cmd(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Short name of the remix."),
    frames_ok: bool = typer.Option(False, "--frames-ok", help="Say that a regenerated picture's frames were read against the script."),
) -> None:
    """Check that a remix's audio, captions and picture match its script. Required before approval."""
    if not _check(_project(target), name, frames_ok=frames_ok):
        raise typer.Exit(1)


@app.command("restyle")
def restyle_cmd(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Remix folder holding look.json and lines.json."),
    audio_from: Optional[str] = typer.Option(None, help="Remix whose finished audio goes under the new picture; look.json's audio_from when omitted."),
    only: str = typer.Option("", help="Pilot: restyle just these shots, such as 2,3,4. The full run later reuses them."),
    resolution: str = typer.Option("360p", help="360p, 720p or 1080p."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do everything except the paid calls, using the source shots."),
    yes: bool = typer.Option(False, "--yes", help="Run even if the estimate is above the auto-run cap."),
    workers: int = typer.Option(3, help="Shots sent to the model at once. The sign-in token lasts an hour, so long videos need more."),
    defer_check: bool = typer.Option(False, "--defer-check", help="Do not run the check now. The video stays unverified until `remix verify` passes."),
) -> None:
    """Restyle every shot of a video, add the remix's audio, and burn in its captions (paid)."""
    project = _project(target)
    try:
        chosen = tuple(int(n) for n in only.replace(" ", "").split(",") if n) or None
        if not dry_run:
            _gate(project, name, dialogue.STAGES)
        todo = restyle.pending_shots(project, name, chosen)
        planned = restyle.billed_usd(todo, resolution)
        typer.echo(f"estimate: up to ${planned:.2f} for {_count(len(todo), 'shot')} still to make at {resolution}")
        if not dry_run and not cost.within_cap(planned) and not yes:
            typer.echo(f"error: that is above the ${config.CAP_USD:g} auto-run cap; pass --yes to run it", err=True)
            raise typer.Exit(1)
        record = restyle.full(project, name, audio_from, resolution, workers=workers, dry_run=dry_run, only=chosen)
    except (ProjectError, restyle.RestyleError, google.GoogleError, MediaError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote projects/{project.slug}/{record['output']} ({record['duration']:.1f}s)")
    typer.echo(f"  {record['restyled']} of {record['shots']} shots restyled, {len(record['failed'])} failed this run, ${record['usd_this_run']:.2f} spent this run")
    for number, why in record["failed"].items():
        typer.echo(f"  shot {number} kept its original picture: {why[:140]}")
    if record.get("regenerated_for_script"):
        typer.echo(f"  regenerated because their line or their correction in look.json changed: shots {record['regenerated_for_script']}")
    if record.get("pending"):
        # A pilot: most of the picture is still the source, so there is no finished video to check yet.
        typer.echo(f"  {_count(len(record['pending']), 'shot')} still show the original picture: this is a pilot, not checked or approvable yet")
        return
    if defer_check and not dry_run:
        typer.echo("NOT verified: the check was deferred. Run `remix verify` before this video is handed over or approved.")
        raise typer.Exit(UNVERIFIED)
    if not dry_run and not _check(project, name):
        raise typer.Exit(UNVERIFIED)


@app.command("restyle-pilot")
def restyle_pilot(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    name: str = typer.Option(..., help="Short name for this pilot."),
    shots: str = typer.Option(..., help="Comma-separated shot numbers, each 3 to 15 s long."),
    style: str = typer.Option(..., help="The new look, and what must stay the same."),
    resolution: str = typer.Option("360p", help="360p, 720p or 1080p."),
    yes: bool = typer.Option(False, "--yes", help="Run even if the estimate is above the auto-run cap."),
) -> None:
    """Restyle a few shots with Gemini Omni and lay source and result frames side by side (paid)."""
    project = _project(target)
    try:
        numbers = [int(n) for n in shots.split(",") if n.strip()]
        planned = cost.total(restyle.estimate(restyle.chosen_shots(project, numbers), resolution))
        typer.echo(f"estimate: up to ${planned:.2f} for {_count(len(numbers), 'shot')} at {resolution}")
        if not cost.within_cap(planned) and not yes:
            typer.echo(f"error: that is above the ${config.CAP_USD:g} auto-run cap; pass --yes to run it", err=True)
            raise typer.Exit(1)
        record = restyle.pilot(project, name, numbers, style, resolution)
    except (ValueError, ProjectError, restyle.RestyleError, google.GoogleError, MediaError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote projects/{project.slug}/remixes/{slugify(name)}/{record['comparison']}; logged up to ${record['usd']:.2f}")
    typer.echo("  reported usage: " + json.dumps(record["shots"][0]["usage"])[:300])


@cost_app.command("prices")
def cost_prices(on: Optional[str] = typer.Option(None, help="Date as YYYY-MM-DD; today when omitted.")) -> None:
    """List the prices that apply on a date."""
    day = _day(on)
    table = cost.load_table()
    typer.echo(f"prices on {day}:")
    for price in table.current(day):
        typer.echo(f"  {price['id']:<32} ${price['usd']:<8g} per {price['unit']:<10} {price['model']}")
    for name, source in table.sources.items():
        typer.echo(f"source {name}: {source['url']} (read {source['read']})")


@cost_app.command("estimate")
def cost_estimate(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    on: Optional[str] = typer.Option(None, help="Date as YYYY-MM-DD; today when omitted."),
) -> None:
    """Price every remix route for one video and save the sheet as cost_estimate.md."""
    project = _project(target)
    day = _day(on)
    try:
        options = cost.estimate_project(project, day)
    except cost.CostError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    sheet = project.root / "cost_estimate.md"
    sheet.write_text(cost.render_options(project.slug, options, day), encoding="utf-8")
    for option in options:
        flag = "" if cost.within_cap(option.usd) else "  (over the cap: asks first)"
        typer.echo(f"  {option.mode:<24}{option.name:<36}${option.usd:>7.2f}{flag}")
    typer.echo(f"saved projects/{project.slug}/cost_estimate.md")


@cost_app.command("report")
def cost_report(by: str = typer.Option("provider", help="Group by: " + ", ".join(cost.SUMMARY_KEYS))) -> None:
    """Total what has been spent so far, from the ledger, and what is left of any recorded credit."""
    rows = cost.entries()
    try:
        groups = cost.summarize(rows, by)
    except cost.CostError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    if not rows:
        typer.echo("the ledger is empty: nothing has been spent yet")
    for key, usd, calls in groups:
        typer.echo(f"  {key:<32}${usd:>9.4f}  {_count(calls, 'call')}")
    if rows:
        typer.echo(f"  {'total':<32}${sum(usd for _, usd, _ in groups):>9.4f}  {_count(len(rows), 'call')}")
    credit = cost.credit_status(rows, date.today())
    if credit:
        typer.echo(
            f"credit: {credit['name']}: ${credit['left']:.2f} left of ${credit['usd']:.2f} "
            f"(as read {credit['as_of']}), ends {credit['ends']} ({_count(credit['days_left'], 'day')} left)"
        )


@cost_app.command("credit")
def cost_credit(
    usd: float = typer.Option(..., help="Balance in dollars as shown by the provider."),
    ends: str = typer.Option(..., help="Expiry date as YYYY-MM-DD."),
    name: str = typer.Option("credit", help="What the provider calls it."),
    provider: str = typer.Option("google", help="Provider whose spend draws on it."),
    as_of: Optional[str] = typer.Option(None, help="Date the balance was read; today when omitted."),
) -> None:
    """Record a provider credit balance so `cost report` can show what is left."""
    credit = cost.set_credit(name, provider, usd, _day(ends), _day(as_of))
    typer.echo(f"recorded {credit['name']}: ${credit['usd']:.2f} as of {credit['as_of']}, ends {credit['ends']}")


@blueprint_app.command("init")
def blueprint_init(
    target: str = typer.Argument(..., help="Project slug or a unique prefix of it."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing blueprint.json."),
) -> None:
    """Write blueprint.json with measurements and shot timings filled in, ready to be completed."""
    project = _project(target)
    try:
        result = bp.init(project, force=force)
    except (bp.BlueprintError, MediaError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote projects/{project.slug}/blueprint.json ({_count(len(result['shots']), 'shot')} to describe)")
    typer.echo(json.dumps(result["metrics"], indent=2))
    typer.echo("next: read overview.md and shots/sheets/*.jpg, fill in the empty fields, then `remix blueprint check`")


@blueprint_app.command("check")
def blueprint_check(target: str = typer.Argument(..., help="Project slug or a unique prefix of it.")) -> None:
    """Validate blueprint.json and, once it is complete, render blueprint.md."""
    project = _project(target)
    try:
        found = bp.check(project)
    except (bp.BlueprintError, MediaError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    if found:
        typer.echo(f"{_count(len(found), 'thing')} to fix in projects/{project.slug}/blueprint.json:")
        for item in found:
            typer.echo(f"  - {item}")
        raise typer.Exit(1)
    typer.echo(f"blueprint complete: projects/{project.slug}/blueprint.md")
