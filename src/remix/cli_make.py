"""`remix make ...`: make a video from a prompt (see make.py)."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from . import cost, elevenlabs, google, make, voices
from .compose import ComposeError
from .media import MediaError

make_app = typer.Typer(no_args_is_help=True, help="Make a video from a prompt: plan, Veo shots, voice, music, compose, check.")


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _find(ref: str):
    try:
        return make.find(ref)
    except make.MakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


def _plan_report(project, plan, problems) -> None:
    scenes = make.schedule(plan)
    typer.echo(f"{plan.get('title', project.slug)}: {_count(len(scenes), 'scene')}, {scenes[-1]['end']:g} s, {_count(len(plan.get('shots', {})), 'generated shot')}")
    for tier in make.TIERS:
        typer.echo(f"  Veo shots at the {tier} tier: ${cost.total(make.estimate_shots(plan, tier)):.2f}")
    if problems:
        typer.echo(f"{_count(len(problems), 'problem')} in the plan:")
        for item in problems:
            typer.echo(f"  - {item}")
    else:
        typer.echo("the plan is valid")


@make_app.command("new")
def make_new(
    title: str = typer.Argument(..., help="A title; it names the project."),
    prompt: str = typer.Option("", help="What the video is about and what it should do."),
    seconds: float = typer.Option(30.0, help="Length in seconds (6 to 600)."),
    aspect: str = typer.Option("16:9", help="Aspect ratios to render, comma separated: 16:9, 1:1, 4:5, 9:16."),
    style: str = typer.Option("", help="The look: photoreal and cinematic when empty."),
    brand: str = typer.Option("", help="A brand to show or name in the first five seconds."),
    cta: str = typer.Option("", help="The call to action to end on."),
    audience: str = typer.Option("", help="Who it is for."),
    voice: str = typer.Option("Kore", help="One of Google's stock voices."),
) -> None:
    """Start a make project (free)."""
    try:
        project = make.create(title, {"prompt": prompt, "seconds": seconds, "aspects": [a.strip() for a in aspect.split(",")], "style": style,
                                      "brand": brand, "cta": cta, "audience": audience, "voice": voice, "brand_in_hook": bool(brand)})
    except make.MakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"made projects/{project.slug}: put pictures and logos in assets/, then `remix make plan {project.slug} --draft` or --file")


@make_app.command("plan")
def make_plan(
    target: str = typer.Argument(..., help="Make project slug."),
    draft: bool = typer.Option(False, "--draft", help="Have Gemini draft the plan from the brief (a few cents)."),
    file: str = typer.Option("", "--file", help="Use this plan.json instead (a path under the data folder)."),
) -> None:
    """Draft or load the plan, check it, and price the shots."""
    project = _find(target)
    try:
        if draft:
            plan, problems = make.draft_plan(project)
        elif file:
            plan = json.loads(Path(file).read_text(encoding="utf-8"))
            problems = make.save_plan(project, plan)
        else:
            plan = make.load_plan(project)
            problems = make.validate(project, plan)
    except (make.MakeError, google.GoogleError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    _plan_report(project, plan, problems)
    if problems:
        raise typer.Exit(1)


@make_app.command("shots")
def make_shots(
    target: str = typer.Argument(..., help="Make project slug."),
    tier: str = typer.Option("pilot", help="pilot (Veo 3.1 Lite 720p), final (Fast 1080p) or hero (Veo 3.1 1080p)."),
    only: str = typer.Option("", help="Just these shot ids, such as v1,v2."),
    redo: bool = typer.Option(False, "--redo", help="Generate again even where a clip exists (paid)."),
    workers: int = typer.Option(3, help="Clips made at once."),
    yes: bool = typer.Option(False, "--yes", help="Run even if the estimate is above the auto-run cap."),
) -> None:
    """Generate the shots with Veo and have Gemini check each clip (paid, logged)."""
    project = _find(target)
    if tier not in make.TIERS:
        typer.echo(f"error: tier must be one of {', '.join(make.TIERS)}", err=True)
        raise typer.Exit(2)
    try:
        chosen = [i for i in only.replace(" ", "").split(",") if i] or None
        plan = make.load_plan(project)
        todo = [s for s in plan["shots"] if (chosen is None or s in chosen) and (redo or not (project.root / "shots" / f"{s}.{tier}.mp4").is_file())]
        typer.echo(f"estimate: ${cost.total(make.estimate_shots(plan, tier, todo)):.2f} for {_count(len(todo), 'clip')} at the {tier} tier")
        result = make.generate_shots(project, tier, chosen, workers=workers, yes=yes, redo=redo)
    except (make.MakeError, google.GoogleError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"made {_count(len(result['made']), 'clip')}: {', '.join(result['made']) or 'none'}")
    for shot_id, why in result["failed"].items():
        typer.echo(f"  shot {shot_id} failed: {why}")
    for item in make.qa_problems(project, plan):
        typer.echo(f"  note: {item}")
    if result["failed"]:
        raise typer.Exit(1)


@make_app.command("voice")
def make_voice(
    target: str = typer.Argument(..., help="Make project slug."),
    pilot: int = typer.Option(0, help="Speak only the first N lines, as a cheap check."),
    redo: str = typer.Option("", help="Line ids to speak again."),
    provider: str = typer.Option("", help="gemini or elevenlabs; remembered for this project. Empty keeps the project's choice."),
) -> None:
    """Speak the plan's voice-over (Gemini: paid, cents) or bring in the takes made in ElevenLabs (free), fitted to its scenes."""
    project = _find(target)
    try:
        if provider:
            make.set_audio_provider(project, provider)
        result = make.speak(project, pilot=pilot, redo=tuple(int(n) for n in redo.replace(" ", "").split(",") if n))
    except (make.MakeError, voices.VoiceError, google.GoogleError, elevenlabs.ElevenLabsError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    came = "brought in from ElevenLabs" if result.get("provider") == "elevenlabs" else "spoken now"
    typer.echo(f"{_count(result['lines'], 'line')}: {result['respoken']} {came}, {result['sped_up']} sped up, {result['slowed']} slowed, ${result['usd']:.4f}")


@make_app.command("music")
def make_music(
    target: str = typer.Argument(..., help="Make project slug."),
    long_model: bool = typer.Option(False, "--long", help="Use the full-length song model even for 30 seconds or less."),
    provider: str = typer.Option("", help="gemini (Lyria) or elevenlabs (the file saved as eleven/music.mp3); remembered for this project."),
) -> None:
    """Generate the score with Lyria (paid, cents), or bring in the score made in ElevenLabs (free)."""
    project = _find(target)
    try:
        if provider:
            make.set_audio_provider(project, provider)
        path = make.score(project, long_model=long_model)
    except (make.MakeError, google.GoogleError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote projects/{project.slug}/{project.rel(path)}")


@make_app.command("eleven")
def make_eleven(
    voice: str = typer.Option("", help="Also look up this voice (a name from your ElevenLabs account, or a voice id)."),
) -> None:
    """Check the ElevenLabs API connection through the official SDK. Reads only: nothing is spoken or spent."""
    state = elevenlabs.status()
    if not state["connected"]:
        typer.echo(f"not connected: {state['reason']}")
        raise typer.Exit(1)
    left = state["credits_left"]
    typer.echo(f"connected: model {state['model']}, " + (f"{left:,} characters of credit left" if left is not None else "credit not readable"))
    if voice:
        try:
            typer.echo(f"voice '{voice}' is {elevenlabs.find_voice(voice)}")
        except elevenlabs.ElevenLabsError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1)


@make_app.command("order")
def make_order(
    target: str = typer.Argument(..., help="Make project slug."),
    provider: str = typer.Option("", help="gemini or elevenlabs; remembered for this project."),
) -> None:
    """Write the sheet of what has to be made in ElevenLabs (every voice line and the music) and say what is still missing. Free."""
    project = _find(target)
    try:
        if provider:
            make.set_audio_provider(project, provider)
        order = make.audio_order(project)
    except make.MakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    if order["provider"] != "elevenlabs":
        typer.echo("this project's audio comes from Gemini: there is nothing to make by hand")
        return
    typer.echo(f"wrote projects/{project.slug}/{order['folder']}/order.md")
    typer.echo(f"{_count(len(order['lines']) - len(order['missing']), 'line')} of {len(order['lines'])} made; music: {'made' if order['music']['have'] else 'to make'}")
    for row in order["lines"]:
        if not row["have"]:
            typer.echo(f"  {row['id']:>3}  {row['voice']}: {row['text']}")


@make_app.command("render")
def make_render(
    target: str = typer.Argument(..., help="Make project slug."),
    aspect: str = typer.Option("", help="One aspect ratio; every aspect in the brief when empty."),
    preview: bool = typer.Option(False, "--preview", help="Allow pilot-tier clips in the picture."),
) -> None:
    """Draw the scenes, mix the sound, burn the captions (free)."""
    project = _find(target)
    try:
        for ratio in [aspect] if aspect else make.brief_of(project)["aspects"]:
            typer.echo(f"rendering {ratio}...")
            final = make.render(project, ratio, preview=preview)
            typer.echo(f"  wrote projects/{project.slug}/{project.rel(final)}")
    except (make.MakeError, voices.VoiceError, ComposeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


@make_app.command("check")
def make_check(
    target: str = typer.Argument(..., help="Make project slug."),
    aspect: str = typer.Option("16:9", help="Which render to check."),
    frames_ok: bool = typer.Option(False, "--frames-ok", help="Say that the contact sheet was read."),
) -> None:
    """Check a render: length, words heard, captions, loudness, hook, clip QA."""
    project = _find(target)
    try:
        result = make.check(project, aspect, frames_ok=frames_ok)
    except (make.MakeError, MediaError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"loudness {result['loudness']['lufs']} LUFS, peak {result['loudness']['true_peak_db']} dB; contact sheet: projects/{project.slug}/{result['sheet']}")
    if result["passed"]:
        typer.echo("checked: the video matches its plan and script")
        return
    typer.echo(f"NOT checked: {_count(len(result['problems']), 'problem')}")
    for item in result["problems"]:
        typer.echo(f"  - {('line ' + str(item['line']) + ': ') if 'line' in item else ''}{item['problem']}")
    raise typer.Exit(1)
