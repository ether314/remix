"""A local web app for walking a video through the remix gates.

Standard library only. It is a control panel over the same functions the CLI uses: long steps run
as `remix ...` subprocesses, one at a time, and the page polls for their progress. It is meant to
be published on 127.0.0.1 only; paid actions also require a custom header so that another site
open in the same browser cannot trigger them.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import blueprint as bp
from . import config, cost, dialogue, elevenlabs, google, make, restyle, verify, voices
from .project import Project, ProjectError, find, ingest, list_projects, now_iso, slugify

PORT = int(os.environ.get("REMIX_WEB_PORT", "8765"))
RESOLUTION = "360p"  # restyles are drafted at 360p; a 720p pass costs about 2.5 times as much and is not offered here yet
PILOT_SHOTS = 2  # shots that must be restyled and looked at before the app will run all the rest
PAGE = Path(__file__).with_name("webapp.html")
VIDEO_TYPES = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")
MAX_UPLOAD = 4 * 1024**3
GATES = ("decompose", "blueprint")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
STARTED = time.time()


class Refused(Exception):
    """A request the app will not carry out; the message is shown to the user."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


class Jobs:
    """Background `remix` commands. Only one runs at a time; the rest wait their turn."""

    def __init__(self) -> None:
        self.table: dict[str, dict] = {}
        self._turn = threading.Lock()

    def start(self, label: str, project: str, args: list[str], usd: float = 0.0) -> dict:
        job = {
            "id": uuid.uuid4().hex[:10], "label": label, "project": project, "status": "queued",
            "log": [], "started": now_iso(), "ended": None, "usd": usd,
        }
        self.table[job["id"]] = job
        threading.Thread(target=self._run, args=(job, args), daemon=True).start()
        return job

    def _run(self, job: dict, args: list[str]) -> None:
        with self._turn:
            job["status"] = "running"
            proc = subprocess.Popen(
                ["remix", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
            )
            for line in proc.stdout:
                if line.strip():
                    job["log"] = (job["log"] + [line.rstrip()])[-40:]
            # Exit code 3: the video was made but did not pass its script check. The step itself is
            # done; the take shows the failed check, and approval stays refused until it passes.
            job["status"] = "done" if proc.wait() in (0, 3) else "failed"
            job["ended"] = now_iso()

    def busy(self, project: str) -> bool:
        return any(j["project"] == project and j["status"] in ("queued", "running") for j in self.table.values())


JOBS = Jobs()


def google_ready() -> bool:
    return bool(os.environ.get("GOOGLE_ACCESS_TOKEN", "").strip() and os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip())


def safe_upload_name(name: str) -> str:
    """A plain file name for the inbox, or a refusal if it is not a video we take."""
    path = Path(unquote(name).replace("\\", "/").split("/")[-1])
    if path.suffix.lower() not in VIDEO_TYPES:
        raise Refused(f"upload a video file ({', '.join(VIDEO_TYPES)})", 400)
    return f"{slugify(path.stem)}{path.suffix.lower()}"


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """(first byte, last byte) for a Range header, or None when the whole file should be sent."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
    if not match or not (match.group(1) or match.group(2)):
        return None
    if match.group(1):
        start = int(match.group(1))
        end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
    else:
        start, end = max(size - int(match.group(2)), 0), size - 1
    if start > end or start >= size:
        raise Refused("requested range is outside the file", 416)
    return start, end


def media_path(project: Project, relative: str) -> Path:
    """A file inside the project folder; anything that escapes the folder is refused."""
    root = project.root.resolve()
    target = (root / unquote(relative)).resolve()
    if root not in target.parents or not target.is_file():
        raise Refused("no such file", 404)
    return target


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def summary(project: Project) -> dict:
    manifest = project.load()
    source = manifest["source"]
    return {
        "slug": project.slug,
        "name": source["original_name"],
        "duration": source["duration"],
        "rights": project.rights(),
        "steps": {name: step.get("status") for name, step in manifest["steps"].items()},
        "busy": JOBS.busy(project.slug),
    }


_GATES: dict[tuple, list[str]] = {}


def dialogue_gate(project: Project, remix: Path) -> list[str] | None:
    """The dialogue gate's problems for a take with a script (None without one), remembered until its files change."""
    if not (remix / "lines.json").is_file():
        return None
    watched = [remix / "lines.json", remix / "look.json", remix / "voice.json", project.root / "blueprint.json", project.root / "timeline.json"]
    look = _read(remix / "look.json") or {}
    if look.get("audio_from"):
        watched.append(project.root / "remixes" / slugify(look["audio_from"]) / "voice.json")
    key = (str(remix), tuple(f.stat().st_mtime if f.is_file() else 0 for f in watched))
    if key not in _GATES:
        try:
            _GATES[key] = dialogue.problems(project, remix.name)
        except (voices.VoiceError, OSError, ValueError, KeyError) as exc:
            _GATES[key] = [f"the dialogue map could not be built: {exc}"]
    return _GATES[key]


def remix_views(project: Project, rows: list[dict]) -> list[dict]:
    folder = project.root / "remixes"
    views = []

    def newest_track(remix: Path) -> float:
        return max((f.stat().st_mtime for f in remix.glob("music.*")), default=0.0)

    takes = [r for r in folder.iterdir() if r.is_dir()] if folder.is_dir() else []
    for remix in sorted(takes, key=newest_track, reverse=True):  # newest take first
        record = _read(remix / "remix.json") or {}
        track = next((f for f in sorted(remix.glob("music.*")) if f.suffix in (".mp3", ".wav")), None)
        prompt, notes = remix / "music_prompt.txt", remix / "music_notes.txt"
        voice, voice_report = remix / "voice.wav", _read(remix / "voice.json")
        look = _read(remix / "look.json")
        clips = remix / "shots"
        views.append(
            {
                **record,
                "name": remix.name,
                "look": {k: look.get(k) for k in ("style", "audio_from", "characters")} if look else None,
                "restyled_shots": sorted(int(f.name[:4]) for f in clips.glob("[0-9][0-9][0-9][0-9]_restyled.mp4")) if clips.is_dir() else [],
                "check_sheets": [project.rel(f) for f in sorted((remix / "check").glob("sheet_*.jpg"))],
                "dialogue": dialogue_gate(project, remix),
                "track": project.rel(track) if track else None,
                "voice": project.rel(voice) if voice.is_file() else None,
                "check": _read(remix / "verify.json") if verify.passed(remix) or (remix / "verify.json").is_file() else None,
                "check_current": verify.passed(remix),
                "voice_summary": {k: voice_report.get(k) for k in ("lines", "sped_up", "overrun", "suspect", "usd", "voices")}
                if voice_report
                else None,
                "prompt": prompt.read_text(encoding="utf-8").strip() if prompt.is_file() else "",
                "notes": notes.read_text(encoding="utf-8")[:6000] if notes.is_file() else "",
                "spent": round(sum(r["usd"] for r in rows if r["project"] == project.slug and r["remix"] == remix.name), 4),
            }
        )
    return views


def project_view(project: Project) -> dict:
    manifest = project.load()
    root = project.root
    timeline = _read(root / "timeline.json")
    transcript = _read(root / "transcript" / "transcript.json")
    blueprint = _read(root / "blueprint.json")
    rows = cost.entries()
    estimate, music_usd = [], None
    if timeline:
        today = date.today()
        estimate = [
            {"mode": o.mode, "name": o.name, "usd": o.usd, "note": o.note, "within_cap": cost.within_cap(o.usd)}
            for o in cost.estimate_project(project, today)
        ]
        music_usd = cost.total(cost.music(cost.load_table(), today, manifest["source"]["duration"]))
    draft = root / "music_prompt_draft.txt"
    return {
        **summary(project),
        "source": manifest["source"],
        "source_file": manifest["source"]["file"],
        "step_details": manifest["steps"],
        "gates": manifest.get("gates", {}),
        "stems": [f"audio/{n}" for n in ("speech.wav", "background.wav") if (root / "audio" / n).is_file()],
        "script": transcript["text"] if transcript else None,
        "dropped": transcript.get("dropped_as_silence", []) if transcript else [],
        "shots": timeline["shots"] if timeline else [],
        "sheets": [s["file"] for s in timeline["sheets"]] if timeline else [],
        "blueprint": blueprint,
        "blueprint_problems": bp.problems(blueprint, timeline, manifest["source"]["duration"])
        if blueprint and timeline
        else None,
        "estimate": estimate,
        "music_usd": music_usd,
        "voices_usd": next((o["usd"] for o in estimate if o["name"] == "Gemini 3.8 Flash TTS"), None),
        "music_prompt_draft": draft.read_text(encoding="utf-8").strip() if draft.is_file() else "",
        # Enough for the page to price any choice of shots: the rate, and the 3 s minimum per clip.
        "restyle": {
            "resolution": RESOLUTION,
            "usd_per_second": round(restyle.billed_usd([{"duration": 100.0}], RESOLUTION) / 100, 5),
            "clip_seconds": list(cost.load_table().limits["omni.clip-seconds"]),
        } if timeline else None,
        "remixes": remix_views(project, rows),
    }


def state() -> dict:
    rows = cost.entries()
    return {
        "projects": [summary(p) for p in list_projects() if not make.is_make(p)],
        "makes": [make_summary(p) for p in make.make_projects()],
        "voices": list(google.VOICES),
        "cap": config.CAP_USD,
        "credit": cost.credit_status(rows, date.today()),
        "spent": round(sum(r["usd"] for r in rows), 4),
        "by_remix": cost.summarize(rows, "remix"),
        "by_day": cost.summarize(rows, "day"),
        "google": {
            "ready": google_ready(),
            "project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            "minutes_running": int((time.time() - STARTED) / 60),
        },
        "jobs": sorted(JOBS.table.values(), key=lambda j: j["started"], reverse=True)[:8],
    }


# --- make a video from a prompt -----------------------------------------------------------------


def make_summary(project: Project) -> dict:
    brief = make.brief_of(project)
    return {"slug": project.slug, "title": brief.get("title", project.slug), "seconds": brief.get("seconds"),
            "aspects": brief.get("aspects"), "busy": JOBS.busy(project.slug)}


def audio_view(project: Project, plan: dict | None) -> dict:
    """Where the voice and music come from, and for ElevenLabs what is still to be made there."""
    provider = make.audio_provider(project)
    out: dict = {"provider": provider, "providers": list(elevenlabs.PROVIDERS), "api": elevenlabs.api_ready()}
    if provider == "elevenlabs" and plan and any(s.get("vo") for s in plan.get("scenes", [])):
        try:
            order = make.audio_order(project, write=False)
        except make.MakeError:
            return out
        out |= {"lines": order["lines"], "missing": order["missing"], "folder": order["folder"], "music": order["music"]}
    return out


def make_set_provider(project: Project, body: dict) -> dict:
    try:
        return {"provider": make.set_audio_provider(project, str(body.get("provider", "")))}
    except make.MakeError as exc:
        raise Refused(str(exc), 400) from None


def make_view(project: Project) -> dict:
    """Everything the page needs to show one make project and decide what is next."""
    brief, root = make.brief_of(project), project.root
    plan_path = root / "plan.json"
    plan = _read(plan_path)
    rows = cost.entries()
    view: dict = {
        **make_summary(project), "brief": brief, "plan": plan,
        "plan_text": plan_path.read_text(encoding="utf-8") if plan_path.is_file() else "",
        "spent": round(sum(r["usd"] for r in rows if r["project"] == project.slug), 4),
        "google_ready": google_ready(), "tiers": list(make.TIERS), "problems": [], "estimates": {}, "shots": [], "scenes": [],
    }
    if plan:
        view["problems"] = make.validate(project, plan)
        view["estimates"] = {t: round(cost.total(make.estimate_shots(plan, t)), 2) for t in make.TIERS}
        view["scenes"] = [{"id": s["id"], "kind": s["kind"], "start": s["start"], "end": s["end"], "vo": s.get("vo", "")} for s in make.schedule(plan)]
        for sid, shot in plan.get("shots", {}).items():
            best = make.best_clip(project, sid)
            qa = _read(best.with_suffix(".qa.json")) if best else None
            view["shots"].append({"id": sid, "seconds": shot["seconds"], "prompt": shot["prompt"], "tiers": make.tiers_present(project, sid),
                                  "clip": project.rel(best) if best else None, "qa": qa})
    view["audio"] = audio_view(project, plan)
    folder = voices.folder(project, make.REMIX)
    voice_report = _read(folder / "voice.json")
    view["voice"] = ({k: voice_report.get(k) for k in ("lines", "sped_up", "slowed", "usd", "voices")} | {"file": project.rel(folder / "voice.wav")}) if voice_report else None
    music = next(iter(sorted(root.glob("music.*"))), None)
    view["music"] = project.rel(music) if music else None
    view["renders"] = {}
    for aspect in brief.get("aspects", []):
        out = root / "out" / aspect.replace(":", "x")
        final = out / "final.mp4"
        view["renders"][aspect] = {
            "final": project.rel(final) if final.is_file() else None, "check": _read(out / "check.json"),
            "sheet": project.rel(out / "check_sheet.jpg") if (out / "check_sheet.jpg").is_file() else None,
            "captions": project.rel(out / "captions.srt") if (out / "captions.srt").is_file() else None,
        }
    return view


def make_new(body: dict) -> dict:
    title = " ".join(str(body.get("title", "")).split())
    if not title:
        raise Refused("give the video a title", 400)
    brief = {k: body.get(k) for k in ("prompt", "style", "brand", "cta", "audience", "voice", "elevenlabs_voice")}
    try:
        brief["audio_provider"] = elevenlabs.provider_of(str(body.get("audio_provider") or "gemini"))
    except elevenlabs.ElevenLabsError as exc:
        raise Refused(str(exc), 400) from None
    brief |= {"seconds": float(body.get("seconds") or 30), "aspects": body.get("aspects") or ["16:9"], "brand_in_hook": bool(body.get("brand"))}
    try:
        project = make.create(title, brief)
    except (make.MakeError, ValueError) as exc:
        raise Refused(str(exc), 400) from None
    return {"slug": project.slug}


def make_save_plan(project: Project, text: str) -> dict:
    try:
        plan = json.loads(text)
    except ValueError as exc:
        raise Refused(f"the plan is not valid JSON: {exc}", 400) from None
    if not isinstance(plan, dict) or not isinstance(plan.get("scenes"), list):
        raise Refused("the plan needs a list of scenes", 400)
    return {"problems": make.save_plan(project, plan)}


def make_job(project: Project, action: str, body: dict) -> dict:
    """Start one step of a make project as a job. Paid steps need Google access and respect the cap."""
    args = ["make", action, project.slug]
    usd = 0.0
    by_hand = action in ("voice", "music") and make.audio_provider(project) == "elevenlabs"  # made in ElevenLabs: free here, nothing to pay Google
    paid = action in ("plan", "shots", "voice", "music") and not by_hand
    if paid and not google_ready():
        raise Refused("Google access is not set up for this run: start the app with start-web.cmd")
    try:
        if action == "plan":
            args.append("--draft")
            usd = 0.01
        elif action == "shots":
            tier = body.get("tier", "pilot")
            if tier not in make.TIERS:
                raise Refused("unknown tier", 400)
            plan = make.load_plan(project)
            only = [str(i) for i in body.get("only") or []]
            todo = [s for s in plan["shots"] if (not only or s in only) and (body.get("redo") or not (project.root / "shots" / f"{s}.{tier}.mp4").is_file())]
            if not todo:
                raise Refused("every shot already has a clip at this tier")
            if tier != "pilot" and any("pilot" not in make.tiers_present(project, s) and not body.get("redo") for s in todo):
                raise Refused("pilot the shots first (the cheapest tier): the full-quality run is the expensive step")
            problems = make.validate(project, plan)
            if problems:
                raise Refused("fix the plan first: " + problems[0])
            usd = cost.total(make.estimate_shots(plan, tier, todo))
            if not cost.within_cap(usd) and not body.get("approve_over_cap"):
                raise Refused(f"${usd:.2f} is over the ${config.CAP_USD:g} cap: tick the approval box to run it")
            args += ["--tier", tier] + (["--only", ",".join(only)] if only else []) + (["--redo"] if body.get("redo") else []) + (["--yes"] if not cost.within_cap(usd) else [])
        elif action == "voice":
            plan = make.load_plan(project)
            if make.validate(project, plan):
                raise Refused("fix the plan first: " + make.validate(project, plan)[0])
            if by_hand and not elevenlabs.api_ready():  # with the API connected, the step makes the missing takes itself
                missing = make.audio_order(project, write=False)["missing"]
                if missing:
                    raise Refused(f"{len(missing)} voice line(s) are not made in ElevenLabs yet: ask Claude to make them there, save the takes yourself, "
                                  f"or set {elevenlabs.KEY_ENV} so the app can")
            elif not by_hand:
                usd = 0.04
        elif action == "music":
            if by_hand and not elevenlabs.api_ready() and elevenlabs.music_file(project) is None:
                raise Refused("the music is not made in ElevenLabs yet: ask Claude to make it there, save it as eleven/music.mp3 in the project, "
                              f"or set {elevenlabs.KEY_ENV} so the app can")
            usd = 0.0 if by_hand else 0.08
        elif action == "render":
            if body.get("aspect"):
                args += ["--aspect", str(body["aspect"])]
            if body.get("preview"):
                args.append("--preview")
        elif action == "check":
            args += ["--aspect", str(body.get("aspect") or "16:9")] + (["--frames-ok"] if body.get("frames_ok") else [])
        else:
            raise Refused("not found", 404)
    except make.MakeError as exc:
        raise Refused(str(exc)) from None
    return JOBS.start(f"{action} for '{project.slug}'", project.slug, args, usd)


# --- actions ------------------------------------------------------------------------------------


def set_gate(project: Project, gate: str, approved: bool) -> dict:
    if gate not in GATES:
        raise Refused(f"unknown gate '{gate}'", 400)
    if gate == "blueprint" and approved:
        view = project_view(project)
        if view["blueprint"] is None or view["blueprint_problems"]:
            raise Refused("the blueprint is not finished, so there is nothing to approve yet")
    manifest = project.load()
    gates = manifest.setdefault("gates", {})
    if approved:
        gates[gate] = now_iso()
    else:
        gates.pop(gate, None)
    project.save(manifest)
    return gates


def start_music(project: Project, name: str, prompt: str, approve_over_cap: bool) -> dict:
    name, prompt = slugify(name), " ".join(prompt.split())
    if not prompt:
        raise Refused("write a prompt for the music first", 400)
    if not google_ready():
        raise Refused("Google access is not set up for this run: start the app with start-web.cmd")
    if any((project.root / "remixes" / name).glob("music.*")):
        raise Refused(f"'{name}' already has a track; pick a new name so the paid one is kept")
    usd = cost.total(cost.music(cost.load_table(), date.today(), project.load()["source"]["duration"]))
    over = not cost.within_cap(usd)
    if over and not approve_over_cap:
        raise Refused(f"${usd:.2f} is over the ${config.CAP_USD:g} cap: tick the approval box to run it")
    args = ["music", project.slug, "--name", name, "--prompt", prompt] + (["--yes"] if over else [])
    return JOBS.start(f"generate music '{name}'", project.slug, args, usd)


def start_voices(project: Project, name: str, approve_over_cap: bool) -> dict:
    """Speak the remix's lines with AI voices. The line list is created from the transcript if missing."""
    name = slugify(name)
    if not google_ready():
        raise Refused("Google access is not set up for this run: start the app with start-web.cmd")
    try:
        if not (voices.folder(project, name) / "lines.json").is_file():
            voices.write_lines(project, name)
        script = voices.load_lines(project, name)
        usd = cost.total(voices.estimate(script, date.today()))
    except voices.VoiceError as exc:
        raise Refused(str(exc)) from None
    if sum(1 for line in script["lines"] if voices.cues(line)) < voices.PERFORMED * len(script["lines"]):
        raise Refused(
            "this script has no vocal cues yet (sighs, chuckles, pauses), so the voices would come out flat. "
            f"In Claude Code, say: Add vocal cues to the lines of '{name}' for {project.slug}"
        )
    over = not cost.within_cap(usd)
    if over and not approve_over_cap:
        raise Refused(f"${usd:.2f} is over the ${config.CAP_USD:g} cap: tick the approval box to run it")
    args = ["voices", project.slug, "--name", name] + (["--yes"] if over else [])
    return JOBS.start(f"generate voices for '{name}'", project.slug, args, usd)


def start_mix(project: Project, name: str, new_voices: bool = False) -> dict:
    name = slugify(name)
    try:
        project.require_reusable()
    except ProjectError as exc:
        raise Refused(str(exc)) from None
    track = next((f for f in sorted((project.root / "remixes" / name).glob("music.*"))), None)
    if track is None:
        raise Refused(f"'{name}' has no music track yet")
    if new_voices and not (project.root / "remixes" / name / "voice.wav").is_file():
        raise Refused(f"'{name}' has no AI voices yet")
    args = ["audio-remix", project.slug, "--name", name, "--music", f"projects/{project.slug}/{project.rel(track)}"]
    label = f"mix '{name}' with {'new' if new_voices else 'original'} voices"
    return JOBS.start(label, project.slug, args + (["--new-voices"] if new_voices else []))


def save_look(project: Project, name: str, audio_from: str, style: str, characters: dict) -> dict:
    """Start a restyle: write its look.json and give it the script of the take whose audio it will use. Free."""
    name, audio_from, style = slugify(name), slugify(audio_from), " ".join(str(style).split())
    try:
        project.require_reusable()
    except ProjectError as exc:
        raise Refused(str(exc)) from None
    if not name or not style:
        raise Refused("give the restyle a name and describe the style", 400)
    source = project.root / "remixes" / audio_from
    if not audio_from or name == audio_from or not (source / "output.mp4").is_file():
        raise Refused("pick a mixed take to supply the sound; the restyle needs a name of its own")
    folder = project.root / "remixes" / name
    if any((folder / "shots").glob("*_restyled.mp4")):
        raise Refused(f"'{name}' already has restyled shots; pick a new name so the paid ones are kept")
    folder.mkdir(parents=True, exist_ok=True)
    look = {
        "style": style,
        "characters": {slugify(k): " ".join(str(v).split()) for k, v in (characters or {}).items() if str(v).strip()},
        "rename": {},
        "keep": restyle.DEFAULT_KEEP,
        "audio_from": audio_from,
    }
    (folder / "look.json").write_text(json.dumps(look, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        if (source / "lines.json").is_file():  # one script behind the sound, the captions and the picture
            (folder / "lines.json").write_bytes((source / "lines.json").read_bytes())
        else:
            voices.write_lines(project, name, force=True)
    except voices.VoiceError as exc:
        raise Refused(str(exc)) from None
    return look


def start_restyle(project: Project, name: str, shots: list, approve_over_cap: bool) -> dict:
    """Restyle the chosen shots (a pilot), or every shot still missing when none are named."""
    name = slugify(name)
    folder = project.root / "remixes" / name
    if not google_ready():
        raise Refused("Google access is not set up for this run: start the app with start-web.cmd")
    try:
        project.require_reusable()
    except ProjectError as exc:
        raise Refused(str(exc)) from None
    if not (folder / "look.json").is_file():
        raise Refused(f"'{name}' has no look yet: describe the style first")
    try:
        only = tuple(sorted({int(n) for n in shots or []})) or None
        todo = restyle.pending_shots(project, name, only)
    except (ValueError, TypeError):
        raise Refused("shot numbers must be whole numbers", 400) from None
    if not todo:
        raise Refused("nothing to make: every chosen shot already has its restyled picture")
    closed = dialogue_gate(project, folder)
    if closed:
        raise Refused("the dialogue gate is closed, so no picture is generated yet: " + closed[0])
    made = len(list((folder / "shots").glob("*_restyled.mp4")))
    if only is None and made < PILOT_SHOTS:
        raise Refused(f"pilot {PILOT_SHOTS} or 3 shots and look at them first: running every shot is the expensive step")
    usd = restyle.billed_usd(todo, RESOLUTION)
    over = not cost.within_cap(usd)
    if over and not approve_over_cap:
        raise Refused(f"${usd:.2f} is over the ${config.CAP_USD:g} cap: tick the approval box to run it")
    args = ["restyle", project.slug, "--name", name, "--workers", "4"]
    args += (["--only", ",".join(str(n) for n in only)] if only else []) + (["--yes"] if over else [])
    label = f"restyle {'shots ' + ', '.join(str(n) for n in only) if only else str(len(todo)) + ' shots'} of '{name}'"
    return JOBS.start(label, project.slug, args, usd)


def confirm_frames(project: Project, name: str) -> dict:
    """The user says they looked at the frame sheets: run the check again with that on record."""
    name = slugify(name)
    record = _read(project.root / "remixes" / name / "remix.json") or {}
    if record.get("mode") != "restyle" or record.get("pending"):
        raise Refused("finish restyling every shot first; there is no complete video to check yet")
    if not any((project.root / "remixes" / name / "check").glob("sheet_*.jpg")):
        raise Refused("run the script check first so there are frame sheets to look at")
    return JOBS.start(f"check '{name}' with the frames confirmed", project.slug, ["verify", project.slug, "--name", name, "--frames-ok"])


def approve_remix(project: Project, name: str, approved: bool) -> dict:
    path = project.root / "remixes" / slugify(name) / "remix.json"
    record = _read(path)
    if not record or not record.get("output"):
        raise Refused("mix this remix before approving it")
    if approved and record.get("pending"):
        raise Refused(f"{len(record['pending'])} shots still show the original picture: restyle them before approving")
    if approved and not verify.passed(path.parent):
        raise Refused("this video has not passed the script check yet: press 'Check script match' first")
    if approved:
        record["approved"] = now_iso()
    else:
        record.pop("approved", None)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# --- HTTP ---------------------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "remix-web"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # keep the container log quiet
        pass

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _guard(self, writing: bool) -> None:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in LOCAL_HOSTS:
            raise Refused("this app only answers on localhost", 403)
        if writing and self.headers.get("X-Remix") != "1":
            raise Refused("missing X-Remix header", 403)

    def _make(self, ref: str) -> Project:
        try:
            return make.find(unquote(ref))
        except make.MakeError as exc:
            raise Refused(str(exc), 404) from None

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            raise Refused("request too large", 413)
        return json.loads(self.rfile.read(length) or b"{}")

    def _send_file(self, path: Path) -> None:
        size = path.stat().st_size
        span = parse_range(self.headers.get("Range"), size)
        start, end = span or (0, size - 1)
        self.send_response(206 if span else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if span:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = handle.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def _upload(self, query: dict) -> dict:
        name = safe_upload_name((query.get("name") or [""])[0])
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD:
            raise Refused("the upload is empty or larger than 4 GB", 413)
        config.INBOX_DIR.mkdir(parents=True, exist_ok=True)
        target = config.INBOX_DIR / name
        with target.open("wb") as out:
            left = length
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    raise Refused("the upload was cut off", 400)
                out.write(chunk)
                left -= len(chunk)
        try:
            project, _ = ingest(target)
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise Refused(f"that file could not be read as a video: {exc}", 400) from None
        job = JOBS.start("decompose", project.slug, ["decompose", project.slug])
        return {"slug": project.slug, "job": job}

    def _route(self, method: str) -> None:
        url = urlsplit(self.path)
        parts = [p for p in url.path.split("/") if p]
        self._guard(writing=method == "POST")

        if method == "GET":
            if not parts:
                page = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
            elif parts == ["api", "state"]:
                self._json(state())
            elif parts[:2] == ["api", "project"] and len(parts) == 3:
                self._json(project_view(find(parts[2])))
            elif parts[:2] == ["api", "make"] and len(parts) == 3:
                self._json(make_view(self._make(parts[2])))
            elif parts[:2] == ["api", "jobs"] and len(parts) == 3 and parts[2] in JOBS.table:
                self._json(JOBS.table[parts[2]])
            elif parts[:1] == ["media"] and len(parts) >= 3:
                self._send_file(media_path(find(parts[1]), "/".join(parts[2:])))
            else:
                raise Refused("not found", 404)
            return

        if parts == ["api", "upload"]:
            self._json(self._upload(parse_qs(url.query)))
            return
        if parts == ["api", "make", "new"]:
            self._json(make_new(self._body()))
            return
        if parts[:2] == ["api", "make"] and len(parts) == 4:
            project, action, body = self._make(parts[2]), parts[3], self._body()
            if JOBS.busy(project.slug) and action not in ("saveplan", "provider"):
                raise Refused("a step is already running for this video; wait for it to finish")
            if action == "saveplan":
                self._json(make_save_plan(project, body.get("plan", "")))
            elif action == "provider":
                self._json(make_set_provider(project, body))
            else:
                self._json(make_job(project, action, body))
            return
        if parts[:2] != ["api", "project"] or len(parts) < 4:
            raise Refused("not found", 404)
        project, action, body = find(parts[2]), parts[3], self._body()
        if JOBS.busy(project.slug) and action in ("decompose", "music", "mix", "voices", "look", "restyle", "frames"):
            raise Refused("a step is already running for this video; wait for it to finish")
        if action == "rights":
            project.set_rights(body.get("status", ""))
            self._json({"rights": project.rights()})
        elif action == "decompose":
            self._json(JOBS.start("decompose", project.slug, ["decompose", project.slug]))
        elif action == "gate":
            self._json(set_gate(project, body.get("gate", ""), bool(body.get("approved", True))))
        elif action == "music":
            self._json(
                start_music(project, body.get("name", ""), body.get("prompt", ""), bool(body.get("approve_over_cap")))
            )
        elif action == "verify":
            name = slugify(body.get("name", ""))
            self._json(JOBS.start(f"check '{name}' against its script", project.slug, ["verify", project.slug, "--name", name]))
        elif action == "voices":
            self._json(start_voices(project, body.get("name", ""), bool(body.get("approve_over_cap"))))
        elif action == "mix":
            self._json(start_mix(project, body.get("name", ""), bool(body.get("voices"))))
        elif action == "look":
            self._json(save_look(project, body.get("name", ""), body.get("audio_from", ""), body.get("style", ""), body.get("characters")))
        elif action == "restyle":
            self._json(start_restyle(project, body.get("name", ""), body.get("shots"), bool(body.get("approve_over_cap"))))
        elif action == "frames":
            self._json(confirm_frames(project, body.get("name", "")))
        elif action == "approve":
            self._json(approve_remix(project, body.get("name", ""), bool(body.get("approved", True))))
        else:
            raise Refused("not found", 404)

    def _handle(self, method: str) -> None:
        try:
            self._route(method)
        except Refused as exc:
            self._json({"error": str(exc)}, exc.status)
        except ProjectError as exc:
            self._json({"error": str(exc)}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # anything unexpected still gets an answer the page can show
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")


def serve(port: int = PORT) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"remix web app on http://localhost:{port} (Google access: {'ready' if google_ready() else 'not set'})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve()
