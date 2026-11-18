"""Make a video from a prompt: plan it, generate the shots with Veo, speak it, score it, compose it, check it.

A make project is a project folder like any other (so voices, captions, the ledger and the web app
work on it unchanged), except it has no source video. Its files:

    brief.json      what was asked for (prompt, length, aspect ratios, style)
    plan.json       the plan: shots to generate, scenes in order, overlays, the voice-over, the music prompt
    assets/         pictures, logos and fonts the plan uses (product screens, a logo, Montserrat)
    shots/          the generated clips: <shot>.<tier>.mp4, with a QA note beside each
    remixes/main/   lines.json, voice.wav, captions (the existing voice and caption machinery)
    music.mp3       the generated score
    out/<aspect>/   final.mp4, the captions, check.json and a contact sheet

Every paid step estimates first, respects the cap, and writes the ledger. Clips are kept: a rerun
only pays for what is missing. Generated footage never carries text; all words on screen are drawn
by `compose`.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from . import compose, config, cost, elevenlabs, google, verify, voices
from .project import SCHEMA, Project, ProjectError, list_projects, now_iso, slugify

ASPECTS = {"16:9": (1920, 1080), "1:1": (1080, 1080), "4:5": (1080, 1350), "9:16": (1080, 1920)}
TIERS = {  # name -> (Veo model, resolution)
    "pilot": ("veo-3.1-lite-generate-001", "720p"),
    "final": ("veo-3.1-fast-generate-001", "1080p"),
    "hero": ("veo-3.1-generate-001", "1080p"),
}
TIER_ORDER = ("hero", "final", "pilot")
CLIP_LENGTHS = (4, 6, 8)
SPEECH_RATE = 2.7  # words a second a voice can say without sounding rushed, at most
REMIX = "main"
NO_TEXT = "No text, no captions, no logos, no watermarks."
KNOWN_OVERLAYS = set(compose.OVERLAYS) | set(compose.NEEDS_ROOT) | {"highlight"}
DEFAULT_NEGATIVE = "text, captions, subtitles, logos, brand names, watermarks, readable license plates, cartoon, illustration, distorted hands, extra fingers, warped faces, blurry, low quality"


class MakeError(RuntimeError):
    pass


# --- projects ---------------------------------------------------------------------------------------


def is_make(project: Project) -> bool:
    try:
        return project.load().get("kind") == "make"
    except (OSError, ValueError):
        return False


def make_projects() -> list[Project]:
    return [p for p in list_projects() if is_make(p)]


def find(ref: str) -> Project:
    name = Path(ref).name
    found = [p for p in make_projects() if p.slug == name] or [p for p in make_projects() if p.slug.startswith(name)] or \
        [p for p in make_projects() if p.slug == f"make-{slugify(name, 30)}"]
    if len(found) != 1:
        raise MakeError(f"no make project matches '{ref}'" if not found else f"'{ref}' matches several: " + ", ".join(p.slug for p in found))
    return found[0]


def create(title: str, brief: dict) -> Project:
    """Start a make project from a title and a brief. Free."""
    slug = f"make-{slugify(title, 30)}"
    project = Project(config.PROJECTS_DIR / slug)
    if project.manifest_path.is_file():
        raise MakeError(f"'{slug}' already exists; pick another title")
    aspects = [a for a in (brief.get("aspects") or ["16:9"]) if a in ASPECTS] or ["16:9"]
    seconds = float(brief.get("seconds") or 30)
    if not 6 <= seconds <= 600:
        raise MakeError("length must be between 6 seconds and 10 minutes")
    for sub in ("assets", "shots", "out"):
        (project.root / sub).mkdir(parents=True, exist_ok=True)
    width, height = ASPECTS[aspects[0]]
    project.save({
        "schema": SCHEMA, "slug": slug, "kind": "make", "created": now_iso(),
        "source": {"file": None, "original_name": title, "duration": seconds, "width": width, "height": height, "fps": compose.FPS},
        "steps": {}, "rights": {"status": "own", "set": now_iso()},
    })
    (project.root / "brief.json").write_text(json.dumps({**brief, "title": title, "seconds": seconds, "aspects": aspects}, indent=2, ensure_ascii=False), encoding="utf-8")
    return project


def brief_of(project: Project) -> dict:
    return json.loads((project.root / "brief.json").read_text(encoding="utf-8"))


def audio_provider(project: Project) -> str:
    """Where the voice and music come from: "gemini" (Google, automatic and paid) or "elevenlabs" (made in the
    user's own ElevenLabs account and saved into the project). It is part of the brief, so the web app can switch it."""
    try:
        return elevenlabs.provider_of(brief_of(project).get("audio_provider"))
    except elevenlabs.ElevenLabsError as exc:
        raise MakeError(str(exc)) from None


def set_audio_provider(project: Project, provider: str) -> str:
    try:
        provider = elevenlabs.provider_of(provider)
    except elevenlabs.ElevenLabsError as exc:
        raise MakeError(str(exc)) from None
    path = project.root / "brief.json"
    brief = brief_of(project)
    brief["audio_provider"] = provider
    path.write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")
    return provider


def load_plan(project: Project) -> dict:
    path = project.root / "plan.json"
    if not path.is_file():
        raise MakeError("there is no plan yet: draft one or write plan.json")
    return json.loads(path.read_text(encoding="utf-8"))


def save_plan(project: Project, plan: dict) -> list[str]:
    """Store the plan (even a flawed one, so it can be fixed) and return what is wrong with it."""
    (project.root / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(float(s.get("seconds", 0)) for s in plan.get("scenes", []))
    manifest = project.load()
    manifest["source"]["duration"] = round(total, 3) or manifest["source"]["duration"]
    project.save(manifest)
    return validate(project, plan)


# --- the plan ---------------------------------------------------------------------------------------


def schedule(plan: dict) -> list[dict]:
    """The scenes with `start` and `end` seconds filled in."""
    t, out = 0.0, []
    for scene in plan["scenes"]:
        out.append({**scene, "start": round(t, 3), "end": round(t + float(scene["seconds"]), 3)})
        t += float(scene["seconds"])
    return out


def clip_length(seconds: float) -> int:
    for length in CLIP_LENGTHS:
        if length >= seconds - 1e-6:
            return length
    raise MakeError(f"a generated clip is at most {CLIP_LENGTHS[-1]} s; split a {seconds:g} s scene in two")


def shot_prompt(plan: dict, shot_id: str) -> str:
    shot = plan["shots"][shot_id]
    chars = {k: v for k, v in (plan.get("characters") or {}).items()}
    try:
        text = shot["prompt"].format_map({"style": plan.get("style", ""), **chars})
    except (KeyError, IndexError, ValueError) as exc:
        raise MakeError(f"shot {shot_id}: the prompt uses {{{exc}}}, which the plan does not define") from None
    return text if "no text" in text.lower() else f"{text} {NO_TEXT}"


def validate(project: Project, plan: dict) -> list[str]:
    """Everything that would make the plan fail or come out badly, in plain words."""
    wrong: list[str] = []
    scenes, shots = plan.get("scenes") or [], plan.get("shots") or {}
    if not scenes:
        return ["the plan has no scenes"]
    brief = brief_of(project) if (project.root / "brief.json").is_file() else {}
    seen: set = set()
    for scene in scenes:
        sid = scene.get("id")
        where = f"scene {sid}"
        if not sid or sid in seen:
            wrong.append(f"{where}: every scene needs its own id")
        seen.add(sid)
        if scene.get("kind") not in ("veo", "image", "title"):
            wrong.append(f"{where}: kind must be veo, image or title")
            continue
        seconds = float(scene.get("seconds") or 0)
        if seconds <= 0:
            wrong.append(f"{where}: seconds must be positive")
            continue
        if scene["kind"] == "veo":
            clip = scene.get("clip")
            if clip not in shots:
                wrong.append(f"{where}: clip '{clip}' is not defined under shots")
            elif float(shots[clip].get("seconds", 0)) not in CLIP_LENGTHS or float(shots[clip]["seconds"]) < seconds - 1e-6:
                wrong.append(f"{where}: its shot '{clip}' makes {shots[clip].get('seconds')} s, which must be 4, 6 or 8 and at least the scene's {seconds:g} s")
        if scene["kind"] == "image":
            file = (scene.get("image") or {}).get("file")
            if not file or not (project.root / file).is_file():
                wrong.append(f"{where}: image file '{file}' is not in the project")
        if scene.get("background") and scene["background"] not in shots:
            wrong.append(f"{where}: background '{scene['background']}' is not defined under shots")
        for o in scene.get("overlays", []):
            if o.get("type") not in KNOWN_OVERLAYS:
                wrong.append(f"{where}: unknown overlay type '{o.get('type')}'")
            for key in ("file", "logo"):
                if o.get(key) and not (project.root / o[key]).is_file():
                    wrong.append(f"{where}: overlay file '{o[key]}' is not in the project")
        if scene.get("vo"):
            words = len(scene["vo"].split())
            room = seconds - float(scene.get("vo_at", 0.25)) - 0.12
            if room <= 0 or words / room > SPEECH_RATE:
                wrong.append(f"{where}: {words} words in {max(room, 0):.1f} s is {words / max(room, 0.1):.1f} words a second; at most {SPEECH_RATE} sounds natural")
    for sid, shot in shots.items():
        if float(shot.get("seconds", 0)) not in CLIP_LENGTHS:
            wrong.append(f"shot {sid}: seconds must be 4, 6 or 8")
        try:
            shot_prompt(plan, sid)
        except MakeError as exc:
            wrong.append(str(exc))
    if any(s.get("vo") for s in scenes):
        if audio_provider(project) == "elevenlabs":
            if not ((plan.get("voice") or {}).get("elevenlabs") or plan.get("elevenlabs_voices")):
                wrong.append("the audio comes from ElevenLabs but the plan names no ElevenLabs voice (voice.elevenlabs)")
        elif (plan.get("voice") or {}).get("name") not in google.VOICES:
            wrong.append(f"voice '{(plan.get('voice') or {}).get('name')}' is not one of Google's voices")
    total = sum(float(s.get("seconds", 0)) for s in scenes)
    if brief.get("seconds") and abs(total - float(brief["seconds"])) > 2.0:
        wrong.append(f"the scenes add up to {total:g} s but the brief asks for {float(brief['seconds']):g} s")
    if brief.get("brand_in_hook") or plan.get("brand", {}).get("in_hook"):
        name = (plan.get("brand") or {}).get("name", "").lower()
        early = [s for s in schedule(plan) if s["start"] < 5.0]
        shown = any(o.get("type") in ("logo", "endcard") and s["start"] + float(o.get("from", 0)) < 5.0 for s in early for o in s.get("overlays", []))
        said = bool(name) and any(name in (s.get("vo") or "").lower() or name in " ".join(str(o.get("text", "")) for o in s.get("overlays", [])).lower() for s in early)
        if not (shown or said):
            wrong.append("the brand is neither shown nor named in the first 5 seconds")
    first = schedule(plan)[0]
    for scene in scenes:
        if scene.get("on_camera") and not (scene.get("kind") == "veo" and shots.get(scene.get("clip"), {}).get("audio")):
            wrong.append(f"scene {scene.get('id')}: a line spoken on camera needs a veo scene whose shot has \"audio\": true")
    if not (first.get("vo") or first.get("overlays") or first.get("on_camera")):
        wrong.append("the first scene has no voice and no on-screen text: nothing hooks the viewer")
    return wrong


def estimate_shots(plan: dict, tier: str, only: list[str] | None = None) -> list[cost.Line]:
    model, resolution = TIERS[tier]
    table, today = cost.load_table(), date.today()
    return [
        table.line(f"clip {sid} ({tier})", google.veo_price_id(model, resolution, bool(shot.get("audio"))), float(shot["seconds"]), today)
        for sid, shot in plan.get("shots", {}).items() if only is None or sid in only
    ]


def tiers_present(project: Project, shot_id: str) -> list[str]:
    return [t for t in TIER_ORDER if (project.root / "shots" / f"{shot_id}.{t}.mp4").is_file()]


def best_clip(project: Project, shot_id: str, allow: tuple[str, ...] = TIER_ORDER) -> Path | None:
    for tier in TIER_ORDER:
        path = project.root / "shots" / f"{shot_id}.{tier}.mp4"
        if tier in allow and path.is_file():
            return path
    return None


# --- paid: the plan itself --------------------------------------------------------------------------

PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"}, "style": {"type": "STRING"}, "music_prompt": {"type": "STRING"},
        "voice_direction": {"type": "STRING"},
        "shots": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "id": {"type": "STRING"}, "seconds": {"type": "INTEGER"}, "prompt": {"type": "STRING"}}, "required": ["id", "seconds", "prompt"]}},
        "scenes": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "id": {"type": "STRING"}, "seconds": {"type": "NUMBER"}, "clip": {"type": "STRING"},
            "vo": {"type": "STRING"}, "headline": {"type": "STRING"}}, "required": ["id", "seconds", "clip"]}},
    },
    "required": ["title", "style", "music_prompt", "shots", "scenes"],
}

PLANNER_RULES = """You plan short and long videos that will be built from generated footage (Google Veo), a spoken voice-over and music.
Rules you must follow:
- Every scene shows one generated clip (`clip` names a shot). A shot is 4, 6 or 8 seconds; a scene's seconds must not exceed its shot's seconds. Several scenes may reuse a shot only if they show different parts of it.
- The scene seconds must add up to the requested length (within one second).
- The first five seconds are the hook: open on a striking image and a line that speaks straight to the viewer's problem. Name or show the brand within five seconds if a brand is given.
- Shot prompts describe one continuous, photoreal, cinematic moment: subject, action, camera movement, lighting, place. Keep the subject centered. Never ask for visible text, signs, logos, screens with readable words, or speech; words on screen are added later as overlays (`headline`).
- Describe any recurring person in the same words each time (put the description in the prompt itself).
- `vo` is the spoken line for that scene, in natural spoken English, at most 2.5 words per second of the scene's seconds. Leave `vo` empty for scenes that need none.
- `headline` is optional big on-screen text for that scene (Title Case, at most 8 words).
- `music_prompt` describes an instrumental score with tempo and how it builds (no vocals).
- End on a clear call to action when the brief names one."""


def draft_plan(project: Project) -> tuple[dict, list[str]]:
    """Have Gemini draft the plan from the brief (a few cents). Returns the plan and its problems."""
    brief = brief_of(project)
    seconds = float(brief["seconds"])
    ask = (
        f"Title: {brief['title']}\nRequested length: {seconds:g} seconds\nBrief: {brief.get('prompt', '')}\n"
        f"Style: {brief.get('style') or 'photoreal and cinematic'}\nBrand: {brief.get('brand') or 'none'}\n"
        f"Call to action: {brief.get('cta') or 'none'}\nAudience: {brief.get('audience') or 'general'}"
    )
    answer, usage = google.generate_json(ask, PLAN_SCHEMA, system=PLANNER_RULES)
    table, today = cost.load_table(), date.today()
    for line in (table.line("plan drafted by Gemini (prompt)", "gemini-flash.input", usage["input_tokens"], today),
                 table.line("plan drafted by Gemini (answer)", "gemini-flash.text-out", usage["output_tokens"], today)):
        cost.record(project.slug, "plan", line, note="draft plan")
    shots = {s["id"]: {"seconds": s["seconds"] if s["seconds"] in CLIP_LENGTHS else clip_length(float(s["seconds"])), "prompt": s["prompt"]} for s in answer["shots"]}
    scenes = []
    for s in answer["scenes"]:
        scene = {"id": s["id"], "kind": "veo", "seconds": float(s["seconds"]), "clip": s["clip"]}
        if s.get("vo", "").strip():
            scene["vo"] = s["vo"].strip()
        if s.get("headline", "").strip():
            scene["overlays"] = [{"type": "headline", "text": s["headline"].strip(), "from": 0.25, "to": max(float(s["seconds"]) - 0.2, 0.5)}]
        scenes.append(scene)
    plan = {
        "title": answer["title"], "style": answer["style"], "negative": DEFAULT_NEGATIVE,
        "voice": {"name": brief.get("voice") or "Kore", "direction": answer.get("voice_direction") or "warm, confident, brisk",
                  **({"elevenlabs": brief["elevenlabs_voice"]} if brief.get("elevenlabs_voice") else {})},
        "music": {"prompt": answer["music_prompt"]}, "brand": {"name": brief.get("brand", ""), "in_hook": bool(brief.get("brand"))},
        "shots": shots, "scenes": scenes,
    }
    return plan, save_plan(project, plan)


# --- paid: the shots --------------------------------------------------------------------------------

QA_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "visible_text": {"type": "ARRAY", "items": {"type": "STRING"}},
        "has_watermark_or_logo": {"type": "BOOLEAN"},
        "artifacts": {"type": "ARRAY", "items": {"type": "STRING"}},
        "summary": {"type": "STRING"},
    },
    "required": ["visible_text", "has_watermark_or_logo", "artifacts", "summary"],
}
QA_QUESTION = (
    "Watch this short generated clip closely. Report: any readable text, letters or numbers anywhere in the picture (visible_text); "
    "whether there is a watermark or a logo (has_watermark_or_logo); anything that looks wrong or unnatural, such as warped faces or "
    "hands, extra limbs, melting or morphing objects, impossible physics, sudden cuts (artifacts); and a two sentence summary of what happens."
)


def generate_shots(project: Project, tier: str, only: list[str] | None = None, workers: int = 3, yes: bool = False,
                   redo: bool = False, watch=None, make=None) -> dict:
    """Generate the missing clips of a tier with Veo and QA each one with Gemini. Paid; kept clips are not paid for again."""
    plan = load_plan(project)
    model, resolution = TIERS[tier]
    todo = [s for s in plan["shots"] if (only is None or s in only) and (redo or not (project.root / "shots" / f"{s}.{tier}.mp4").is_file())]
    lines = estimate_shots(plan, tier, todo)
    planned = cost.total(lines)
    if not cost.within_cap(planned) and not yes:
        raise MakeError(f"${planned:.2f} is over the ${config.CAP_USD:g} cap: ask for the go-ahead and pass --yes")
    table, today = cost.load_table(), date.today()
    negative = plan.get("negative") or DEFAULT_NEGATIVE
    generator = make or google.generate_veo

    def one(shot_id: str):
        shot = plan["shots"][shot_id]
        try:
            video, _ = generator(shot_prompt(plan, shot_id), model=model, aspect=brief_of(project).get("veo_aspect", "16:9"),
                                 resolution=resolution, seconds=int(shot["seconds"]), negative=negative,
                                 **({"audio": True} if shot.get("audio") else {}))  # a clip with sound: speech and the room
        except google.GoogleError as exc:
            return shot_id, None, str(exc)[:300]
        target = project.root / "shots" / f"{shot_id}.{tier}.mp4"
        target.write_bytes(video)
        line = table.line(f"Veo clip {shot_id}", google.veo_price_id(model, resolution, bool(shot.get("audio"))), float(shot["seconds"]), today)
        cost.record(project.slug, "shots", line, estimated_usd=line.usd, note=f"{tier} {model} {resolution}")
        return shot_id, target, None

    results = {"made": [], "failed": {}, "estimate": round(planned, 4), "tier": tier}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for shot_id, path, error in pool.map(one, todo):
            if error:
                results["failed"][shot_id] = error
            else:
                results["made"].append(shot_id)
                qa_clip(project, shot_id, tier, watch)
    return results


def qa_clip(project: Project, shot_id: str, tier: str, watch=None) -> dict:
    """Ask Gemini what is in a generated clip and whether anything is wrong with it (about a cent)."""
    clip = project.root / "shots" / f"{shot_id}.{tier}.mp4"
    note = clip.with_suffix(".qa.json")
    try:
        answer, usage = (watch or google.watch_video)(clip.read_bytes(), QA_QUESTION, QA_SCHEMA)
    except google.GoogleError as exc:
        answer, usage = {"error": str(exc)[:200]}, {}
    table, today = cost.load_table(), date.today()
    if usage:
        for line in (table.line("clip QA (video in)", "gemini-flash.input", usage["input_tokens"], today),
                     table.line("clip QA (answer)", "gemini-flash.text-out", usage["output_tokens"], today)):
            cost.record(project.slug, "shots", line, note=f"QA {shot_id}")
    note.write_text(json.dumps(answer, indent=2, ensure_ascii=False), encoding="utf-8")
    return answer


def qa_problems(project: Project, plan: dict) -> list[str]:
    """What Gemini flagged in the chosen clips. A flag can be waived by name with a reason under
    `"accepted_notes": {"<shot>": {"text": "reason"}}` in the plan; a waived flag is a person's decision, so the reason is required."""
    wrong = []
    accepted = plan.get("accepted_notes") or {}
    for sid in plan.get("shots", {}):
        clip = best_clip(project, sid)
        note = clip.with_suffix(".qa.json") if clip else None
        if not clip:
            wrong.append(f"shot {sid} has no clip yet")
        elif note and note.is_file():
            qa = json.loads(note.read_text(encoding="utf-8"))
            waived = accepted.get(sid) or {}
            if qa.get("visible_text") and not waived.get("text"):
                wrong.append(f"shot {sid}: generated text in the picture ({', '.join(qa['visible_text'][:3])})")
            if qa.get("has_watermark_or_logo") and not waived.get("text"):
                wrong.append(f"shot {sid}: a watermark or logo appears")
            if not waived.get("artifacts"):
                for artifact in (qa.get("artifacts") or [])[:2]:
                    wrong.append(f"shot {sid}: {artifact}")
    return wrong


# --- voice, captions, score ---------------------------------------------------------------------------


def voice_for(plan: dict, speaker: str, provider: str) -> str:
    """The voice a speaker has under the chosen audio provider: a Gemini stock voice, or a voice in the ElevenLabs account."""
    if provider == "elevenlabs":
        return (plan.get("elevenlabs_voices") or {}).get(speaker) or (plan.get("voice") or {}).get("elevenlabs") or elevenlabs.DEFAULT_VOICE
    return (plan.get("voices") or {}).get(speaker) or (plan.get("voice") or {}).get("name")


def script_of(project: Project, plan: dict) -> dict:
    """The remix's script (what lines.json holds) from the scenes' voice-over lines, timed by where each scene starts."""
    lines, speakers = [], {}
    base = plan.get("voice") or {}
    provider = audio_provider(project)
    for scene in schedule(plan):
        if not scene.get("vo"):
            continue
        start = scene["start"] + float(scene.get("vo_at", 0.25))
        room = scene["end"] - start - 0.12
        words = len(scene["vo"].split())
        speaker = scene.get("speaker", "narrator")
        line = {"id": len(lines) + 1, "start": round(start, 3), "end": round(start + min(room, max(words / 2.4, 0.5)), 3),
                "speaker": speaker, "text": scene["vo"]}
        for key in ("say", "style"):
            if scene.get(key):
                line[key] = scene[key]
        if scene.get("captions_off"):  # the words are already on screen as text
            line["captions"] = False
        lines.append(line)
        speakers[speaker] = voice_for(plan, speaker, provider)
    if not lines:
        raise MakeError("no scene has a voice-over line")
    speakers = dict(sorted(speakers.items(), key=lambda kv: kv[0] != "narrator"))  # the narrator's captions are the first style, white
    directions = {s: (plan.get("directions") or {}).get(s) or base.get("direction", "") for s in speakers}
    return {"voices": speakers, "directions": directions, "audio": provider, "lines": lines}


def write_script(project: Project, plan: dict) -> Path:
    folder = voices.folder(project, REMIX)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "lines.json"
    path.write_text(voices.dump_script(script_of(project, plan)), encoding="utf-8")
    return path


def audio_order(project: Project, write: bool = True) -> dict:
    """What the audio provider still has to make. For ElevenLabs: every line and the music, and which are already saved."""
    plan = load_plan(project)
    provider = audio_provider(project)
    if provider != "elevenlabs":
        return {"provider": provider}
    total = sum(float(s["seconds"]) for s in plan["scenes"])
    music = {"prompt": (plan.get("music") or {}).get("prompt", ""), "seconds": total}
    return {"provider": provider, **elevenlabs.order(project, REMIX, script_of(project, plan), music=music, write=write)}


def speak(project: Project, pilot: int = 0, workers: int = 4, redo: tuple[int, ...] = (), synth=None) -> dict:
    """Speak the voice-over: with Gemini (paid, cents) or, for an ElevenLabs project, bring in the takes made there (free)."""
    plan = load_plan(project)
    provider = audio_provider(project)
    path = write_script(project, plan)
    rerolls = 2
    if provider == "elevenlabs" and synth is None:
        script = json.loads(path.read_text(encoding="utf-8"))
        order = elevenlabs.order(project, REMIX, script)
        wanted = [line["id"] for line in script["lines"][:pilot or None]]
        missing = [i for i in order["missing"] if i in wanted]
        if missing and elevenlabs.api_ready():  # a key is set: the SDK speaks the lines that have no take yet (credits, not dollars)
            try:
                elevenlabs.fill_voice(project, REMIX, script, ids=missing)
            except elevenlabs.ElevenLabsError as exc:
                raise MakeError(str(exc)) from None
            order = elevenlabs.order(project, REMIX, script)
            missing = [i for i in order["missing"] if i in wanted]
        if missing:
            raise MakeError(f"{len(missing)} of {len(wanted)} lines are not made in ElevenLabs yet (line {', '.join(map(str, missing))}): "
                            f"the order is in {order['folder']}/order.md; save each take there as NNNN.mp3, or set {elevenlabs.KEY_ENV} "
                            "so the API makes them, then run this step again")
        synth, rerolls = elevenlabs.Takes(project, REMIX, script), 0  # a saved take cannot be spoken again at another pace
    return voices.generate(project, REMIX, synth=synth or google.synthesize, workers=workers, pilot=pilot, redo=redo, rerolls=rerolls, provider=provider)


def import_music(project: Project, plan: dict) -> Path:
    """Use the score made in ElevenLabs (saved as eleven/music.mp3). Free: it comes out of the account's credits."""
    source = elevenlabs.music_file(project)
    total = sum(float(s["seconds"]) for s in plan["scenes"])
    if source is None and elevenlabs.api_ready():  # a key is set: the SDK composes it from the plan's music prompt
        try:
            source = elevenlabs.fill_music(project, (plan.get("music") or {}).get("prompt", ""), total)
        except elevenlabs.ElevenLabsError as exc:
            raise MakeError(str(exc)) from None
    if source is None:
        raise MakeError(f"the music is not made in ElevenLabs yet: make about {total:g} s from the prompt in the order "
                        f"({project.rel(elevenlabs.folder(project, REMIX))}/order.md) and save it as eleven/music.mp3, "
                        f"or set {elevenlabs.KEY_ENV} so the API makes it")
    for old in project.root.glob("music.*"):  # one score only: the mixer takes the first music.* it finds
        old.unlink()
    target = project.root / f"music{source.suffix.lower()}"
    shutil.copyfile(source, target)
    cost.record(project.slug, "music", cost.load_table().line("music made in ElevenLabs (plan credits)", "elevenlabs.music", 1, date.today()), estimated_usd=0.0)
    return target


def score(project: Project, long_model: bool = False, generate=google.generate_music) -> Path:
    """Generate the music with Lyria (a few cents), or bring in the score made in ElevenLabs."""
    plan = load_plan(project)
    if audio_provider(project) == "elevenlabs" and generate is google.generate_music:
        return import_music(project, plan)
    prompt = (plan.get("music") or {}).get("prompt")
    if not prompt:
        raise MakeError("the plan has no music prompt")
    total = sum(float(s["seconds"]) for s in plan["scenes"])
    clip_model = total <= 30 and not long_model
    model = "lyria-3-clip-preview" if clip_model else google.MUSIC_MODEL
    table, today = cost.load_table(), date.today()
    try:
        audio, mime, notes = generate(prompt, model=model)
    except google.GoogleError:
        if not clip_model:
            raise
        clip_model, model = False, google.MUSIC_MODEL
        audio, mime, notes = generate(prompt, model=model)
    lines = [table.line("music clip", "lyria-3-clip.clip", 1, today)] if clip_model else cost.music(table, today, total)
    for line in lines:
        cost.record(project.slug, "music", line, estimated_usd=line.usd)
    path = project.root / f"music{google.EXTENSIONS.get(mime, '.mp3')}"
    path.write_bytes(audio)
    if notes:
        (project.root / "music_notes.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    return path


# --- sound effects (made here, free) --------------------------------------------------------------------


def sfx(kind: str, rate: int = 48000) -> np.ndarray:
    rng = np.random.default_rng(7)
    if kind == "ping":
        t = np.arange(int(rate * 0.6)) / rate
        return (0.55 * np.sin(2 * np.pi * 1318 * t) * np.exp(-t * 8) + 0.25 * np.sin(2 * np.pi * 2637 * t) * np.exp(-t * 13)).astype(np.float32)
    if kind == "ding":
        t = np.arange(int(rate * 1.0)) / rate
        return (0.45 * np.sin(2 * np.pi * 1760 * t) * np.exp(-t * 4.5) + 0.2 * np.sin(2 * np.pi * 3520 * t) * np.exp(-t * 7)).astype(np.float32)
    if kind == "tick":
        t = np.arange(int(rate * 0.05)) / rate
        return (0.35 * np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 110) + 0.2 * rng.standard_normal(len(t)) * np.exp(-t * 160)).astype(np.float32)
    if kind == "thud":
        t = np.arange(int(rate * 0.6)) / rate
        return (0.9 * np.sin(2 * np.pi * (45 + 60 * np.exp(-t * 14)) * t) * np.exp(-t * 6.5)).astype(np.float32)
    if kind in ("whoosh", "riser"):
        length = 0.7 if kind == "whoosh" else 1.6
        n = int(rate * length)
        t = np.arange(n) / rate
        noise = rng.standard_normal(n).astype(np.float32)
        env = np.sin(np.pi * np.clip(t / length, 0, 1)) ** (2 if kind == "whoosh" else 1)
        if kind == "riser":
            env = (t / length) ** 1.8
        out, state = np.empty(n, np.float32), 0.0
        for i in range(n):  # a one-pole low-pass that opens up: the sweep
            k = 0.02 + 0.5 * (i / n) ** 1.5
            state += k * (noise[i] - state)
            out[i] = state
        return (out * env * (0.7 if kind == "whoosh" else 0.5)).astype(np.float32)
    raise MakeError(f"unknown sound effect '{kind}'")


def sfx_track(plan: dict, rate: int = 48000) -> np.ndarray | None:
    scenes = schedule(plan)
    cues = [(s["start"] + float(c.get("at", 0)), c["kind"], float(c.get("gain", 1.0))) for s in scenes for c in s.get("sfx", [])]
    if not cues:
        return None
    total = scenes[-1]["end"]
    track = np.zeros(int((total + 2) * rate), np.float32)
    level = 10 ** (float(plan.get("sfx_gain_db", -14)) / 20)  # effects sit well under the voice and music; plan["sfx_gain_db"] sets it
    for at, kind, gain in cues:
        clip = sfx(kind, rate) * gain * level
        i = int(at * rate)
        track[i : i + len(clip)] += clip[: max(len(track) - i, 0)]
    return np.clip(track, -1, 1)


# --- compose and mix -------------------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if proc.returncode:
        raise MakeError(f"{cmd[0]} failed: {proc.stderr.strip()[-500:]}")


def render(project: Project, aspect: str = "16:9", preview: bool = False, progress=None) -> Path:
    """Render one aspect ratio. Only one render of a project runs at a time: two would overwrite each other's files."""
    lock = project.root / "work" / "render.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_file() and time.time() - lock.stat().st_mtime < 1800:
        raise MakeError("another render of this project is running; wait for it to finish")
    lock.write_text(now_iso(), encoding="utf-8")
    try:
        return _render(project, aspect, preview, progress)
    finally:
        lock.unlink(missing_ok=True)


def _render(project: Project, aspect: str = "16:9", preview: bool = False, progress=None) -> Path:
    """Draw every scene, put the sound under it, burn the captions, and write out/<aspect>/final.mp4."""
    import soundfile as sf

    plan = load_plan(project)
    problems = validate(project, plan)
    if problems:
        raise MakeError("the plan has problems: " + "; ".join(problems[:4]))
    size = ASPECTS[aspect]
    tag = aspect.replace(":", "x")
    work, out = project.root / "work" / tag, project.root / "out" / tag
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    allow = TIER_ORDER if not preview else ("pilot",) + TIER_ORDER
    scenes = schedule(plan)
    parts = []
    for i, scene in enumerate(scenes):
        if scene.get("on_camera") and not scene.get("captions_off"):  # words spoken on camera get a caption too
            cam = scene["on_camera"]
            shown = {"type": "subtitle", "text": cam["text"], "from": float(cam.get("from", 0.15)), "to": float(cam.get("to", float(scene["seconds"]) - 0.1))}
            scene = {**scene, "overlays": [*scene.get("overlays", []), shown]}
        clip = best_clip(project, scene["clip"], allow) if scene.get("clip") else None
        background = best_clip(project, scene["background"], allow) if scene.get("background") else None
        if scene["kind"] == "veo" and clip is None:
            raise MakeError(f"scene {scene['id']} needs its shot '{scene['clip']}': generate the shots first")
        key = hashlib.sha1(json.dumps([scene, aspect, [str(p) + str(p.stat().st_size) for p in (clip, background) if p],
                                       compose.FPS, compose.VERSION], sort_keys=True, default=str).encode()).hexdigest()[:16]
        part, keyfile = work / f"{i:02d}_{scene['id']}.mp4", work / f"{i:02d}_{scene['id']}.key"
        if not (part.is_file() and keyfile.is_file() and keyfile.read_text() == key):
            compose.render_scene(scene, part, size, project.root, project.root / "assets", clip, background)
            keyfile.write_text(key)
        parts.append(part)
        if progress:
            progress(i + 1, len(scenes))
    (work / "parts.txt").write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    silent = work / "picture.mp4"
    _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "parts.txt", "-c", "copy", silent.name], cwd=work)

    total = scenes[-1]["end"]
    inputs, graph, labels = ["-i", str(silent)], [], []

    def add_input(*args: str) -> int:
        inputs.extend(args)
        return sum(1 for arg in inputs if arg == "-i") - 1

    stereo = "aformat=sample_rates=48000:channel_layouts=stereo"
    folder = voices.folder(project, REMIX)
    voice = folder / "voice.wav"
    spoken = []  # what the music steps back for: the voice-over and any line spoken on camera
    if voice.is_file():
        write_script(project, plan)  # keep the script's timing and speaker order in step with the plan (free)
        n = add_input("-i", str(voice))
        graph.append(f"[{n}:a]{stereo},apad,atrim=0:{total:.3f}[vo]")
        labels.append("vo")
        spoken.append("vo")
    # The sound a generated clip came with: a line spoken on camera, and the room it was shot in.
    for k, scene in enumerate(scenes):
        shot = plan["shots"].get(scene.get("clip") or "", {})
        clip = best_clip(project, scene["clip"], allow) if scene.get("clip") else None
        if scene["kind"] == "veo" and shot.get("audio") and clip is not None and "audio" in verify.stream_facts(clip):
            seconds = float(scene["seconds"])
            n = add_input("-t", f"{seconds:.3f}", "-i", str(clip))
            graph.append(f"[{n}:a]{stereo},volume={float(scene.get('clip_gain_db', 0))}dB,afade=t=in:d=0.05,"
                         f"afade=t=out:st={max(seconds - 0.2, 0):.2f}:d=0.2,adelay={int(scene['start'] * 1000)}:all=1,apad,atrim=0:{total:.3f}[cs{k}]")
            labels.append(f"cs{k}")
            if scene.get("on_camera"):
                spoken.append(f"cs{k}")
    music = next(iter(sorted(project.root.glob("music.*"))), None)
    if music:
        n = add_input("-i", str(music))
        gain = float((plan.get("music") or {}).get("gain_db", -15))
        graph.append(f"[{n}:a]{stereo},apad,atrim=0:{total:.3f},volume={gain}dB,"
                     f"afade=t=in:d=0.3,afade=t=out:st={max(total - 0.8, 0):.2f}:d=0.8[mus]")
        if spoken:  # the music steps back while anyone speaks
            graph += [f"[{label}]asplit=2[{label}][{label}key]" for label in spoken]
            keys = "".join(f"[{label}key]" for label in spoken)
            graph.append(f"{keys}amix=inputs={len(spoken)}:normalize=0[side]" if len(spoken) > 1 else f"{keys}anull[side]")
            graph.append("[mus][side]sidechaincompress=threshold=0.03:ratio=9:attack=15:release=400[duck]")
            labels.append("duck")
        else:
            labels.append("mus")
    effects = sfx_track(plan)
    if effects is not None:
        sfx_path = work / "sfx.wav"
        sf.write(str(sfx_path), np.stack([effects, effects], axis=1), 48000, subtype="PCM_16")
        n = add_input("-i", str(sfx_path))
        graph.append(f"[{n}:a]{stereo},atrim=0:{total:.3f}[sfx]")
        labels.append("sfx")
    video_filter = "null"
    if voice.is_file():
        stem = f"captions_{tag}"
        voices.write_captions(project, REMIX, size=size, stem=stem)
        video_filter = f"subtitles={stem}.ass"
    final = out / "final.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs]
    if labels:
        mix = "".join(f"[{l}]" for l in labels) + f"amix=inputs={len(labels)}:normalize=0:duration=longest,loudnorm=I=-14:TP=-1.5:LRA=9,aresample=48000,atrim=0:{total:.3f}[a]"
        cmd += ["-filter_complex", ";".join(graph + [mix]), "-map", "0:v:0", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-vf", video_filter, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(compose.FPS)]
    cmd += (["-c:a", "aac", "-b:a", "192k", "-ar", "48000"] if labels else []) + ["-movflags", "+faststart", "-t", f"{total:.3f}", str(final.resolve())]
    _run(cmd, cwd=folder if voice.is_file() else work)
    if voice.is_file():
        (out / "captions.srt").write_bytes((folder / f"captions_{tag}.srt").read_bytes())
    (out / "render.json").write_text(json.dumps({"aspect": aspect, "preview": preview, "made": now_iso(), "seconds": total}), encoding="utf-8")
    return final


# --- the check ---------------------------------------------------------------------------------------------


def loudness(path: Path) -> dict:
    proc = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    integrated = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", proc.stderr)
    peak = re.findall(r"Peak:\s+(-?[\d.]+)\s+dBFS", proc.stderr)
    return {"lufs": float(integrated[-1]) if integrated else None, "true_peak_db": float(peak[-1]) if peak else None}


def contact_sheet(video: Path, out: Path, every: float = 1.0, columns: int = 6) -> Path:
    from PIL import Image, ImageDraw

    length = verify.stream_facts(video).get("video", {}).get("seconds") or 0
    times = [i * every for i in range(int(length / every))] or [0.0]
    width, height = 320, 180
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout.strip().split(",")
    if len(probe) == 2:
        vw, vh = int(probe[0]), int(probe[1])
        height = int(width * vh / vw)
    rows = (len(times) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (width + 3) + 3, rows * (height + 16) + 3), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    tmp = out.with_suffix(".tmp.jpg")
    for k, t in enumerate(times):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-update", "1", str(tmp)], check=True)
        x, y = 3 + (k % columns) * (width + 3), 3 + (k // columns) * (height + 16)
        sheet.paste(Image.open(tmp).convert("RGB").resize((width, height)), (x, y))
        draw.text((x + 2, y + height + 2), f"{t:.0f}s", fill=(255, 210, 120))
    tmp.unlink(missing_ok=True)
    sheet.save(out, quality=86)
    return out


def on_camera_problems(project: Project, plan: dict, final: Path, listen) -> list[dict]:
    """A line a character speaks on camera must be heard as written, inside its scene."""
    try:
        heard = listen(final)
    except IndexError:  # the recognizer's music bug: hear each clip by itself instead
        heard = []
        for scene in schedule(plan):
            clip = best_clip(project, scene["clip"]) if scene.get("on_camera") and scene.get("clip") else None
            if clip:
                heard += [{**w, "start": w["start"] + scene["start"]} for w in listen(clip) if w["start"] < float(scene["seconds"])]
    wrong = []
    for scene in schedule(plan):
        cam = scene.get("on_camera")
        if not cam:
            continue
        want = verify.words_of(cam["text"])
        got = [token for w in heard if scene["start"] - 0.3 <= w["start"] < scene["end"] + 0.5 for token in verify.words_of(w["word"])]
        same = sum(block.size for block in difflib.SequenceMatcher(None, want, got, autojunk=False).get_matching_blocks())
        if want and same / len(want) < verify.MATCH:
            wrong.append({"problem": f"scene {scene['id']}: the line spoken on camera is not heard as written (heard: {' '.join(got)[:80] or 'nothing'})"})
    return wrong


def check(project: Project, aspect: str = "16:9", frames_ok: bool = False, listen=verify.listen) -> dict:
    """Check a rendered video against its plan and script: length, words, captions, loudness, hook, clip QA."""
    plan = load_plan(project)
    tag = aspect.replace(":", "x")
    out = project.root / "out" / tag
    final = out / "final.mp4"
    if not final.is_file():
        raise MakeError(f"render {aspect} first")
    problems: list[dict] = []
    total = sum(float(s["seconds"]) for s in plan["scenes"])
    facts = verify.stream_facts(final)
    picture, sound = (facts.get(k, {}).get("seconds") for k in ("video", "audio"))
    for label, value in (("picture", picture), ("sound", sound)):
        if value is None or abs(value - total) > 0.2:
            problems.append({"problem": f"the {label} is {value if value is None else round(value, 2)} s long, the plan {total:g} s"})
    folder = voices.folder(project, REMIX)
    lines = json.loads((folder / "lines.json").read_text(encoding="utf-8"))["lines"] if (folder / "lines.json").is_file() else []
    if lines:
        voice_wav = folder / "voice.wav"
        try:
            heard = listen(final)
        except IndexError:  # faster-whisper's word timing breaks on some music stretches: hear the clean voice track instead
            heard = listen(voice_wav)
            # ...and prove that same track is in the finished mix, at the right place
            late, found = verify.voice_offset(final, voice_wav)
            if found < verify.FOUND:
                problems.append({"problem": "the voice track could not be found in the video's sound"})
            elif abs(late) > verify.OFFSET:
                problems.append({"problem": f"the voices sit {late:+.2f} s from where the plan puts them"})
        problems += verify.audio_problems(lines, heard)
        problems += verify.caption_problems(folder, lines, f"captions_{tag}.srt")
        report = json.loads((folder / "voice.json").read_text(encoding="utf-8")) if (folder / "voice.json").is_file() else {}
        if report.get("script") != voices.script_hash(lines):
            problems.append({"problem": "the voice was spoken from a different script than the plan now gives: speak again"})
        late = [r["id"] for r in report.get("report", []) if abs(r.get("end_error", 0)) > 0.6]
        if late:
            problems.append({"problem": f"voice lines {late} end more than 0.6 s from their slot"})
    if any(scene.get("on_camera") for scene in plan["scenes"]):
        problems += on_camera_problems(project, plan, final, listen)
    sound_level = loudness(final)
    if sound_level["lufs"] is not None and not -16.5 <= sound_level["lufs"] <= -12.5:
        problems.append({"problem": f"loudness is {sound_level['lufs']} LUFS; platforms expect about -14"})
    if sound_level["true_peak_db"] is not None and sound_level["true_peak_db"] > -0.9:
        problems.append({"problem": f"true peak {sound_level['true_peak_db']} dB is too hot (limit -1)"})
    problems += [{"problem": p} for p in validate(project, plan) if "brand" in p or "hook" in p]
    problems += [{"problem": p} for p in qa_problems(project, plan)]
    sheet = contact_sheet(final, out / "check_sheet.jpg")
    if not frames_ok:
        problems.append({"problem": "the frames have not been read: look at check_sheet.jpg (every second of the video), then rerun with --frames-ok"})
    result = {"checked": now_iso(), "aspect": aspect, "passed": not problems, "loudness": sound_level, "seconds": {"picture": picture, "sound": sound, "plan": total},
              "problems": problems, "sheet": project.rel(sheet)}
    (out / "check.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
