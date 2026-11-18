"""Re-voicing and captions for a remix.

`lines.json` is the script of a remix: one entry per spoken line with its time slot and speaker.
It starts as the original transcript and can be edited or translated before any voice is made.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from . import cost, google
from .media import read_pcm_f32
from .project import Project, slugify

RATE = 24000
MIN_SPEED, MAX_SPEED = 0.90, 1.12  # tempo change a take may get to fill its line's time; beyond this it sounds dragged or rushed
MAX_SPEEDUP = MAX_SPEED
SHORT_LINE, SHORT_SPEED = 0.8, 1.25  # lines whose original lasts under 0.8 s (a word or two) may be tightened up to 1.25x
FIT = 0.25  # seconds a line's words may end away from the end of the original utterance before the line is spoken again
GAP = 0.08  # breathing room kept before the next line
HOP = 0.01  # seconds per loudness frame
SPEECH_RANGE = 20.0  # a frame holds a voice when it is within this many dB of the loud frames
REAL_PAUSE = 0.2  # a gap between words at least this long counts as one speaker handing over
DEFAULT_VOICES = ("Charon", "Kore", "Puck", "Aoede")  # handed to speakers in order of appearance
CAPTION_COLORS = ("&H00FFFFFF", "&H0078D7FF", "&H00FFD27F", "&H0090EE90")  # ASS colors are BGR
CAPTION_CHARS = 42


# Inline vocal cues that Gemini 3.8 TTS performs rather than reads, written in angle brackets inside
# a line's `say` text (Google's prompting guide, read 2026-11-17). Anything else in angle brackets
# risks being read aloud, so it is refused before any paid call.
VOCAL_TAGS = frozenset((
    "argh", "breath", "heavy breath", "exhales", "cackle", "cheer", "chuckle", "chuckles", "cough", "cry", "gasp",
    "giggle", "groan", "growl", "grunt", "grr", "hiss", "laugh", "laughter", "moan", "pant", "pff", "phew", "scream",
    "shout", "shriek", "sigh", "sighs", "sneeze", "snicker", "snort", "sob", "throat-clearing", "tsk", "whimper",
    "whispers", "whispering", "yawn", "short pause", "long pause",
))
# Hesitation words a `say` text may add to its line: spoken, but too small to caption.
FILLERS = frozenset("uh um uhm hm hmm mm oh ah eh huh well so yeah okay ok like right look man whoa wow heh ha".split())
PERFORMED = 0.25  # share of lines that must carry a vocal cue before a full voice run (the user wants expressive voices)
TAG = re.compile(r"<([^<>]*)>")
CUE_SECONDS = {"short pause": 0.25, "long pause": 1.0}  # every other cue is taken as half a second


class VoiceError(RuntimeError):
    pass


def performed(line: dict) -> str:
    """What is sent to the voice model for a line: its `say` text (words plus vocal cues), or the plain text."""
    return line.get("say") or line["text"]


def cues(line: dict) -> list[str]:
    return [tag.strip().lower() for tag in TAG.findall(line.get("say") or "")]


def cue_seconds(line: dict) -> float:
    return sum(CUE_SECONDS.get(tag, 0.5) for tag in cues(line))


def lead_seconds(line: dict) -> float:
    """Time the vocal cues in front of a line's first word may take (a chuckle measured about 1.2 s in a pilot)."""
    opening = re.match(r"\s*((?:<[^<>]*>\s*)+)", line.get("say") or "")
    if not opening:
        return 0.0
    return sum(CUE_SECONDS.get(tag.strip().lower(), 1.2) for tag in TAG.findall(opening.group(1)))


def performance_problems(lines: list[dict]) -> list[str]:
    """Why a script's `say` texts cannot be spoken as they are.

    `text` is what the captions show and the script check listens for, so `say` must hold exactly
    those words in order. It may add vocal cues from VOCAL_TAGS and small hesitation words, nothing else.
    """
    from .verify import words_of

    problems = []
    for line in lines:
        if not line.get("say"):
            continue
        unknown = [tag for tag in cues(line) if tag not in VOCAL_TAGS]
        if unknown:
            problems.append(f"line {line['id']}: <{unknown[0]}> is not a vocal cue the voice model performs; it would be read aloud")
        wanted, at = words_of(line["text"]), 0
        for word in words_of(TAG.sub(" ", line["say"])):
            if at < len(wanted) and word == wanted[at]:
                at += 1
            elif word not in FILLERS:
                problems.append(f"line {line['id']}: `say` has the word '{word}' where `text` does not; captions and voice would differ")
                break
        else:
            if at < len(wanted):
                problems.append(f"line {line['id']}: `say` leaves out '{wanted[at]}' from `text`")
    return problems


def tight_lines(lines: list[dict], duration: float) -> list[int]:
    """Lines whose words and vocal cues are unlikely to fit before the next line starts."""
    tight = []
    for i, line in enumerate(lines):
        following = lines[i + 1]["start"] if i + 1 < len(lines) else duration
        needed = len(TAG.sub(" ", performed(line)).split()) / 2.6 + 0.3 + cue_seconds(line)
        if cues(line) and needed > (following - line["start"]) * MAX_SPEEDUP:
            tight.append(line["id"])
    return tight


def folder(project: Project, name: str) -> Path:
    return project.root / "remixes" / slugify(name)


def script_hash(lines: list[dict]) -> str:
    """A fingerprint of a script: its lines' ids, time slots, speakers and words.

    Voices, captions and a regenerated picture each record the fingerprint of the script they were
    made from, so anything made from a different version of lines.json is caught by code.
    """
    import hashlib

    rows = [[line["id"], round(line["start"], 3), round(line["end"], 3), line["speaker"], line["text"]] for line in lines]
    return hashlib.sha1(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def spoken_in(shot: dict, lines: list[dict]) -> str:
    """The words spoken during a shot: every line whose middle falls inside it."""
    return " ".join(l["text"] for l in lines if shot["start"] <= (l["start"] + l["end"]) / 2 < shot["end"])


def snap_to_pauses(words: list[dict], owners: list[str], reach: int = 2) -> list[str]:
    """Move each change of speaker to the longest pause within a couple of words of it.

    Picture cuts rarely land exactly between two people's lines; the pause between them does.
    """
    owners = list(owners)
    i = 1
    while i < len(words):
        if owners[i] == owners[i - 1]:
            i += 1
            continue
        low, high = max(i - reach, 0), min(i + reach, len(words))
        inside = [j for j in range(low, high + 1) if 0 < j < len(words)]
        best = max(inside, key=lambda j: words[j]["start"] - words[j - 1]["end"])
        if words[best]["start"] - words[best - 1]["end"] < REAL_PAUSE:
            # No real pause nearby: a stray word or two at the edge of a segment joins its neighbor.
            edges = [j for j in (0, len(words)) if low <= j <= high]
            if edges:
                best = min(edges, key=lambda j: abs(j - i))
        before, after = owners[i - 1], owners[i]
        for j in range(min(i, best), max(i, best)):
            owners[j] = after if best < i else before
        i = max(i, best) + 1
    return owners


def build_lines(project: Project) -> dict:
    """The original script as a line list, with each line's speaker taken from the blueprint."""
    transcript_path = project.root / "transcript" / "transcript.json"
    if not transcript_path.is_file():
        raise VoiceError("there is no transcript: run `remix decompose` first")
    segments = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
    blueprint_path = project.root / "blueprint.json"
    shots = json.loads(blueprint_path.read_text(encoding="utf-8"))["shots"] if blueprint_path.is_file() else []

    from .steps.transcript import join_words

    def speaker_at(time: float) -> str | None:
        shot = next((s for s in shots if s["start"] <= time < s["end"]), None)
        return (shot or {}).get("speaker") or None

    lines, last = [], "narrator"
    for seg in segments:
        words = seg.get("words") or [{"start": seg["start"], "end": seg["end"], "word": seg["text"]}]
        owners = []
        for word in words:  # a word belongs to whoever speaks in the shot it falls in
            last = speaker_at((word["start"] + word["end"]) / 2) or last
            owners.append(last)
        owners = snap_to_pauses(words, owners)
        first = 0
        for i in range(1, len(words) + 1):
            if i == len(words) or owners[i] != owners[first]:
                group = words[first:i]
                lines.append(
                    {
                        "id": len(lines) + 1,
                        "start": group[0]["start"],
                        "end": group[-1]["end"],
                        "speaker": owners[first],
                        "text": join_words(group),
                    }
                )
                first = i
    speakers = list(dict.fromkeys(line["speaker"] for line in lines))
    return {
        "voices": {s: DEFAULT_VOICES[i % len(DEFAULT_VOICES)] for i, s in enumerate(speakers)},
        "directions": {s: "" for s in speakers},
        "lines": lines,
    }


def write_lines(project: Project, name: str, force: bool = False) -> Path:
    path = folder(project, name) / "lines.json"
    if path.is_file() and not force:
        raise VoiceError("lines.json already exists for this remix; pass --force to rebuild it from the transcript")
    path.parent.mkdir(parents=True, exist_ok=True)
    script = build_lines(project)
    # Transcript stamps sit before the real speech; a script starts life on the measured utterances.
    for line, real in zip(script["lines"], utterances(project, script["lines"])):
        if real["found"]:
            line["start"], line["end"] = real["start"], real["end"]
    path.write_text(dump_script(script), encoding="utf-8")
    return path


def load_lines(project: Project, name: str) -> dict:
    path = folder(project, name) / "lines.json"
    if not path.is_file():
        raise VoiceError(f"no lines.json for this remix: run `remix lines {project.slug} --name {slugify(name)}` first")
    script = json.loads(path.read_text(encoding="utf-8"))
    # Gemini's voices are a fixed list; ElevenLabs voices are named in the user's account, so a script made for them is not checked here.
    unknown = [] if script.get("audio") == "elevenlabs" else sorted({v for v in script["voices"].values() if v not in google.VOICES})
    if unknown:
        raise VoiceError(f"unknown voice(s) in lines.json: {', '.join(unknown)}")
    wrong = performance_problems(script["lines"])
    if wrong:
        raise VoiceError("lines.json cannot be spoken as written: " + "; ".join(wrong[:5]))
    return script


def estimate(script: dict, on: date) -> list[cost.Line]:
    seconds = sum(line["end"] - line["start"] for line in script["lines"])
    characters = sum(len(performed(line)) for line in script["lines"])
    return cost.speech(cost.load_table(), on, seconds, characters)


def loudness(samples, rate: int):
    """Loudness in dBFS of each 10 ms of a mono signal."""
    import numpy as np

    size = int(rate * HOP)
    usable = len(samples) // size * size
    if not usable:
        return np.zeros(0)
    power = np.mean(np.square(samples[:usable].reshape(-1, size), dtype=np.float64), axis=1)
    return 10 * np.log10(np.maximum(power, 1e-12))


def _runs(mask) -> list[tuple[int, int, bool]]:
    import numpy as np

    if len(mask) == 0:
        return []
    bounds = [0, *(np.flatnonzero(np.diff(mask.astype(np.int8))) + 1).tolist(), len(mask)]
    return [(a, b, bool(mask[a])) for a, b in zip(bounds, bounds[1:])]


def voiced(db, close: float = 0.12, blip: float = 0.05):
    """Which 10 ms frames hold a voice: near the level of the loud frames, short gaps closed, blips dropped."""
    import numpy as np

    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    active = db > max(float(np.percentile(db, 95)) - SPEECH_RANGE, -50.0)
    for a, b, on in _runs(active):
        if not on and 0 < a and b < len(active) and (b - a) * HOP < close:
            active[a:b] = True
    for a, b, on in _runs(active):
        if on and (b - a) * HOP < blip:
            active[a:b] = False
    return active


def spans(active) -> list[tuple[float, float]]:
    """(start, end) in seconds of every stretch of voice."""
    return [(a * HOP, b * HOP) for a, b, on in _runs(active) if on]


def stem_talk(project: Project) -> list[tuple[float, float]] | None:
    """Every stretch of speech in the original speech stem, or None when there is no stem."""
    import numpy as np

    stem = project.root / "audio" / "speech.wav"
    if not stem.is_file():
        return None
    return spans(voiced(loudness(np.frombuffer(read_pcm_f32(stem, 8000), dtype=np.float32), 8000)))


def utterances(project: Project, lines: list[dict]) -> list[dict]:
    """Where each line's ORIGINAL utterance starts and ends, measured on the speech stem.

    The picture's mouths move with the original performance, so this is when a mouth is moving
    for the line. Whisper's word stamps, which the script's time slots start from, sit about a
    quarter second before the real onset after a pause (over a second in places), which is enough
    to see. Lines with no speech under them in the stem come back with `found` false.
    """
    talk = stem_talk(project)
    if talk is None:
        return [{"start": line["start"], "end": line["end"], "found": False} for line in lines]
    found = []
    for i, line in enumerate(lines):
        low = line["start"] - 0.15
        following = lines[i + 1]["start"] if i + 1 < len(lines) else float("inf")
        near = [(a, b) for a, b in talk if b > low and a < line["end"] - 0.03]
        while len(near) > 1 and near[0][0] < low and near[0][1] < line["start"] + 0.2:
            near = near[1:]  # the tail of the line before, not the start of this one
        if not near:
            found.append({"start": line["start"], "end": line["end"], "found": False})
            continue
        start = near[0][0] if near[0][0] >= low else line["start"]  # already talking: the script's split stands
        end = near[-1][1]
        if end > following + 0.25:  # the same breath runs on into the next line
            end = min(line["end"], following)
        found.append({"start": round(start, 2), "end": round(max(end, start + 0.1), 2), "found": True})
    for one, two in zip(found, found[1:]):
        if one["end"] > two["start"]:
            one["end"] = max(two["start"], one["start"] + 0.1)
    return found


def dump_script(script: dict) -> str:
    """lines.json text with one line of the script per row, so it stays readable and easy to edit."""
    head = {key: value for key, value in script.items() if key != "lines"}
    rows = ",\n".join("    " + json.dumps(line, ensure_ascii=False) for line in script["lines"])
    return json.dumps(head, indent=2, ensure_ascii=False)[:-2] + ',\n  "lines": [\n' + rows + "\n  ]\n}\n"


def anchor(project: Project, name: str) -> list[dict]:
    """Move every line's time slot onto its original utterance, and say what moved.

    After this the slots are when the mouths move, so voices, captions and shot prompts that are
    built from the slots line up with the picture.
    """
    path = folder(project, name) / "lines.json"
    script = load_lines(project, name)
    moved = []
    for line, real in zip(script["lines"], utterances(project, script["lines"])):
        if real["found"] and (abs(real["start"] - line["start"]) > 0.005 or abs(real["end"] - line["end"]) > 0.005):
            moved.append({"id": line["id"], "start": round(real["start"] - line["start"], 2), "end": round(real["end"] - line["end"], 2)})
            line["start"], line["end"] = real["start"], real["end"]
    if moved:
        path.write_text(dump_script(script), encoding="utf-8")
    return moved


def fit_speed(seconds: float, target: float) -> float:
    """The tempo change that makes a take's words last as long as the original utterance, within what sounds natural."""
    if target <= 0 or seconds <= 0:
        return 1.0
    # A word or two can be tightened further than a sentence before it sounds rushed.
    return min(max(seconds / target, MIN_SPEED), SHORT_SPEED if target < SHORT_LINE else MAX_SPEED)


def cue_parts(clip, line: dict) -> tuple[int, int]:
    """Samples taken by a vocal cue in front of a take's first word and after its last one.

    A cue is a sound of its own (a sigh, a chuckle) set off from the words by a short pause, so it
    is taken as everything before the first pause, or after the last. With no pause to find, the
    answer is 0 and the line's timing has to be judged by ear.
    """
    say = line.get("say") or ""
    front, back = re.match(r"\s*<([^<>]*)>", say), re.search(r"<([^<>]*)>\s*$", say)
    wants_lead = bool(front) and front.group(1).strip().lower() not in CUE_SECONDS
    wants_tail = bool(back) and back.group(1).strip().lower() not in CUE_SECONDS and bool(TAG.sub("", say[: back.start()]).strip())
    if not (wants_lead or wants_tail):
        return 0, 0
    runs, length = spans(voiced(loudness(clip, RATE), close=0.10)), len(clip) / RATE
    lead = int(runs[1][0] * RATE) if wants_lead and len(runs) >= 2 and runs[1][0] < 0.6 * length else 0
    tail = len(clip) - int(runs[-2][1] * RATE) if wants_tail and len(runs) >= 2 and runs[-1][0] > max(0.4 * length, lead / RATE) else 0
    return lead, tail


def word_span(clip, lead: int = 0, tail: int = 0) -> tuple[int, int]:
    """Samples where a take's audible words begin and end, between any cues.

    AI takes open with a soft breath and trail off quietly. Measured the way the original speech is
    measured, the words start a little after the clip does, and that is the point that has to land
    on the mouth's first movement.
    """
    low, high = lead / RATE, (len(clip) - tail) / RATE
    inside = [(a, b) for a, b in spans(voiced(loudness(clip, RATE))) if b > low + 0.01 and a < high - 0.01]
    if not inside:
        return lead, len(clip) - tail
    return max(int(inside[0][0] * RATE), lead), min(int(inside[-1][1] * RATE), len(clip) - tail)


def trim_silence(samples, floor: float = 0.004, margin: int = RATE // 30):
    """Drop the quiet lead-in and tail a speech model adds, keeping a few milliseconds of margin."""
    import numpy as np

    loud = np.flatnonzero(np.abs(samples) > floor)
    if len(loud) == 0:
        return samples[:0]
    return samples[max(loud[0] - margin, 0) : loud[-1] + margin]


def _decode(wav: bytes, speed: float = 1.0):
    """WAV bytes to mono float samples at RATE, optionally sped up without changing pitch."""
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        handle.write(wav)
        handle.flush()
        if speed == 1.0:
            return np.frombuffer(read_pcm_f32(Path(handle.name), RATE), dtype=np.float32)
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", handle.name,
             "-af", f"atempo={speed:.4f}", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
            capture_output=True, check=True,
        )
        return np.frombuffer(proc.stdout, dtype=np.float32)


def _encode(samples) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def too_long(text: str, seconds: float) -> bool:
    """Whether a spoken line took far longer than its words should: a sign the model said something extra."""
    expected = len(text.split()) / 2.3 + 0.4  # about 140 words a minute plus a breath
    return seconds > 2.2 * expected + 1.0


def generate(
    project: Project, name: str, synth=google.synthesize, workers: int = 4, pilot: int = 0, redo: tuple[int, ...] = (),
    rerolls: int = 2, provider: str = "gemini",
) -> dict:
    """Speak the lines of lines.json, fit each to its slot, and write voice.wav plus a report.

    A line's slot is the time of its original utterance (see `anchor`): its first word is placed
    on the slot's start and its tempo is changed by at most 12% so the words end with the slot.

    With `pilot`, only the first that many lines are spoken, into voice_pilot.wav: a cheap check
    of voices and pacing before paying for the whole script.
    """
    import numpy as np
    import soundfile as sf

    script = load_lines(project, name)
    lines = script["lines"][:pilot] if pilot else script["lines"]
    if not lines:
        raise VoiceError("lines.json has no lines to speak")
    duration = project.load()["source"]["duration"]

    # Every spoken line is kept as a take. A rerun speaks only the lines whose words, cues, voice or
    # style changed, plus any listed in `redo`, so good takes survive and a fix costs one line.
    takes = folder(project, name) / ("takes_pilot" if pilot else "takes")
    takes.mkdir(parents=True, exist_ok=True)

    def speak(line: dict, pace: str = "", attempt: int = 0):
        import hashlib

        # The performed text carries the vocal cues; a line's own `style` replaces its speaker's delivery note.
        style = line.get("style") or script["directions"].get(line["speaker"], "")
        voice_name = script["voices"][line["speaker"]]
        parts = [performed(line), voice_name, style]
        if hasattr(synth, "fingerprint"):  # a take made elsewhere (ElevenLabs): the file's own bytes decide whether it changed
            parts.append(synth.fingerprint(performed(line)))
        key = hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        take, note = takes / f"{line['id']:04d}.wav", takes / f"{line['id']:04d}.json"
        if not attempt and line["id"] not in redo and take.is_file() and note.is_file() and json.loads(note.read_text(encoding="utf-8")).get("key") == key:
            return take.read_bytes(), {}
        wav, usage = synth(performed(line), voice_name, ", ".join(part for part in (style, pace) if part))
        if attempt:  # every paid take is kept; the one that fits best becomes the line's take below
            (takes / f"{line['id']:04d}_try{attempt}.wav").write_bytes(wav)
        else:
            take.write_bytes(wav)
            note.write_text(json.dumps({"key": key, **usage}), encoding="utf-8")
        return wav, {**usage, "respoken": 1, "key": key}

    def tried(line: dict) -> int:
        note = takes / f"{line['id']:04d}.json"
        return json.loads(note.read_text(encoding="utf-8")).get("tries", 0) if note.is_file() else 0

    def measure(wav: bytes, line: dict) -> float:
        """Seconds a take's words still end after the slot's end once its tempo is fitted (negative: before)."""
        clip = trim_silence(_decode(wav))
        first, last = word_span(clip, *cue_parts(clip, line))
        words, target = (last - first) / RATE, max(line["end"] - line["start"], 0.2)
        return words / fit_speed(words, target) - target

    tokens, rerolled = {"text_tokens": 0, "audio_tokens": 0, "characters": 0, "respoken": 0}, set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        spoken = list(pool.map(speak, lines))
        for _, usage in spoken:
            for key in tokens:
                tokens[key] += usage.get(key, 0)
        # A take that cannot be brought to the length of the original utterance is spoken again with a
        # word about pace, which sounds better than squeezing or stretching it. The closest take wins.
        for attempt in range(1, rerolls + 1):
            off = [(i, measure(spoken[i][0], line)) for i, line in enumerate(lines)]
            # A line is retaken at most `rerolls` times in its life, not on every run: once it has been
            # tried, its take stays put, so a video that was checked keeps the voices it was checked with.
            off = [(i, error) for i, error in off if abs(error) > FIT and tried(lines[i]) < attempt]
            if not off:
                continue
            retakes = list(pool.map(
                lambda item: speak(lines[item[0]], "speaking quickly" if item[1] > 0 else "speaking slowly, unhurried", attempt), off
            ))
            for (i, error), (wav, usage) in zip(off, retakes):
                for key in tokens:
                    tokens[key] += usage.get(key, 0)
                note = takes / f"{lines[i]['id']:04d}.json"
                kept = json.loads(note.read_text(encoding="utf-8")) if note.is_file() else {"key": usage["key"]}
                if abs(measure(wav, lines[i])) < abs(error):
                    spoken[i] = (wav, usage)
                    rerolled.add(lines[i]["id"])
                    (takes / f"{lines[i]['id']:04d}.wav").write_bytes(wav)
                    kept = {"key": usage["key"], "try": attempt}
                note.write_text(json.dumps({**kept, "tries": attempt}), encoding="utf-8")

    def fade(clip, at_start: bool):
        clip = clip.copy()
        ramp = np.linspace(0.0, 1.0, min(RATE // 50, len(clip)), dtype=np.float32)
        if at_start:
            clip[: len(ramp)] *= ramp
        else:
            clip[len(clip) - len(ramp) :] *= ramp[::-1]
        return clip

    track = np.zeros(int(duration * RATE) + RATE, dtype=np.float32)
    report, placed_end = [], 0.0
    for i, (line, (wav, usage)) in enumerate(zip(lines, spoken)):
        clip = trim_silence(_decode(wav))
        natural = len(clip) / RATE
        lead, tail = cue_parts(clip, line)
        first, last = word_span(clip, lead, tail)
        target = max(line["end"] - line["start"], 0.2)
        speed = fit_speed((last - first) / RATE, target)
        if abs(speed - 1.0) < 0.02:
            speed = 1.0
        else:
            clip = _decode(_encode(clip), speed)
            lead, tail = int(lead / speed), int(tail / speed)
            first, last = word_span(clip, lead, tail)
        following = lines[i + 1]["start"] if i + 1 < len(lines) else duration
        notes = []
        if not lead and lead_seconds(line) > CUE_SECONDS["long pause"]:
            notes.append("leading cue not told apart from the words: check where the first word lands")
        # The first audible WORD lands on the slot's start, where the mouth starts moving. A breath
        # or a cue in front of it plays in the gap before the line; what does not fit there is cut.
        begin_at = line["start"] - first / RATE
        earliest = max(placed_end + 0.03, 0.0)
        if begin_at < earliest:
            cut = min(int((earliest - begin_at) * RATE) + 1, first)
            if lead and cut > RATE // 20:
                notes.append("leading cue left out or shortened: no gap before the line")
            clip, first, last, lead = fade(clip[cut:], True), first - cut, last - cut, max(lead - cut, 0)
            begin_at = line["start"] - first / RATE
        if tail and begin_at + len(clip) / RATE > following - GAP:
            clip, tail = fade(clip[: len(clip) - tail], False), 0
            notes.append("trailing cue left out: the next line starts too soon")
        begin = int(max(begin_at, 0.0) * RATE)
        clip = clip[: max(len(track) - begin, 0)]
        track[begin : begin + len(clip)] += clip
        words_start, words_end = begin_at + first / RATE, begin_at + last / RATE
        placed_end = begin_at + len(clip) / RATE
        report.append(
            {
                "id": line["id"],
                "speaker": line["speaker"],
                "begin": round(begin_at, 2),
                "words_start": round(words_start, 2),
                "words_end": round(words_end, 2),
                "end": round(placed_end, 2),
                "seconds": round(len(clip) / RATE, 2),
                "slot": round(target, 2),
                "speed": round(speed, 2),
                "end_error": round(words_end - line["end"], 2),  # against the end of the original utterance
                "overrun": round(max(words_end - (following - GAP), 0), 2),  # into the next line
                "past_end": round(max(words_end - line["end"], 0), 2),
                "cues": cues(line),
                "notes": notes,
                "retaken": line["id"] in rerolled,
                "suspect": too_long(TAG.sub(" ", performed(line)), natural - cue_seconds(line)),
            }
        )

    # Sit the new voices at the level the original voices had.
    track = track[: int(duration * RATE)]
    speech_stem = project.root / "audio" / "speech.wav"
    if speech_stem.is_file():
        old = np.frombuffer(read_pcm_f32(speech_stem, RATE), dtype=np.float32)
        new_rms, old_rms = float(np.sqrt(np.mean(np.square(track)))), float(np.sqrt(np.mean(np.square(old))))
        if new_rms > 0 and old_rms > 0:
            track = track * (old_rms / new_rms)
    peak = float(np.abs(track).max())
    if peak > 0.98:
        track = track * (0.98 / peak)

    out = folder(project, name)
    voice = out / ("voice_pilot.wav" if pilot else "voice.wav")
    sf.write(str(voice), track, RATE, subtype="PCM_16")

    today = date.today()
    table = cost.load_table()
    if provider == "elevenlabs":  # spoken in the user's own account: credits, not dollars
        actual = [table.line("ElevenLabs speech (plan credits)", "elevenlabs.speech", tokens["characters"], today)]
        planned = 0.0
    else:
        actual = [
            table.line("new voice audio", "flash-tts.audio-out", tokens["audio_tokens"], today),
            table.line("script text sent to the voice model", "flash-tts.text-in", tokens["text_tokens"], today),
        ]
        planned = cost.total(estimate({**script, "lines": lines}, today))
    for item in actual:
        if tokens["respoken"]:  # a run that reused every take made no paid call
            cost.record(project.slug, out.name, item, estimated_usd=planned, note="pilot" if pilot else "")
    summary = {
        "voices": script["voices"],
        "script": script_hash(script["lines"]),
        "file": voice.name,
        "lines": len(lines),
        "with_cues": sum(1 for line in lines if cues(line)),
        "respoken": tokens["respoken"],
        "sped_up": sum(1 for r in report if r["speed"] > 1.0),
        "slowed": sum(1 for r in report if r["speed"] < 1.0),
        "retaken": len(rerolled),
        "off": sum(1 for r in report if abs(r["end_error"]) > 0.3),
        "overrun": sum(1 for r in report if r["overrun"] > 0.05),
        "suspect": sum(1 for r in report if r["suspect"]),
        "usd": cost.total(actual),
        "estimated_usd": planned,
        "tokens": {key: tokens[key] for key in ("text_tokens", "audio_tokens")},
        "provider": provider,
        "characters": tokens["characters"],
        "report": report,
    }
    (out / ("voice_pilot.json" if pilot else "voice.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# --- captions -----------------------------------------------------------------------------------


def split_cue(start: float, end: float, text: str, max_chars: int = CAPTION_CHARS) -> list[dict]:
    """Break one line into caption-sized cues, sharing its time in proportion to the text."""
    words, chunks, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    total = sum(len(c) for c in chunks) or 1
    cues, cursor = [], start
    for chunk in chunks:
        length = (end - start) * len(chunk) / total
        cues.append({"start": round(cursor, 3), "end": round(cursor + length, 3), "text": chunk})
        cursor += length
    return cues


def _ass_time(seconds: float) -> str:
    centis = int(round(max(seconds, 0) * 100))
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    return f"{hours}:{minutes:02d}:{rest // 100:02d}.{rest % 100:02d}"


def write_captions(project: Project, name: str, size: tuple[int, int] | None = None, stem: str = "captions") -> dict:
    """<stem>.srt and a styled <stem>.ass (one color per speaker) from the remix's lines.

    `size` is the picture the captions will be burned into (width, height); it defaults to the source video's.
    """
    from .steps.transcript import to_srt

    script = load_lines(project, name)
    info = project.load()["source"]
    width, height = size or (info["width"], info["height"])
    info = {**info, "width": width, "height": height}
    out = folder(project, name)
    speakers = list(script["voices"])
    cues = [
        {**cue, "speaker": line["speaker"]}
        for line in script["lines"]
        if line.get("captions", True)  # a line whose words are already on screen as text can opt out
        for cue in split_cue(line["start"], line["end"], line["text"])
    ]
    (out / f"{stem}.srt").write_text(to_srt(cues), encoding="utf-8")

    size = max(round(info["height"] * 0.042), 18)
    header = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {info['width']}", f"PlayResY: {info['height']}",
        "WrapStyle: 0", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, "
        "Outline, Shadow, Alignment, MarginL, MarginR, MarginV",
    ]
    for i, speaker in enumerate(speakers):
        color = CAPTION_COLORS[i % len(CAPTION_COLORS)]
        header.append(
            f"Style: {slugify(speaker)},DejaVu Sans,{size},{color},&H00000000,&H80000000,1,1,"
            f"{max(size // 12, 2)},0,2,40,40,{round(info['height'] * 0.22)}"
        )
    header += ["", "[Events]", "Format: Layer, Start, End, Style, Text"]
    events = [
        f"Dialogue: 0,{_ass_time(c['start'])},{_ass_time(c['end'])},{slugify(c['speaker'])},"
        + c["text"].replace("\n", " ").replace("{", "(").replace("}", ")")
        for c in cues
    ]
    (out / f"{stem}.ass").write_text("\n".join(header + events) + "\n", encoding="utf-8")
    return {"cues": len(cues), "outputs": [project.rel(out / f"{stem}.srt"), project.rel(out / f"{stem}.ass")]}


def burn_captions(project: Project, name: str, video: Path) -> Path:
    """Render captions.ass into a copy of a video. The picture is re-encoded; the sound is copied."""
    out = folder(project, name)
    if not (out / "captions.ass").is_file():
        raise VoiceError("write the captions first")
    target = out / f"{video.stem}_captioned.mp4"
    # Run inside the remix folder so the subtitles filter gets a plain file name with nothing to escape.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(video.resolve()),
         "-vf", "subtitles=captions.ass", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-c:a", "copy", target.name],
        cwd=out, check=True,
    )
    return target


def fit_note(overrun: int, lines: int) -> str:
    if not overrun:
        return "every line fits its slot"
    return f"{overrun} of {lines} lines run into the next line even at {MAX_SPEED}x; shorten those lines"
