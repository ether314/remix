"""Check that a remix's video says and shows what its script says, in sync.

Standing rule from the user: every remix, on every regeneration, is checked before it is handed
over or approved. The script (lines.json, or the original transcript when a remix keeps the
original voices) is the reference; the audio, the captions and the picture are compared with it.

Sync is enforced by code, not by remembering to look:
- every part records the fingerprint of the script it was made from, and a mismatch fails;
- the voice track is located inside the finished video's sound and its offset is measured;
- picture length, sound length and frame count are compared with the source;
- every caption cue is compared with its line's words and time slot.
What code cannot judge, whether the action in a regenerated shot suits its line, is laid out on
frame sheets (`check/sheet_NN.jpg`) for Claude to read before passing `--frames-ok`.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path

from . import config, voices
from .media import ffmpeg, read_pcm_f32, run as run_tool
from .project import Project, now_iso, slugify

MATCH = 0.8  # share of a line's words that must be heard, in order
DRIFT = 0.75  # seconds a line may start away from its slot, as heard by Whisper
REACH = 1.5  # heard words further than this from a line's slot do not count as that line
OFFSET = 0.06  # seconds the voice track may sit early or late in the finished video
LENGTH = 0.15  # seconds picture, sound and source may differ in length
OVERLAP = 0.05  # seconds two spoken lines may sound at once
# Gates against the original utterance, which is when the picture's mouths move. From a review of a
# finished remix on 2026-11-17: beyond these the mismatch is visible.
START = 0.30  # seconds the voice may start before or after the mouth does
END_LATE, END_EARLY = 0.30, 0.50  # seconds the voice may go on after the mouth stops, or stop before it does
FOREIGN = 0.15  # extra seconds, over the original, a line may be heard on a shot of another speaker
CUE = 0.05  # seconds a caption cue may differ from its line's slot
FOUND = 0.4  # correlation below which the voice track was not found in the video's sound


def words_of(text: str) -> list[str]:
    plain = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", plain.replace("-", " ")).split()


def listen(path: Path) -> list[dict]:
    """Every word spoken in a media file, with its start time."""
    import numpy as np
    from faster_whisper import WhisperModel

    audio = np.frombuffer(read_pcm_f32(path, 16000), dtype=np.float32).copy()
    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=6)
    found, _ = model.transcribe(
        audio, beam_size=5, word_timestamps=True, vad_filter=False, condition_on_previous_text=False
    )
    return [{"word": w.word, "start": w.start, "end": w.end} for seg in found for w in (seg.words or [])]


def stream_facts(path: Path) -> dict:
    """Length and frame count of a file's first video and audio streams (None where the container does not say)."""
    out = run_tool(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,nb_frames", "-of", "json", str(path)]).stdout
    facts: dict = {}
    for stream in json.loads(out).get("streams", []):
        kind = stream.get("codec_type")
        if kind in ("video", "audio") and kind not in facts:
            try:
                seconds = float(stream.get("duration"))
            except (TypeError, ValueError):
                seconds = None
            frames = int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None
            facts[kind] = {"seconds": seconds, "frames": frames}
    return facts


def voice_offset(output: Path, voice: Path, reach: float = 2.0) -> tuple[float, float]:
    """Where the voice track sits in a finished video's sound: (seconds late, how clearly it was found 0-1).

    Compares loudness envelopes in 10 ms steps, so music mixed over the voices does not hide them.
    """
    import numpy as np

    def envelope(path: Path):
        samples = np.frombuffer(read_pcm_f32(path, 8000), dtype=np.float32)
        usable = len(samples) // 80 * 80
        return np.sqrt(np.mean(np.square(samples[:usable].reshape(-1, 80), dtype=np.float64), axis=1))

    mixed, dry = envelope(output), envelope(voice)
    n, hops = min(len(mixed), len(dry)), int(reach * 100)
    if n <= 2 * hops + 100:
        hops = max((n - 20) // 4, 0)
    if n < 20:
        return 0.0, 0.0
    mixed, dry = mixed[:n] - mixed[:n].mean(), dry[:n] - dry[:n].mean()
    best, score = 0, -1.0
    for lag in range(-hops, hops + 1):  # lag > 0: the voices arrive late in the video
        a, b = mixed[max(lag, 0) : n + min(lag, 0)], dry[max(-lag, 0) : n - max(lag, 0)]
        scale = float(np.sqrt(a.dot(a) * b.dot(b)))
        value = float(a.dot(b)) / scale if scale else 0.0
        if value > score:
            best, score = lag, value
    return best / 100, max(score, 0.0)


def audio_problems(lines: list[dict], heard_words: list[dict]) -> list[dict]:
    problems = []
    script_words = {word for line in lines for word in words_of(voices.TAG.sub(" ", voices.performed(line)))}
    # Line the whole transcript up with the whole script once, in order, then judge each line by the
    # heard words it was matched to. A fixed time window per line lost first words stamped a little early.
    heard = [(token, word["start"], word.get("end", word["start"])) for word in heard_words for token in words_of(word["word"])]
    script = [(token, i) for i, line in enumerate(lines) for token in words_of(line["text"])]
    matched: dict[int, list[int]] = {i: [] for i in range(len(lines))}
    aligner = difflib.SequenceMatcher(None, [token for token, _ in script], [token for token, _, _ in heard], autojunk=False)
    for block in aligner.get_matching_blocks():
        for k in range(block.size):
            matched[script[block.a + k][1]].append(block.b + k)
    for i, line in enumerate(lines):
        # A vocal cue such as <sigh> must be performed, never read out as the word "sigh".
        cue_words = {word for tag in voices.cues(line) for word in words_of(tag)} - script_words
        following = lines[i + 1]["start"] if i + 1 < len(lines) else float("inf")
        read_aloud = sorted({
            token for word in heard_words if line["start"] - 0.2 <= word["start"] < following
            for token in words_of(word["word"]) if token in cue_words
        })
        if read_aloud:
            problems.append({"line": line["id"], "problem": f"the voice read a vocal cue aloud ('{' '.join(read_aloud)}') instead of performing it",
                             "script": voices.performed(line)})
        expected = words_of(line["text"])
        if not expected:
            continue
        near = [n for n in matched[i] if line["start"] - REACH <= heard[n][1] < following + REACH]
        if len(near) / len(expected) < MATCH:
            around = [token for token, start, _ in heard if line["start"] - 0.5 <= start < following + 0.4]
            problems.append({"line": line["id"], "problem": "the audio does not say this line",
                             "script": line["text"], "heard": " ".join(around)[:90]})
            continue
        # Whisper often stamps the first word after a pause early, so a word that spans the slot's start is on time.
        _, began, ended = heard[near[0]]
        off = 0.0 if began <= line["start"] <= ended else (began if began > line["start"] else ended) - line["start"]
        if not -DRIFT <= off <= DRIFT + voices.lead_seconds(line):
            problems.append({"line": line["id"], "problem": f"the line starts {off:+.1f} s off its slot", "script": line["text"]})
    return problems


def picture_problems(project: Project, lines: list[dict], record: dict, frames_ok: bool) -> list[dict]:
    if record.get("mode") != "audio":  # a regenerated picture: someone has to read its frames
        return [] if frames_ok else [{"problem": "the frames have not been read against the script: check captions "
                                      "and action shot by shot on the sheets in check/, then rerun with --frames-ok"}]
    transcript = project.root / "transcript" / "transcript.json"
    if not transcript.is_file():
        return []
    original = " " + " ".join(words_of(json.loads(transcript.read_text(encoding="utf-8"))["text"])) + " "
    changed = [line["id"] for line in lines if " " + " ".join(words_of(line["text"])) + " " not in original]
    if not changed:
        return []
    return [{"problem": f"the picture is the original, but {len(changed)} line(s) were rewritten, so its mouths and any "
             f"baked-in captions no longer match the audio", "lines": changed}]


def _srt_seconds(stamp: str) -> float:
    hours, minutes, rest = stamp.strip().replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def caption_problems(folder: Path, lines: list[dict]) -> list[dict]:
    """captions.srt must be exactly what the script produces: the same words in the same time slots."""
    captions = folder / "captions.srt"
    if not captions.is_file():
        return []
    shown = []
    for block in re.split(r"\n\s*\n", captions.read_text(encoding="utf-8").strip()):
        rows = [row for row in block.splitlines() if row.strip()]
        timing = next((row for row in rows if "-->" in row), None)
        if timing:
            start, end = (_srt_seconds(part) for part in timing.split("-->"))
            shown.append((start, end, " ".join(rows[rows.index(timing) + 1 :])))
    wanted = [cue for line in lines for cue in voices.split_cue(line["start"], line["end"], line["text"])]
    if words_of(" ".join(text for _, _, text in shown)) != words_of(" ".join(cue["text"] for cue in wanted)):
        return [{"problem": "captions.srt does not match the script: rerun `remix captions`"}]
    late = [
        cue["text"] for cue, (start, end, _) in zip(wanted, shown)
        if len(wanted) == len(shown) and (abs(cue["start"] - start) > CUE or abs(cue["end"] - end) > CUE)
    ]
    if len(wanted) != len(shown) or late:
        return [{"problem": f"captions.srt has the script's words at other times than the script ({len(late) or 'cue count differs'}): "
                 "rerun `remix captions`"}]
    return []


def script_problems(project: Project, folder: Path, record: dict, lines: list[dict]) -> list[dict]:
    """Everything in the video must have been made from the script as it is now."""
    problems, current = [], voices.script_hash(lines)
    if record.get("script") and record["script"] != current:
        problems.append({"problem": "lines.json changed after this video was made, so the video no longer matches its script: make it again"})
    sound = project.root / "remixes" / slugify(record["audio_from"]) if record.get("audio_from") else folder
    report = sound / "voice.json"
    if record.get("voices") == "new" and report.is_file():
        spoken_from = json.loads(report.read_text(encoding="utf-8")).get("script")
        if spoken_from and spoken_from != current:
            problems.append({"problem": "the voices were spoken from a different version of the script: rerun `remix voices`, then remake the video"})
    timeline = project.root / "timeline.json"
    if record.get("mode") == "restyle" and not record.get("dry_run") and timeline.is_file():
        wrong = []
        for shot in json.loads(timeline.read_text(encoding="utf-8"))["shots"]:
            note = folder / "shots" / f"{shot['index']:04d}_spoken.txt"
            if note.is_file() and words_of(note.read_text(encoding="utf-8")) != words_of(voices.spoken_in(shot, lines)):
                wrong.append(shot["index"])
        if wrong:
            problems.append({"problem": f"{len(wrong)} shot(s) were generated for a different line than the script now gives them: "
                             f"rerun `remix restyle` to regenerate shots {wrong}"})
    return problems


def timing_problems(project: Project, folder: Path, record: dict, output: Path, offset=voice_offset) -> tuple[list[dict], dict]:
    """Measured sync: lengths, frame count, where the voice track sits, and lines that run over."""
    problems, measured = [], {}
    facts = stream_facts(output)
    picture, sound = (facts.get(kind, {}).get("seconds") for kind in ("video", "audio"))
    source = project.load()["source"]["duration"]
    measured.update({"picture_seconds": picture, "sound_seconds": sound, "source_seconds": source})
    if picture is not None and sound is not None and abs(picture - sound) > LENGTH:
        problems.append({"problem": f"the picture is {picture:.2f} s long and the sound {sound:.2f} s"})
    if picture is not None and abs(picture - source) > LENGTH:
        problems.append({"problem": f"the picture is {picture:.2f} s long, the source video {source:.2f} s: cuts no longer line up"})
    frames = facts.get("video", {}).get("frames")
    if record.get("frames") and frames is not None:
        measured["frames"] = {"made": frames, "expected": record["frames"]}
        if abs(frames - record["frames"]) > 3:
            problems.append({"problem": f"the picture has {frames} frames where the shots add up to {record['frames']}"})
    if record.get("voices") == "new":
        sound_folder = project.root / "remixes" / slugify(record["audio_from"]) if record.get("audio_from") else folder
        if (sound_folder / "voice.wav").is_file():
            late, found = offset(output, sound_folder / "voice.wav")
            measured["voice_offset_seconds"], measured["voice_found"] = round(late, 2), round(found, 2)
            if found < FOUND:
                problems.append({"problem": "the voice track could not be found in the video's sound: the video was not made from this voice.wav"})
            elif abs(late) > OFFSET:
                problems.append({"problem": f"the voices sit {late:+.2f} s off the picture"})
        report = sound_folder / "voice.json"
        rows = json.loads(report.read_text(encoding="utf-8")).get("report", []) if report.is_file() else []
        for one, two in zip(rows, rows[1:]):
            if "end" in one and "begin" in two and one["end"] - two["begin"] > OVERLAP:
                problems.append({"line": one["id"], "problem": f"this line and the next are heard at once for {one['end'] - two['begin']:.2f} s: shorten it"})
    return problems, measured


def on_screen(project: Project, folder: Path) -> dict[int, set[str]]:
    """Which characters each shot shows: the blueprint's lists, plus any the remix's look.json adds for a shot."""
    blueprint, look = project.root / "blueprint.json", folder / "look.json"
    shown: dict[int, set[str]] = {}
    if blueprint.is_file():
        for character in json.loads(blueprint.read_text(encoding="utf-8")).get("characters", []):
            for number in character.get("shots") or []:
                shown.setdefault(number, set()).add(character["id"])
    if look.is_file():
        for number, extra in (json.loads(look.read_text(encoding="utf-8")).get("shots") or {}).items():
            if extra.get("characters"):
                shown[int(number)] = set(extra["characters"])
    return shown


def sync_problems(project: Project, folder: Path, record: dict, lines: list[dict]) -> tuple[list[dict], dict]:
    """Each line's voice against its ORIGINAL utterance, measured on both tracks. No transcription.

    The picture's mouths move when the original line was spoken. A voice that starts or stops away
    from that, or that is heard over a shot of the other speaker, is what looks out of sync, and no
    whole-track offset or word check can see it.
    """
    import numpy as np

    sound = project.root / "remixes" / slugify(record["audio_from"]) if record.get("audio_from") else folder
    if record.get("voices") != "new" or not (sound / "voice.wav").is_file():
        return [], {}
    real = voices.utterances(project, lines)
    active = voices.voiced(voices.loudness(np.frombuffer(read_pcm_f32(sound / "voice.wav", 8000), dtype=np.float32), 8000))
    report = _read_report(sound)
    for row in report.values():  # a sigh or a chuckle around the words is a sound, not a moving mouth saying the line
        for a, b in ((row.get("begin"), row.get("words_start")), (row.get("words_end"), row.get("end"))):
            if a is not None and b is not None and b - a > 0.02:
                active[int(a / voices.HOP) : int(b / voices.HOP)] = False
    timeline = project.root / "timeline.json"
    shots = json.loads(timeline.read_text(encoding="utf-8"))["shots"] if timeline.is_file() else []
    shown, talkers = on_screen(project, folder), {line["speaker"] for line in lines}

    def on_others(speaker: str, start: float, end: float) -> float:
        """Seconds of [start, end] that fall on shots showing another speaker and not this one."""
        return sum(
            max(min(end, shot["end"]) - max(start, shot["start"]), 0.0) for shot in shots
            if speaker not in shown.get(shot["index"], set()) and (shown.get(shot["index"], set()) & talkers)
        )

    problems, starts, ends = [], [], []
    for i, (line, utterance) in enumerate(zip(lines, real)):
        if not utterance["found"]:
            continue
        low = (real[i - 1]["end"] + utterance["start"]) / 2 if i else 0.0
        high = (utterance["end"] + real[i + 1]["start"]) / 2 if i + 1 < len(lines) else len(active) * voices.HOP
        frames = np.flatnonzero(active[int(low / voices.HOP) : int(high / voices.HOP)])
        if len(frames) == 0:
            problems.append({"line": line["id"], "problem": "no voice is heard while this line's mouth moves"})
            continue
        begin, finish = int(low / voices.HOP) * voices.HOP + frames[0] * voices.HOP, int(low / voices.HOP) * voices.HOP + (frames[-1] + 1) * voices.HOP
        early, late = begin - utterance["start"], finish - utterance["end"]
        starts.append(early)
        ends.append(late)
        if abs(early) > START:
            problems.append({"line": line["id"], "problem": f"the voice starts {early:+.2f} s from where the mouth starts moving ({utterance['start']:.2f} s)"})
        if late > END_LATE or late < -END_EARLY:
            problems.append({"line": line["id"], "problem": f"the voice stops {late:+.2f} s from where the mouth stops ({utterance['end']:.2f} s)"})
        extra = on_others(line["speaker"], begin, finish) - on_others(line["speaker"], utterance["start"], utterance["end"])
        if extra > FOREIGN:
            problems.append({"line": line["id"], "problem": f"{extra:.2f} s more of this line than in the original is heard over a shot of another speaker"})
    measured = {
        "lines_timed": len(starts),
        "mean_start_error": round(float(np.mean(np.abs(starts))), 2) if starts else None,
        "mean_end_error": round(float(np.mean(np.abs(ends))), 2) if ends else None,
        "worst_start_error": round(float(max(starts, key=abs)), 2) if starts else None,
        "worst_end_error": round(float(max(ends, key=abs)), 2) if ends else None,
    }
    return problems, measured


def _read_report(sound: Path) -> dict[int, dict]:
    path = sound / "voice.json"
    return {row["id"]: row for row in json.loads(path.read_text(encoding="utf-8")).get("report", [])} if path.is_file() else {}


def frame_sheets(project: Project, folder: Path, output: Path, lines: list[dict], columns: int = 6, rows: int = 3) -> list[str]:
    """Frames of the finished video laid out for reading: three per spoken line, early, middle and late.

    Each tile is labeled with its shot, time and who is speaking, the caption that must be on
    screen at that moment, and who else is in the shot and so must have a closed mouth. Shots with
    no line get a tile from their middle, and every shot one from near its end.
    """
    from PIL import Image, ImageDraw, ImageFont

    timeline = project.root / "timeline.json"
    if not timeline.is_file():
        return []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    all_shots = json.loads(timeline.read_text(encoding="utf-8"))["shots"]
    every_cue = [cue for l in lines for cue in voices.split_cue(l["start"], l["end"], l["text"])]
    shown, talkers = on_screen(project, folder), {l["speaker"] for l in lines}

    def shot_at(moment: float) -> dict:
        return next((s for s in all_shots if s["start"] <= moment < s["end"]), all_shots[-1])

    def quiet(shot: dict, speaker: str | None) -> str:
        others = sorted((shown.get(shot["index"], set()) & talkers) - {speaker})
        return "mouth closed: " + ", ".join(others) if others else ""

    for line in lines:
        cues = voices.split_cue(line["start"], line["end"], line["text"])
        for share in (0.2, 0.55, 0.9):
            moment = line["start"] + share * (line["end"] - line["start"])
            cue = next((c for c in cues if c["start"] <= moment < c["end"]), cues[-1])
            shot = shot_at(moment)
            seen = "" if line["speaker"] in shown.get(shot["index"], {line["speaker"]}) else " (off screen)"
            tiles.append((shot["index"], moment, f"line {line['id']}: {line['speaker'].upper()} speaks{seen}", cue["text"], quiet(shot, line["speaker"])))
    for shot in all_shots:
        length, mine = shot["end"] - shot["start"], [when for n, when, *_ in tiles if n == shot["index"]]
        if not mine:
            tiles.append((shot["index"], shot["start"] + length / 2, "nobody speaks", "(no line)", quiet(shot, None)))
        # The video model sometimes invents a new picture late in a shot (a person walking in, a cut),
        # so each shot also shows a frame from near its end unless a line tile already sits there.
        late = shot["start"] + 0.9 * length
        if length >= 1.0 and not any(when >= shot["start"] + 0.7 * length for when in mine):
            showing = next((c["text"] for c in every_cue if c["start"] <= late < c["end"]), "(no caption)")
            speaking = next((l["speaker"] for l in lines if l["start"] <= late < l["end"]), None)
            tiles.append((shot["index"], late, "END OF SHOT", showing, quiet(shot, speaking)))
    tiles.sort(key=lambda tile: tile[1])

    check = folder / "check"
    check.mkdir(exist_ok=True)
    for old in check.glob("sheet_*.jpg"):
        old.unlink()
    width, label, written = 250, 76, []
    scrap = check / "_tile.jpg"
    for page in range(0, len(tiles), columns * rows):
        batch, sheet, height = tiles[page : page + columns * rows], None, 0
        for i, (shot, moment, who, caption, still) in enumerate(batch):
            ffmpeg("-ss", f"{moment:.3f}", "-i", str(output), "-frames:v", "1", "-update", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(scrap))
            frame = Image.open(scrap).convert("RGB")
            if sheet is None:
                height = frame.height
                sheet = Image.new("RGB", (columns * (width + 8) + 8, rows * (height + label + 8) + 8), (18, 18, 18))
            x, y = 8 + (i % columns) * (width + 8), 8 + (i // columns) * (height + label + 8)
            sheet.paste(frame.resize((width, height)), (x, y))
            draw = ImageDraw.Draw(sheet)
            draw.text((x, y + height + 2), f"shot {shot}  {int(moment // 60)}:{moment % 60:04.1f}  {who}", fill=(255, 210, 120), font=font)
            text, wrapped = caption, []
            while text:
                cut = text if len(text) <= 36 else text[: text.rfind(" ", 0, 36) if " " in text[:36] else 36]
                wrapped.append(cut.strip())
                text = text[len(cut) :].strip()
            for n, row in enumerate(wrapped[:2]):
                draw.text((x, y + height + 18 + n * 14), row, fill=(235, 235, 235), font=font)
            draw.text((x, y + height + 48), still, fill=(255, 140, 140), font=font)
        path = check / f"sheet_{page // (columns * rows) + 1:02d}.jpg"
        sheet.save(path, quality=88)
        written.append(project.rel(path))
    scrap.unlink(missing_ok=True)
    return written


def run(project: Project, name: str, frames_ok: bool = False, listen=listen, offset=voice_offset) -> dict:
    """Verify one remix and write verify.json next to it. `passed` is what approval depends on."""
    folder = voices.folder(project, name)
    record_path = folder / "remix.json"
    if not record_path.is_file():
        raise voices.VoiceError("this remix has no finished video to verify yet")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    script = folder / "lines.json"
    lines = (json.loads(script.read_text(encoding="utf-8")) if script.is_file() else voices.build_lines(project))["lines"]
    if record.get("voices") == "none":
        lines = []
    output = project.root / record["output"]
    timing, measured = timing_problems(project, folder, record, output, offset)
    sync, mouths = sync_problems(project, folder, record, lines)
    measured["against_original_speech"] = mouths
    gate = []
    if script.is_file() and lines:
        from . import dialogue

        gate = [{"problem": "dialogue gate: " + item} for item in dialogue.problems(project, name, ("script", "picture"))]
    problems = (
        script_problems(project, folder, record, lines)
        + timing
        + sync
        + gate
        + audio_problems(lines, listen(output))
        + picture_problems(project, lines, record, frames_ok)
        + caption_problems(folder, lines)
    )
    sheets = frame_sheets(project, folder, output, lines) if record.get("mode") != "audio" else []
    result = {"checked": now_iso(), "output": record["output"], "lines": len(lines), "passed": not problems,
              "frames_read": frames_ok, "measured": measured, "sheets": sheets, "problems": problems}
    (folder / "verify.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def passed(folder: Path) -> bool:
    """Whether the remix's current video has a passing check (a check older than the video does not count)."""
    check, record = folder / "verify.json", folder / "remix.json"
    if not (check.is_file() and record.is_file()):
        return False
    return json.loads(check.read_text(encoding="utf-8"))["passed"] and check.stat().st_mtime >= record.stat().st_mtime
