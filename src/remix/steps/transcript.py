"""Transcribe the speech stem with word-level timing (faster-whisper)."""
from __future__ import annotations

import json
import math
import os

from ..media import read_pcm_f32
from ..project import Project

PARAGRAPH_GAP = 1.5  # seconds of silence that starts a new paragraph in script.txt
WHISPER_SAMPLE_RATE = 16000
# In the separated speech stem, real lines measured -17 to -24 dBFS and text Whisper invented over
# non-speech measured -43 dBFS (synthetic music leaking in) and -65 dBFS (a real video's quiet tail).
SPEECH_FLOOR_DBFS = -35.0


def stem_level(samples, start: float, end: float, rate: int = WHISPER_SAMPLE_RATE) -> float:
    """RMS level in dBFS of the samples between two times."""
    import numpy as np

    chunk = samples[int(start * rate) : int(end * rate)]
    if len(chunk) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    return 20 * math.log10(max(rms, 1e-9))


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def join_words(words: list[dict]) -> str:
    """Rebuild text from timed words. A word flagged 'glue' attaches to the one before it,

    which covers hyphenated tails ("self" + "-test") and languages written without spaces.
    """
    text = ""
    for word in words:
        text += word["word"] if not text or word.get("glue") else " " + word["word"]
    return text


def caption_cues(words: list[dict], max_chars: int = 42, max_gap: float = 0.8) -> list[dict]:
    """Group timed words into caption-sized cues, breaking at sentence ends, pauses and line length.

    Whisper often returns one long segment for continuous speech, which is useless as a subtitle.
    """
    cues: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            cues.append({"start": current[0]["start"], "end": current[-1]["end"], "text": join_words(current)})
            current.clear()

    for word in words:
        if current and not word.get("glue"):
            if (
                len(join_words(current + [word])) > max_chars
                or word["start"] - current[-1]["end"] > max_gap
                or current[-1]["word"].endswith((".", "?", "!"))
            ):
                flush()
        current.append(word)
    flush()
    return cues


def to_srt(cues: list[dict]) -> str:
    return "\n".join(
        f"{i}\n{format_srt_time(cue['start'])} --> {format_srt_time(cue['end'])}\n{cue['text']}\n"
        for i, cue in enumerate(cues, 1)
    )


def to_script(segments: list[dict]) -> str:
    paragraphs: list[list[str]] = []
    last_end = None
    for seg in segments:
        if last_end is None or seg["start"] - last_end > PARAGRAPH_GAP:
            paragraphs.append([])
        paragraphs[-1].append(seg["text"])
        last_end = seg["end"]
    return "\n\n".join(" ".join(lines) for lines in paragraphs) + ("\n" if paragraphs else "")


def run(project: Project, model_name: str, language: str | None = None) -> dict:
    audio_dir = project.root / "audio"
    speech, original = audio_dir / "speech.wav", audio_dir / "original.wav"
    source = speech if speech.is_file() else original
    if not source.is_file():
        return {"status": "skipped", "reason": "no audio to transcribe"}

    import ctranslate2
    import numpy as np
    from faster_whisper import WhisperModel

    # Decode with ffmpeg and hand Whisper the samples: faster-whisper's own file decoder
    # passes PyAV an argument that current PyAV releases no longer accept.
    samples = np.frombuffer(read_pcm_f32(source, WHISPER_SAMPLE_RATE), dtype=np.float32).copy()

    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = WhisperModel(
        model_name,
        device=device,
        compute_type="int8",
        cpu_threads=max(4, (os.cpu_count() or 8) // 2),
    )
    # Voice-activity detection clips quiet line openings, and without it Whisper invents text over
    # silence. On the separated stem the level tells the two apart, so VAD stays off there and
    # segments with no voice energy are dropped (and recorded) instead.
    from_stem = source == speech
    found, info = model.transcribe(
        samples,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=not from_stem,
        vad_parameters=None if from_stem else {"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )

    segments, dropped = [], []
    for seg in found:
        text = seg.text.strip()
        if not text:
            continue
        level = round(stem_level(samples, seg.start, seg.end), 1)
        if from_stem and level < SPEECH_FLOOR_DBFS:
            dropped.append(
                {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text, "level_dbfs": level}
            )
            continue
        words = []
        for w in seg.words or []:
            word = {
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "word": w.word.strip(),
                "prob": round(w.probability, 3),
            }
            if words and not w.word.startswith(" "):
                word["glue"] = True  # Whisper marks a new word with a leading space
            words.append(word)
        segments.append(
            {
                "id": len(segments) + 1,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text,
                "words": words,
            }
        )

    out_dir = project.root / "transcript"
    out_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "model": model_name,
        "audio": project.rel(source),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "segments": segments,
        "text": " ".join(seg["text"] for seg in segments),
        "dropped_as_silence": dropped,
    }
    json_path, srt_path, script_path = (
        out_dir / "transcript.json",
        out_dir / "transcript.srt",
        out_dir / "script.txt",
    )
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    words = [word for seg in segments for word in seg["words"]]
    srt_path.write_text(to_srt(caption_cues(words) if words else segments), encoding="utf-8")
    script_path.write_text(to_script(segments), encoding="utf-8")

    return {
        "status": "done",
        "model": model_name,
        "device": device,
        "language": info.language,
        "segments": len(segments),
        "words": len(words),
        "dropped_as_silence": len(dropped),
        "outputs": [project.rel(p) for p in (json_path, srt_path, script_path)],
    }
