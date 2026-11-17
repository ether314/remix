"""Remix outputs. Each remix lives in projects/<slug>/remixes/<name>/ with a remix.json record."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from .media import audio_seconds, ffmpeg, probe, read_pcm_f32
from .project import Project, now_iso, slugify

LEVEL_RATE = 22050
FADE_IN, FADE_OUT = 0.5, 2.0
END_GUARD = 0.3  # fade used when a track's own ending is kept
TEMPO_FIT = 0.08  # largest tempo change used to make a track end with the picture


class RemixError(RuntimeError):
    pass


def level_dbfs(path: Path, start: float | None = None, seconds: float | None = None) -> float:
    """RMS level of a file's audio (or of one window of it), measured on its mono mix."""
    import numpy as np

    samples = np.frombuffer(read_pcm_f32(path, LEVEL_RATE, start, seconds), dtype=np.float32)
    if len(samples) == 0:
        raise RemixError(f"{path.name} has no audio")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20 * math.log10(max(rms, 1e-9))


def compare_levels(project: Project, output: Path) -> dict:
    """Source versus remix loudness in two windows: where the talking starts, and the ending.

    Nobody has to trust that a mix sounds balanced: these numbers show whether it sits where the
    original did.
    """
    duration = project.load()["source"]["duration"]
    first_word = 0.0
    transcript = project.root / "transcript" / "transcript.json"
    if transcript.is_file():
        segments = json.loads(transcript.read_text(encoding="utf-8"))["segments"]
        first_word = segments[0]["start"] if segments else 0.0
    windows = {
        "talking": (min(first_word, max(duration - 20, 0)), min(20.0, duration)),
        "ending": (max(duration - 10.5, 0), min(9.0, duration)),
    }
    return {
        name: {
            "start": round(start, 1),
            "seconds": round(seconds, 1),
            "source_dbfs": round(level_dbfs(project.source, start, seconds), 1),
            "remix_dbfs": round(level_dbfs(output, start, seconds), 1),
        }
        for name, (start, seconds) in windows.items()
    }


def audio_remix(
    project: Project,
    name: str,
    music: Path,
    music_gain_db: float = 0.0,
    keep_speech: bool = True,
    voice: Path | None = None,
) -> dict:
    """Put new music under the original picture, with the original voices, new ones, or none.

    The music is looped or trimmed to the video's length and set to the level the old background
    had, so the balance against the voices carries over.
    """
    project.require_reusable()
    audio_dir = project.root / "audio"
    speech, background = audio_dir / "speech.wav", audio_dir / "background.wav"
    if not (speech.is_file() and background.is_file()):
        raise RemixError("the stems are missing: run `remix decompose` on this project first")
    if not music.is_file():
        raise RemixError(f"music file not found: {music}")
    if voice is not None:
        if not voice.is_file():
            raise RemixError(f"voice track not found: {voice}")
        speech = voice

    folder = project.root / "remixes" / slugify(name)
    folder.mkdir(parents=True, exist_ok=True)
    script = None
    if voice is not None and (folder / "lines.json").is_file():
        from . import voices

        script = voices.script_hash(json.loads((folder / "lines.json").read_text(encoding="utf-8"))["lines"])
        report = voice.with_suffix(".json")
        spoken_from = json.loads(report.read_text(encoding="utf-8")).get("script") if report.is_file() else None
        if spoken_from and spoken_from != script:
            raise RemixError(
                "voice.wav was spoken from a different version of lines.json, so it would not match the captions "
                "or the picture: rerun `remix voices` first"
            )
    music_copy = folder / f"music{music.suffix.lower()}"
    if music.resolve() != music_copy.resolve():
        shutil.copyfile(music, music_copy)

    gain = level_dbfs(background) - level_dbfs(music_copy) + music_gain_db
    source = project.source
    duration = project.load()["source"]["duration"]

    # A track within 8% of the video's length is nudged in tempo to end with the picture, which
    # keeps its own ending. Anything shorter loops; anything longer is cut with a fade.
    ratio = audio_seconds(music_copy) / duration
    fade_out = FADE_OUT
    if abs(ratio - 1) <= TEMPO_FIT:
        fit, tempo, loop = f"tempo changed {ratio - 1:+.1%} to end with the picture", f"atempo={ratio:.5f},", []
        fade_out = END_GUARD  # the track brings its own ending; only guard against a click
    elif ratio < 1:
        fit, tempo, loop = "looped to fill the video", "", ["-stream_loop", "-1"]
    else:
        fit, tempo, loop = "cut at the end of the video", "", []

    stereo = "aformat=sample_rates=48000:channel_layouts=stereo"
    limiter = "alimiter=limit=0.95:level=0"  # level=0: catch peaks without re-normalizing the mix
    bed = (
        f"[2:a]{stereo},{tempo}atrim=0:{duration:.3f},volume={gain:.2f}dB,"
        f"afade=t=in:d={FADE_IN},afade=t=out:st={max(duration - fade_out, 0):.3f}:d={fade_out}"
    )
    if keep_speech:
        graph = (
            f"[1:a]{stereo}[speech];{bed}[music];"
            f"[speech][music]amix=inputs=2:duration=longest:normalize=0,{limiter}[mix]"
        )
    else:
        graph = f"{bed},{limiter}[mix]"

    matroska = source.suffix.lower() in (".mkv", ".webm")
    output = folder / ("output.mkv" if matroska else "output.mp4")
    ffmpeg(
        "-i", str(source),
        "-i", str(speech),
        *loop, "-i", str(music_copy),
        "-filter_complex", graph,
        "-map", "0:v:0", "-map", "[mix]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.3f}",
        *([] if matroska else ["-movflags", "+faststart"]),
        str(output),
    )

    record = {
        "name": folder.name,
        "mode": "audio",
        "created": now_iso(),
        "voices": "none" if not keep_speech else "new" if voice is not None else "original",
        "script": script,
        "music": project.rel(music_copy),
        "music_gain_db": round(gain, 2),
        "requested_trim_db": music_gain_db,
        "music_fit": fit,
        "output": project.rel(output),
        "duration": probe(output)["duration"],
        "levels": compare_levels(project, output),
    }
    (folder / "remix.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
