"""Thin wrappers around ffmpeg and ffprobe."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

HDR_TRANSFERS = ("smpte2084", "arib-std-b67")


class MediaError(RuntimeError):
    """An ffmpeg/ffprobe call failed or the file is not usable video."""


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise MediaError(f"{cmd[0]} exited with {proc.returncode}:\n{tail}")
    return proc


def ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", *args])


def audio_seconds(path: Path) -> float:
    """Length of an audio (or any media) file in seconds."""
    out = run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise MediaError(f"could not read the length of {path.name}") from None


def read_pcm_f32(path: Path, sample_rate: int, start: float | None = None, seconds: float | None = None) -> bytes:
    """Decode any media file's audio (all of it, or one window) to mono 32-bit float PCM."""
    window = ([] if start is None else ["-ss", f"{start:.3f}"]) + ([] if seconds is None else ["-t", f"{seconds:.3f}"])
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
            *window,
            "-i", str(path),
            "-vn",
            "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.decode(errors="replace").strip().splitlines()[-12:])
        raise MediaError(f"ffmpeg exited with {proc.returncode}:\n{tail}")
    return proc.stdout


def parse_rate(value: str | None) -> float | None:
    """Turn an ffprobe rate such as '30000/1001' into frames per second."""
    if not value:
        return None
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return float(rate) if rate > 0 else None


def probe(path: Path) -> dict:
    """Facts about a video file that the pipeline needs, with rotation already applied."""
    out = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    ).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise MediaError(f"{path.name} has no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    avg = parse_rate(video.get("avg_frame_rate"))
    nominal = parse_rate(video.get("r_frame_rate"))
    fps = avg or nominal

    rotation = 0
    for side_data in video.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = int(round(float(side_data["rotation"])))
    if not rotation and (video.get("tags") or {}).get("rotate"):
        rotation = int(video["tags"]["rotate"])
    width, height = int(video["width"]), int(video["height"])
    if abs(rotation) % 180 == 90:
        width, height = height, width

    return {
        "duration": round(float(fmt.get("duration") or video.get("duration") or 0), 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3) if fps else None,
        "video_codec": video.get("codec_name"),
        "rotation": rotation,
        "vfr_suspected": bool(avg and nominal and abs(avg - nominal) / nominal > 0.02),
        "hdr": video.get("color_transfer") in HDR_TRANSFERS,
        "has_audio": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "audio_channels": int(audio["channels"]) if audio and audio.get("channels") else None,
        "container": fmt.get("format_name"),
    }
