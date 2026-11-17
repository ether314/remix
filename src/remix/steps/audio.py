"""Pull the soundtrack out of the source video as 44.1 kHz stereo WAV."""
from __future__ import annotations

from ..media import ffmpeg
from ..project import Project

SAMPLE_RATE = 44100  # what Demucs expects


def run(project: Project) -> dict:
    if not project.load()["source"]["has_audio"]:
        return {"status": "skipped", "reason": "source has no audio stream"}
    out = project.root / "audio" / "original.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        "-i", str(project.source),
        "-map", "0:a:0", "-vn",
        "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(out),
    )
    return {"status": "done", "outputs": [project.rel(out)]}
