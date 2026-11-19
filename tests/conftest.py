"""Shared fixtures: a tiny decomposed-enough project built with ffmpeg (no ML models)."""
import shutil
import subprocess

import pytest

from remix import config
from remix.project import ingest
from remix.steps import audio


@pytest.fixture(autouse=True)
def no_real_elevenlabs(monkeypatch):
    """No test reaches the real ElevenLabs API, even when the user's key is in the environment (Docker Compose passes it in
    from .env). Tests that exercise the SDK path set a made-up key and a fake client themselves."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_MODEL", raising=False)
    monkeypatch.delenv("ELEVENLABS_MUSIC_MODEL", raising=False)


def make_tone(path, frequency: int, seconds: float, volume: float = 1.0):
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}",
            "-af", f"volume={volume}", str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture()
def tone():
    return make_tone


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config, "INBOX_DIR", tmp_path / "inbox")
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(video),
        ],
        check=True,
    )
    made, _ = ingest(video)
    audio.run(made)
    # Stand-ins for the Demucs stems: these tests are about mixing and serving, not separation.
    for name in ("speech.wav", "background.wav"):
        shutil.copyfile(made.root / "audio" / "original.wav", made.root / "audio" / name)
    return made
