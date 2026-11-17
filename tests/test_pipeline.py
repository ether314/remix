"""Runs the model-free steps against a tiny generated video (needs ffmpeg, not the ML models)."""
import json
import subprocess

import pytest

from remix import blueprint, config, overview
from remix.project import ingest
from remix.steps import audio, shots


@pytest.fixture()
def two_shot_video(tmp_path):
    path = tmp_path / "Two Shots.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2",
            "-f", "lavfi", "-i", "smptebars=size=640x360:rate=30:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        check=True,
    )
    return path


def test_ingest_audio_shots_overview(two_shot_video, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")

    project, created = ingest(two_shot_video)
    assert created and project.slug.startswith("two-shots-")
    assert ingest(two_shot_video) == (project, False)

    assert audio.run(project)["status"] == "done"
    assert (project.root / "audio" / "original.wav").is_file()

    result = shots.run(project)
    assert result["shots"] == 2 and result["keyframes_failed"] == 0
    doc = json.loads((project.root / "shots" / "shots.json").read_text())
    assert abs(doc["shots"][0]["end"] - 2.0) < 0.2
    assert all((project.root / k["file"]).is_file() for s in doc["shots"] for k in s["keyframes"])
    assert (project.root / doc["sheets"][0]["file"]).is_file()

    assert overview.build(project)["status"] == "done"
    assert "| 2 |" in (project.root / "overview.md").read_text()

    skeleton = blueprint.init(project)
    assert skeleton["metrics"]["shot_count"] == 2 and skeleton["metrics"]["aspect"] == "16:9"
    assert [shot["index"] for shot in skeleton["shots"]] == [1, 2]
    assert "summary is empty" in blueprint.check(project)
    assert not (project.root / "blueprint.md").exists()
