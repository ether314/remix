"""Rights gating and the audio mixer, against a tiny generated video (needs ffmpeg, no ML models)."""
import json

import pytest

from remix import remixes
from remix.media import probe
from remix.project import ProjectError


def test_rights_decide_whether_footage_can_be_reused(project):
    with pytest.raises(ProjectError, match="no rights recorded"):
        project.require_reusable()
    project.set_rights("template")
    with pytest.raises(ProjectError, match="template"):
        project.require_reusable()
    project.set_rights("own", note="confirmed by the owner")
    project.require_reusable()
    assert project.load()["rights"]["note"] == "confirmed by the owner"
    with pytest.raises(ProjectError):
        project.set_rights("borrowed")


def test_audio_remix_sets_new_music_to_the_old_background_level(project, tmp_path, tone):
    music = tone(tmp_path / "bed.wav", 220, 1, volume=0.1)  # quiet and shorter than the video
    project.set_rights("own")

    record = remixes.audio_remix(project, "First Try", music)

    output = project.root / record["output"]
    info = probe(output)
    assert record["name"] == "first-try" and record["music_fit"] == "looped to fill the video"
    assert info["has_audio"] and abs(info["duration"] - 3.0) < 0.2
    old_level = remixes.level_dbfs(project.root / "audio" / "background.wav")
    assert record["music_gain_db"] > 0
    assert remixes.level_dbfs(music) + record["music_gain_db"] == pytest.approx(old_level, abs=0.1)
    assert json.loads((output.parent / "remix.json").read_text())["output"] == record["output"]
    # The same tone at the same level in both windows: the remix should sit close to the source.
    assert set(record["levels"]) == {"talking", "ending"}
    assert abs(record["levels"]["talking"]["source_dbfs"] - record["levels"]["talking"]["remix_dbfs"]) < 8


def test_music_close_to_the_video_length_is_fitted_by_tempo(project, tmp_path, tone):
    project.set_rights("own")
    nearly = remixes.audio_remix(project, "nearly", tone(tmp_path / "nearly.wav", 220, 2.85))
    longer = remixes.audio_remix(project, "longer", tone(tmp_path / "longer.wav", 220, 6))
    assert nearly["music_fit"].startswith("tempo changed -")
    assert longer["music_fit"] == "cut at the end of the video"
    assert abs(nearly["duration"] - 3.0) < 0.2 and abs(longer["duration"] - 3.0) < 0.2


def test_audio_remix_refuses_template_projects(project, tmp_path, tone):
    project.set_rights("template")
    with pytest.raises(ProjectError):
        remixes.audio_remix(project, "nope", tone(tmp_path / "bed.wav", 220, 1))
    assert not (project.root / "remixes").exists()
