"""Shot detection, keyframes, and contact sheets sized for Claude to read."""
from __future__ import annotations

import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..media import MediaError, ffmpeg
from ..project import Project

KEYFRAME_LONG_EDGE = 960
SECONDS_PER_KEYFRAME = 6.0
FRAMES_PER_SHEET = 12
MIN_SHOT_SECONDS = 0.5


def keyframe_times(start: float, end: float, max_frames: int = 8) -> list[float]:
    """Evenly spaced sample times inside a shot: one per ~6 s, at least one, at most max_frames."""
    length = max(end - start, 0.0)
    count = min(max(math.ceil(length / SECONDS_PER_KEYFRAME), 1), max_frames)
    return [round(start + (i + 0.5) * length / count, 3) for i in range(count)]


def fmt_time(seconds: float) -> str:
    minutes, secs = divmod(round(max(seconds, 0.0), 1), 60)
    return f"{int(minutes)}:{secs:04.1f}"


def sheet_layout(width: int, height: int) -> tuple[int, int, int]:
    """(columns, tile width, tile height) for 12-up sheets that stay under ~2000 px wide."""
    columns, tile_width = (4, 480) if width >= height else (6, 320)
    return columns, tile_width, max(2, round(tile_width * height / width))


def make_proxy(source: Path, out: Path, fps: float | None) -> float:
    """A small constant-frame-rate copy, so cut times are right even for variable-rate sources."""
    rate = min(fps or 30.0, 30.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = "scale='if(gt(iw,ih),-2,360)':'if(gt(iw,ih),360,-2)'"
    ffmpeg(
        "-i", str(source),
        "-map", "0:v:0", "-an",
        "-vf", f"fps={rate:.3f},{scale}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        str(out),
    )
    return rate


def detect(proxy: Path, detector: str, threshold: float | None) -> list[tuple[float, float]]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector

    video = open_video(str(proxy))
    min_len = max(1, round(MIN_SHOT_SECONDS * video.frame_rate))
    if detector == "content":
        chosen = ContentDetector(threshold=27.0 if threshold is None else threshold, min_scene_len=min_len)
    else:
        chosen = AdaptiveDetector(
            adaptive_threshold=3.0 if threshold is None else threshold, min_scene_len=min_len
        )
    manager = SceneManager()
    manager.add_detector(chosen)
    manager.detect_scenes(video, show_progress=False)
    return [(a.get_seconds(), b.get_seconds()) for a, b in manager.get_scene_list(start_in_scene=True)]


def extract_frame(source: Path, time: float, out: Path) -> bool:
    edge = KEYFRAME_LONG_EDGE
    scale = f"scale='if(gt(iw,ih),min({edge},iw),-2)':'if(gt(iw,ih),-2,min({edge},ih))'"
    try:
        ffmpeg(
            "-ss", f"{time:.3f}", "-i", str(source),
            "-map", "0:v:0", "-frames:v", "1", "-update", "1",
            "-vf", scale, "-q:v", "3",
            str(out),
        )
    except MediaError:
        return False
    return out.is_file() and out.stat().st_size > 0


def build_sheets(frames: list[dict], out_dir: Path) -> list[dict]:
    """Paste keyframes into labeled grids. Each frame dict needs 'path' and 'label'."""
    from PIL import Image, ImageDraw, ImageFont

    if not frames:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(frames[0]["path"]) as first:
        columns, tile_w, tile_h = sheet_layout(*first.size)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    pad, label_h = 8, 30
    sheets = []
    for number, offset in enumerate(range(0, len(frames), FRAMES_PER_SHEET), 1):
        batch = frames[offset : offset + FRAMES_PER_SHEET]
        rows = math.ceil(len(batch) / columns)
        sheet = Image.new(
            "RGB",
            (columns * tile_w + (columns + 1) * pad, rows * (tile_h + label_h + pad) + pad),
            (18, 18, 18),
        )
        draw = ImageDraw.Draw(sheet)
        for i, frame in enumerate(batch):
            x = pad + (i % columns) * (tile_w + pad)
            y = pad + (i // columns) * (tile_h + label_h + pad)
            with Image.open(frame["path"]) as image:
                tile = image.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x, y))
            draw.text((x + 4, y + tile_h + 3), frame["label"], fill=(240, 240, 240), font=font)
        path = out_dir / f"sheet_{number:02d}.jpg"
        sheet.save(path, quality=85)
        sheets.append({"path": path, "frames": batch})
    return sheets


def run(
    project: Project,
    detector: str = "adaptive",
    threshold: float | None = None,
    max_keyframes: int = 8,
) -> dict:
    info = project.load()["source"]
    source = project.source
    shots_dir = project.root / "shots"
    proxy = project.root / "video" / "analysis_360p.mp4"

    rate = make_proxy(source, proxy, info.get("fps"))
    spans = detect(proxy, detector, threshold) or [(0.0, info["duration"])]
    if info["duration"] > spans[-1][0]:
        spans[-1] = (spans[-1][0], info["duration"])  # the proxy can end a frame early or late

    key_dir = shots_dir / "keyframes"
    for stale in (key_dir, shots_dir / "sheets"):
        shutil.rmtree(stale, ignore_errors=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    shots, jobs = [], []
    for index, (start, end) in enumerate(spans, 1):
        shot = {
            "index": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "keyframes": [],
        }
        shots.append(shot)
        for k, time in enumerate(keyframe_times(start, end, max_keyframes), 1):
            jobs.append((shot, time, key_dir / f"shot_{index:04d}_{k}.jpg"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        extracted = list(pool.map(lambda job: extract_frame(source, job[1], job[2]), jobs))

    frames = []
    for (shot, time, path), ok in zip(jobs, extracted):
        if not ok:
            continue
        shot["keyframes"].append({"time": time, "file": project.rel(path)})
        frames.append(
            {"path": path, "label": f"S{shot['index']:02d}  {fmt_time(time)}", "shot": shot["index"], "time": time}
        )

    sheets = build_sheets(frames, shots_dir / "sheets")
    document = {
        "detector": detector,
        "threshold": threshold,
        "analysis_fps": rate,
        "shots": shots,
        "sheets": [
            {
                "file": project.rel(sheet["path"]),
                "frames": [{"shot": f["shot"], "time": f["time"]} for f in sheet["frames"]],
            }
            for sheet in sheets
        ],
    }
    shots_path = shots_dir / "shots.json"
    shots_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    return {
        "status": "done",
        "detector": detector,
        "shots": len(shots),
        "keyframes": len(frames),
        "keyframes_failed": len(jobs) - len(frames),
        "sheets": len(sheets),
        "outputs": [project.rel(shots_path), project.rel(proxy)] + [s["file"] for s in document["sheets"]],
    }
