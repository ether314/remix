"""Project folders: one per ingested video, with a manifest that tracks each step."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .media import probe

SCHEMA = 1
RIGHTS = ("own", "client", "template")


class ProjectError(RuntimeError):
    pass


def slugify(name: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:max_len].strip("-") or "video"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Project:
    root: Path

    @property
    def slug(self) -> str:
        return self.root.name

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def load(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save(self, manifest: dict) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    @property
    def source(self) -> Path:
        return self.root / self.load()["source"]["file"]

    def step(self, name: str) -> dict:
        return self.load()["steps"].get(name, {})

    def record(self, name: str, status: str, **fields) -> None:
        manifest = self.load()
        manifest["steps"][name] = {"status": status, "finished": now_iso(), **fields}
        self.save(manifest)

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def rights(self) -> str | None:
        return (self.load().get("rights") or {}).get("status")

    def set_rights(self, status: str, note: str = "") -> None:
        if status not in RIGHTS:
            raise ProjectError(f"rights must be one of: {', '.join(RIGHTS)}")
        manifest = self.load()
        manifest["rights"] = {"status": status, "set": now_iso(), **({"note": note} if note else {})}
        self.save(manifest)

    def require_reusable(self) -> None:
        """Raise unless this video's own footage, music and voices may appear in an output."""
        status = self.rights()
        if status == "template":
            raise ProjectError(
                "this project is a template: its footage, music and voices cannot be reused in an output"
            )
        if status is None:
            raise ProjectError(
                "this project has no rights recorded, so it is treated as a template; "
                f"run `remix rights {self.slug} own` (or client) if its footage may be reused"
            )


def resolve_file(ref: str) -> Path | None:
    """Find a video by path, by path under the data folder, or by name inside inbox/."""
    for candidate in (Path(ref), config.DATA_DIR / ref, config.INBOX_DIR / ref):
        if candidate.is_file():
            return candidate
    return None


def list_projects() -> list[Project]:
    root = config.PROJECTS_DIR
    if not root.is_dir():
        return []
    return [Project(p) for p in sorted(root.iterdir()) if (p / "manifest.json").is_file()]


def find(ref: str) -> Project:
    """Look a project up by slug or by a unique prefix of it."""
    name = Path(ref).name
    matches = [p for p in list_projects() if p.slug == name] or [
        p for p in list_projects() if p.slug.startswith(name)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ProjectError(f"no video file or project matches '{ref}'")
    raise ProjectError(f"'{ref}' matches several projects: " + ", ".join(p.slug for p in matches))


def ingest(path: Path) -> tuple[Project, bool]:
    """Copy a video into its own project folder. Returns the project and whether it is new."""
    sha = file_sha256(path)
    project = Project(config.PROJECTS_DIR / f"{slugify(path.stem)}-{sha[:8]}")
    if project.manifest_path.is_file():
        return project, False

    info = probe(path)  # fails here, before anything is created, if this is not a video
    project.root.mkdir(parents=True, exist_ok=True)
    target = project.root / f"source{path.suffix.lower()}"
    shutil.copyfile(path, target)
    project.save(
        {
            "schema": SCHEMA,
            "slug": project.slug,
            "created": now_iso(),
            "source": {
                "file": target.name,
                "original_name": path.name,
                "sha256": sha,
                "size_bytes": path.stat().st_size,
                **info,
            },
            "steps": {},
        }
    )
    return project, True
