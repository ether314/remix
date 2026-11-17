"""Paths and model choices, overridable through environment variables."""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("REMIX_DATA", "/data"))
INBOX_DIR = DATA_DIR / "inbox"
PROJECTS_DIR = DATA_DIR / "projects"

WHISPER_MODEL = os.environ.get("REMIX_WHISPER_MODEL", "large-v3-turbo")
DEMUCS_MODEL = os.environ.get("REMIX_DEMUCS_MODEL", "htdemucs")

# Paid steps run without asking only while a remix's estimate stays at or under this many dollars.
CAP_USD = float(os.environ.get("REMIX_CAP_USD", "5"))
