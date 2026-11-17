"""Prices, per-video estimates and the spend ledger.

Rule of the project: every paid call gets an estimate before it runs and a ledger line after.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from . import config
from .project import now_iso

PRICES_PATH = Path(__file__).with_name("prices.yaml")
LEDGER_NAME = "ledger.jsonl"
CREDIT_NAME = "credit.json"
SUMMARY_KEYS = ("provider", "model", "project", "remix", "day", "month")


class CostError(RuntimeError):
    pass


@dataclass(frozen=True)
class Line:
    """One priced item: what it is, how much of it, and what that costs."""

    what: str
    provider: str
    model: str
    price_id: str
    quantity: float
    unit: str
    unit_usd: float
    usd: float


@dataclass(frozen=True)
class Option:
    """One way of doing a remix, priced."""

    mode: str
    name: str
    lines: tuple[Line, ...]
    note: str = ""

    @property
    def usd(self) -> float:
        return total(self.lines)


class PriceTable:
    def __init__(self, data: dict):
        self.prices: list[dict] = data["prices"]
        self.rates: dict = data.get("rates", {})
        self.limits: dict = data.get("limits", {})
        self.sources: dict = data.get("sources", {})

    def price(self, price_id: str, on: date) -> dict:
        matches = [
            p
            for p in self.prices
            if p["id"] == price_id and p.get("from", date.min) <= on <= p.get("until", date.max)
        ]
        if not matches:
            raise CostError(f"prices.yaml has no price for '{price_id}' on {on}")
        if len(matches) > 1:
            raise CostError(f"prices.yaml has overlapping prices for '{price_id}' on {on}")
        return matches[0]

    def line(self, what: str, price_id: str, quantity: float, on: date) -> Line:
        price = self.price(price_id, on)
        per = 1_000_000 if price["unit"] == "1M tokens" else 1
        return Line(
            what=what,
            provider=price["provider"],
            model=price["model"],
            price_id=price_id,
            quantity=round(quantity, 3),
            unit="tokens" if per > 1 else price["unit"],
            unit_usd=price["usd"],
            usd=round(quantity / per * price["usd"], 4),
        )

    def current(self, on: date) -> list[dict]:
        return [p for p in self.prices if p.get("from", date.min) <= on <= p.get("until", date.max)]


def load_table(path: Path = PRICES_PATH) -> PriceTable:
    import yaml

    return PriceTable(yaml.safe_load(path.read_text(encoding="utf-8")))


def total(lines) -> float:
    return round(sum(line.usd for line in lines), 4)


def within_cap(usd: float) -> bool:
    """Whether a paid step may run without asking first."""
    return usd <= config.CAP_USD


# --- how much has to be generated ---------------------------------------------------------------


def omni_seconds(shots: list[float], table: PriceTable) -> float:
    """Seconds Gemini Omni must produce for these shots: clips run from 3 to 15 s, whole seconds."""
    shortest, longest = table.limits["omni.clip-seconds"]
    seconds = 0.0
    for length in shots:
        pieces = max(1, math.ceil(length / longest))
        seconds += max(shortest * pieces, math.ceil(length))
    return seconds


def veo_seconds(shots: list[float], table: PriceTable) -> float:
    """Seconds Veo must produce for these shots: every clip is 4, 6 or 8 s long."""
    allowed = sorted(table.limits["veo-3.1.clip-seconds"])
    seconds = 0.0
    for length in shots:
        pieces = max(1, math.ceil(length / allowed[-1]))
        piece = length / pieces
        seconds += pieces * next(a for a in allowed if a >= piece - 1e-9)
    return seconds


# --- estimators ---------------------------------------------------------------------------------


def music(table: PriceTable, on: date, seconds: float) -> list[Line]:
    songs = max(1, math.ceil(seconds / table.limits["lyria-3-pro.max-seconds"]))
    return [table.line(f"new music ({songs} song{'' if songs == 1 else 's'})", "lyria-3-pro.song", songs, on)]


def speech(table: PriceTable, on: date, seconds: float, characters: int = 0, model: str = "flash-tts") -> list[Line]:
    lines = [table.line("new voice audio", f"{model}.audio-out", seconds * table.rates["tts.audio-out"], on)]
    if characters:
        lines.append(table.line("script text sent to the voice model", f"{model}.text-in", characters / 4, on))
    return lines


def omni_video(
    table: PriceTable, on: date, seconds: float, resolution: str, source_seconds: float = 0.0
) -> list[Line]:
    lines = [
        table.line(
            f"video at {resolution}",
            "omni-1.2-flash.video-out",
            seconds * table.rates["omni.video-out"][resolution],
            on,
        )
    ]
    if source_seconds:
        lines.append(
            table.line(
                "source video sent to the model (upper bound)",
                "omni-1.2-flash.input",
                source_seconds * table.rates["omni.video-in-max"],
                on,
            )
        )
    return lines


def veo_video(table: PriceTable, on: date, seconds: float, resolution: str, tier: str) -> list[Line]:
    return [table.line(f"silent video at {resolution}", f"{tier}.video.{resolution}", seconds, on)]


def keyframes(table: PriceTable, on: date, count: int) -> list[Line]:
    return [table.line("restyled keyframe images", "nano-banana-2.1.image.1k", count, on)]


def options_for(
    table: PriceTable,
    on: date,
    duration: float,
    shots: list[float],
    speech_seconds: float,
    script_characters: int,
) -> list[Option]:
    """Every remix route for one video, priced side by side."""
    omni = omni_seconds(shots, table)
    veo = veo_seconds(shots, table)
    frames = keyframes(table, on, len(shots))
    clips = f"{len(shots)} shots"
    voice = speech(table, on, speech_seconds, script_characters)
    return [
        Option("audio remix", "new music, original voices", tuple(music(table, on, duration))),
        Option("audio remix", "new music and new voices", tuple(music(table, on, duration) + voice)),
        Option("re-voice / translate", "Gemini 3.8 Flash TTS", tuple(voice)),
        Option(
            "re-voice / translate",
            "Gemini 3.8 Flash-Lite TTS",
            tuple(speech(table, on, speech_seconds, script_characters, model="flash-lite-tts")),
        ),
        Option(
            "restyle", "Gemini Omni 1.2 Flash, 360p draft",
            tuple(omni_video(table, on, omni, "360p", source_seconds=omni)), f"{clips}, {omni:.0f} s to generate",
        ),
        Option(
            "restyle", "Gemini Omni 1.2 Flash, 720p",
            tuple(omni_video(table, on, omni, "720p", source_seconds=omni)), f"{clips}, {omni:.0f} s to generate",
        ),
        Option(
            "restyle", "Gemini Omni 1.2 Flash, 1080p",
            tuple(omni_video(table, on, omni, "1080p", source_seconds=omni)), f"{clips}, {omni:.0f} s to generate",
        ),
        Option(
            "regenerate", "Veo 3.1 Lite, 720p, silent",
            tuple(veo_video(table, on, veo, "720p", "veo-3.1-lite") + frames), f"{clips}, {veo:.0f} s to generate",
        ),
        Option(
            "regenerate", "Veo 3.1 Fast, 720p, silent",
            tuple(veo_video(table, on, veo, "720p", "veo-3.1-fast") + frames), f"{clips}, {veo:.0f} s to generate",
        ),
        Option(
            "regenerate", "Gemini Omni 1.2 Flash, 720p",
            tuple(omni_video(table, on, omni, "720p") + frames), f"{clips}, {omni:.0f} s to generate",
        ),
        Option(
            "regenerate", "Veo 3.1, 720p, silent",
            tuple(veo_video(table, on, veo, "720p", "veo-3.1") + frames), f"{clips}, {veo:.0f} s to generate",
        ),
        Option("motion-graphics rebuild", "rendered from code, no provider", (), "free"),
    ]


def estimate_project(project, on: date, table: PriceTable | None = None) -> list[Option]:
    """Price every remix route for a decomposed project from its real shots and speech."""
    from .blueprint import spoken_seconds

    root = project.root
    timeline_path = root / "timeline.json"
    if not timeline_path.is_file():
        raise CostError("timeline.json is missing: run `remix decompose` on this project first")
    shots = [shot["duration"] for shot in json.loads(timeline_path.read_text(encoding="utf-8"))["shots"]]
    words, characters = [], 0
    transcript_path = root / "transcript" / "transcript.json"
    if transcript_path.is_file():
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        words = [word for seg in transcript["segments"] for word in seg["words"]]
        characters = len(transcript["text"])
    return options_for(
        table or load_table(), on, project.load()["source"]["duration"], shots, spoken_seconds(words), characters
    )


def render_options(title: str, options: list[Option], on: date) -> str:
    lines = [
        f"# Cost estimate: {title}",
        "",
        f"Prices as of {on}. One pass each, with no retakes; visual modes usually need 1.5-2x. "
        f"Options above the ${config.CAP_USD:g} auto-run cap are marked.",
        "",
        "| Mode | Option | Estimate | Over cap | Notes |",
        "|---|---|---|---|---|",
    ]
    for option in options:
        over = "" if within_cap(option.usd) else "asks first"
        lines.append(f"| {option.mode} | {option.name} | ${option.usd:.2f} | {over} | {option.note} |")
    lines += ["", "## Line items", ""]
    for option in options:
        if not option.lines:
            continue
        lines.append(f"**{option.mode}: {option.name}** (${option.usd:.2f})")
        for item in option.lines:
            per = " per 1M" if item.unit == "tokens" else f" per {item.unit}"
            lines.append(
                f"- {item.what}: {item.quantity:,.0f} {item.unit} on {item.model} "
                f"at ${item.unit_usd:g}{per} = ${item.usd:.2f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- ledger -------------------------------------------------------------------------------------


def ledger_path() -> Path:
    return config.PROJECTS_DIR / LEDGER_NAME


def record(project: str, remix: str, line: Line, estimated_usd: float | None = None, note: str = "") -> dict:
    """Append one paid call to the ledger. `line` carries the units actually billed."""
    entry = {"at": now_iso(), "project": project, "remix": remix, **asdict(line), "estimated_usd": estimated_usd}
    if note:
        entry["note"] = note
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def entries() -> list[dict]:
    path = ledger_path()
    if not path.is_file():
        return []
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]


def credit_path() -> Path:
    return config.PROJECTS_DIR / CREDIT_NAME


def set_credit(name: str, provider: str, usd: float, ends: date, as_of: date) -> dict:
    """Record a provider credit balance as read on a date, so the report can show what is left."""
    credit = {"name": name, "provider": provider, "usd": usd, "as_of": as_of.isoformat(), "ends": ends.isoformat()}
    path = credit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(credit, indent=2), encoding="utf-8")
    return credit


def credit_status(rows: list[dict], today: date) -> dict | None:
    """The recorded credit minus what the ledger shows spent with that provider since it was read."""
    path = credit_path()
    if not path.is_file():
        return None
    credit = json.loads(path.read_text(encoding="utf-8"))
    spent = sum(
        row["usd"] for row in rows if row["provider"] == credit["provider"] and row["at"][:10] >= credit["as_of"]
    )
    return {
        **credit,
        "spent": round(spent, 4),
        "left": round(credit["usd"] - spent, 4),
        "days_left": (date.fromisoformat(credit["ends"]) - today).days,
    }


def summarize(rows: list[dict], by: str) -> list[tuple[str, float, int]]:
    """(key, dollars, calls) sorted by key, grouping ledger rows by a field or by day/month."""
    if by not in SUMMARY_KEYS:
        raise CostError(f"group by one of: {', '.join(SUMMARY_KEYS)}")
    dollars: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    for row in rows:
        key = row["at"][:10] if by == "day" else row["at"][:7] if by == "month" else str(row.get(by) or "-")
        dollars[key] += row["usd"]
        calls[key] += 1
    return [(key, round(dollars[key], 4), calls[key]) for key in sorted(dollars)]
