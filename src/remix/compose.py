"""Draw the scenes of a made video: generated clips, product screens and title cards, with motion graphics on top.

Every scene is rendered frame by frame (PIL) and piped into ffmpeg, so on-screen text is always real
text in the brand's type, never something a video model drew. A scene is a dict from the plan:

    {"kind": "veo" | "image" | "title", "seconds": 4.0, "clip": "v1" (a generated shot), "background": "v4",
     "image": {"file": "assets/x.jpg", "crop": [x0, y0, x1, y1], "zoom": [1.0, 1.2], "focus": [0.5, 0.5]},
     "overlays": [{"type": "headline", "text": "...", "from": 0.2, "to": 2.4}, ...]}

Overlay types: headline, bubble, tabs, chips, timeline, highlight, logo, endcard. Times are seconds
from the start of the scene.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FPS = 30
VERSION = "4"  # bump when scenes are drawn differently: it is part of the render cache key
NAVY, TEAL, CYAN, DEEP, WHITE = (18, 40, 64), (30, 155, 170), (103, 232, 249), (23, 114, 126), (255, 255, 255)
DANGER = (200, 30, 30)
_FONTS: dict = {}


class ComposeError(RuntimeError):
    pass


def font(assets: Path, size: int, weight: int = 800, italic: bool = True) -> ImageFont.FreeTypeFont:
    key = (str(assets), size, weight, italic)
    if key not in _FONTS:
        path = assets / ("Montserrat-Italic.ttf" if italic else "Montserrat.ttf")
        try:
            loaded = ImageFont.truetype(str(path), size)
            loaded.set_variation_by_axes([weight])
        except (OSError, ValueError, AttributeError):
            loaded = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        _FONTS[key] = loaded
    return _FONTS[key]


def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def ease_out(x: float) -> float:
    return 1 - (1 - clamp(x)) ** 3


def ease_out_back(x: float) -> float:
    x = clamp(x)
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def lerp(a: float, b: float, x: float) -> float:
    return a + (b - a) * x


def window(t: float, start: float, end: float, fade_in: float = 0.18, fade_out: float = 0.18) -> float:
    """0 to 1 opacity of something shown from `start` to `end`."""
    if t < start or t > end:
        return 0.0
    return min(clamp((t - start) / fade_in), clamp((end - t) / fade_out) if fade_out else 1.0)


def wrap(text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and f.getlength(trial) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    return lines + ([current] if current else [])


def put_text(base: Image.Image, text: str, f: ImageFont.FreeTypeFont, center: tuple[float, float], color=WHITE,
             alpha: float = 1.0, scale: float = 1.0, shadow: bool = True) -> tuple[int, int, int, int]:
    """Draw text centered at `center` (optionally scaled and faded) and return its box."""
    if alpha <= 0:
        return (0, 0, 0, 0)
    left, top, right, bottom = f.getbbox(text)
    pad = 6
    mask = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    ImageDraw.Draw(mask).text((pad - left, pad - top), text, font=f, fill=255)
    if scale != 1.0:
        mask = mask.resize((max(1, int(mask.width * scale)), max(1, int(mask.height * scale))), Image.LANCZOS)
    x, y = int(center[0] - mask.width / 2), int(center[1] - mask.height / 2)
    faded = mask.point(lambda v: int(v * clamp(alpha)))
    if shadow:
        base.paste(Image.new("RGB", mask.size, (0, 0, 0)), (x + 3, y + 4), faded.point(lambda v: int(v * 0.45)))
    base.paste(Image.new("RGB", mask.size, color), (x, y), faded)
    return (x, y, x + mask.width, y + mask.height)


def rounded(base: Image.Image, box, radius: int, fill=None, outline=None, width: int = 2) -> None:
    ImageDraw.Draw(base, "RGBA").rounded_rectangle([int(v) for v in box], radius, fill=fill, outline=outline, width=width)


def hex_alpha(color, a: float):
    return (*color, int(255 * clamp(a)))


# --- backgrounds ----------------------------------------------------------------------------------


class Stream:
    """The brand's navy field with slow cyan currents, for scenes that have no footage behind them."""

    def __init__(self, size):
        self.size = size
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt(((xx - w * 0.55) / w) ** 2 + ((yy - h * 0.45) / h) ** 2)
        glow = np.clip(1 - dist * 1.5, 0, 1)[..., None]
        edge, mid = np.array((10, 24, 40), np.float32), np.array((26, 58, 90), np.float32)
        self.base = (edge + (mid - edge) * glow).astype(np.uint8)

    def frame(self, t: float) -> Image.Image:
        img = Image.fromarray(self.base.copy())
        draw = ImageDraw.Draw(img, "RGBA")
        w, h = self.size
        xs = np.linspace(0, w, 160)
        for i in range(6):
            ys = h * (0.22 + 0.11 * i) + h * 0.045 * np.sin(2 * math.pi * (xs / w * 1.1 - t * 0.12 + i * 0.6))
            draw.line([(float(x), float(y)) for x, y in zip(xs, ys)], fill=(*CYAN, 38 + 8 * i), width=max(2, h // 260))
        return img


def vignette(size, strength: float = 0.5, bottom: float = 0.0) -> np.ndarray:
    """A multiplier mask: darker corners, and optionally a darker bottom for captions to sit on."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = 1 - strength * np.clip(d - 0.55, 0, 1) ** 1.6
    if bottom:
        mask *= 1 - bottom * np.clip((yy / h - 0.62) / 0.38, 0, 1) ** 1.4
    return mask[..., None].astype(np.float32)


def graded(img: Image.Image, mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.asarray(img, np.float32) * mask, 0, 255).astype(np.uint8))


_BARS: dict = {}


def bars(clip: Path) -> str:
    """An ffmpeg crop filter that removes black bars baked into a clip ('' when it has none)."""
    key = (str(clip), clip.stat().st_size)
    if key not in _BARS:
        proc = subprocess.run(["ffmpeg", "-v", "info", "-ss", "0.5", "-t", "2.5", "-i", str(clip), "-an",
                               "-vf", "cropdetect=limit=28:round=2:reset=0", "-f", "null", "-"], capture_output=True, text=True)
        found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", proc.stderr)
        frame = re.search(r", (\d{3,5})x(\d{3,5})", proc.stderr)
        crop = ""
        if found and frame:
            cw, ch, cx, cy = map(int, found[-1])
            fw, fh = int(frame.group(1)), int(frame.group(2))
            if (cw < fw * 0.97 or ch < fh * 0.97) and cw > fw * 0.5 and ch > fh * 0.5:
                crop = f"crop={cw}:{ch}:{cx}:{cy},"
        _BARS[key] = crop
    return _BARS[key]


def decode(clip: Path, size, count: int, start: float = 0.0, speed: float = 1.0):
    """Yield `count` frames of a clip, scaled to cover `size` and cropped, repeating the last if it runs out."""
    w, h = size
    chain = bars(clip) + (f"setpts=PTS/{speed:.4f}," if abs(speed - 1) > 1e-3 else "")
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(clip), "-an",
           "-vf", f"{chain}scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    last, size_bytes = None, w * h * 3
    try:
        for _ in range(count):
            data = proc.stdout.read(size_bytes) if proc.stdout else b""
            if len(data) == size_bytes:
                last = Image.frombytes("RGB", (w, h), data)
            if last is None:
                raise ComposeError(f"could not read any frame of {clip.name}")
            yield last
    finally:
        proc.kill()
        proc.wait()


# --- overlays -------------------------------------------------------------------------------------


def ov_headline(img, o, t, size, assets):
    w, h = size
    alpha = window(t, o.get("from", 0), o.get("to", 99), 0.16, 0.16)
    if alpha <= 0:
        return
    unit = min(w, h)
    f = font(assets, int(unit * float(o.get("size", 0.075))))
    pos = {"upper": 0.27, "center": 0.46, "lower": 0.68}.get(o.get("pos", "center"), float(o.get("pos", 0.46)) if str(o.get("pos", "")).replace(".", "").isdigit() else 0.46)
    scale = lerp(0.9, 1.0, ease_out_back((t - o.get("from", 0)) / 0.28))
    lines = wrap(o["text"], f, int(w * 0.84))
    gap = f.size * 1.18
    top = h * pos - gap * (len(lines) - 1) / 2
    accent = {a.lower().strip(".,!?") for a in o.get("accent", [])}  # "Free." and "Free" both pick out the word Free
    if o.get("kicker"):
        put_text(img, o["kicker"].upper(), font(assets, int(unit * 0.026), 700, False), (w / 2, top - gap * 0.95), CYAN, alpha)
    for i, line in enumerate(lines):
        words, widths = line.split(), None
        if accent and any(word.strip(".,!?").lower() in accent for word in words):
            # draw word by word so accented words are cyan
            total = f.getlength(line)
            x = w / 2 - total * scale / 2
            for word in words:
                piece = word + " "
                cx = x + f.getlength(piece) * scale / 2
                color = CYAN if word.strip(".,!?").lower() in accent else WHITE
                put_text(img, word, f, (cx - f.getlength(" ") * scale / 2, top + i * gap), color, alpha, scale)
                x += f.getlength(piece) * scale
        else:
            put_text(img, line, f, (w / 2, top + i * gap), WHITE, alpha, scale)


def ov_bubble(img, o, t, size, assets):
    w, h = size
    a = window(t, o.get("from", 0), o.get("to", 99), 0.2, 0.15)
    if a <= 0:
        return
    unit = min(w, h)
    f = font(assets, int(unit * 0.052), 700, False)
    bw = int(w * (0.74 if w > h else 0.86))
    lines = wrap(o["text"], f, bw - int(unit * 0.2))
    bh = int(len(lines) * f.size * 1.3 + unit * 0.12)
    slide = (1 - ease_out_back((t - o.get("from", 0)) / 0.35)) * h * 0.06
    x0, y0 = (w - bw) / 2, h * 0.42 + slide
    rounded(img, (x0 + 4, y0 + 8, x0 + bw + 4, y0 + bh + 8), 26, fill=(0, 0, 0, int(90 * a)))
    rounded(img, (x0, y0, x0 + bw, y0 + bh), 26, fill=hex_alpha(WHITE, 0.96 * a))
    r = int(unit * 0.045)
    cx, cy = x0 + r * 1.6, y0 + bh / 2
    ImageDraw.Draw(img, "RGBA").ellipse((cx - r, cy - r, cx + r, cy + r), fill=hex_alpha(NAVY, a))
    put_text(img, o.get("who", "CFO"), font(assets, int(r * 0.62), 800, False), (cx, cy), WHITE, a, shadow=False)
    for i, line in enumerate(lines):
        put_text(img, line, f, (x0 + r * 3.1 + f.getlength(line) / 2, y0 + unit * 0.06 + f.size * (0.65 + 1.3 * i)), NAVY, a, shadow=False)


def ov_tabs(img, o, t, size, assets):
    w, h = size
    start, end = o.get("from", 0), o.get("to", 99)
    a = window(t, start, end, 0.12, 0.1)
    if a <= 0:
        return
    unit = min(w, h)
    items = o["items"]
    # a grey spreadsheet grid with an error cell, and the tabs piling up under it
    gx0, gy0, gx1, gy1 = w * 0.09, h * 0.05, w * 0.91, h * 0.5
    rounded(img, (gx0, gy0, gx1, gy1), 14, fill=(236, 240, 245, int(235 * a)))
    draw = ImageDraw.Draw(img, "RGBA")
    cols, rows = 7, 8
    for c in range(1, cols):
        x = lerp(gx0, gx1, c / cols)
        draw.line([(x, gy0), (x, gy1)], fill=(160, 172, 186, int(170 * a)), width=2)
    for r in range(1, rows):
        y = lerp(gy0, gy1, r / rows)
        draw.line([(gx0, y), (gx1, y)], fill=(160, 172, 186, int(170 * a)), width=2)
    if o.get("error"):
        blink = 1.0 if int((t - start) * 5) % 2 == 0 else 0.55
        ex0, ey0 = lerp(gx0, gx1, 3 / cols), lerp(gy0, gy1, 3 / rows)
        ex1, ey1 = lerp(gx0, gx1, 4 / cols), lerp(gy0, gy1, 4 / rows)
        draw.rectangle((ex0, ey0, ex1, ey1), fill=(*DANGER, int(255 * a * blink)))
        put_text(img, o["error"], font(assets, int(unit * 0.034), 800, False), ((ex0 + ex1) / 2, (ey0 + ey1) / 2), WHITE, a * blink, shadow=False)
    f = font(assets, int(unit * 0.03), 700, False)
    gap = (end - start) * 0.62 / max(len(items), 1)
    x, y, row_h = gx0, gy1 + unit * 0.012, unit * 0.062
    for i, label in enumerate(items):
        born = start + 0.1 + i * gap
        if t < born:
            continue
        tw = f.getlength(label) + unit * 0.06
        if x + tw > gx1:
            x, y = gx0, y + row_h * 1.12
        pop = ease_out_back((t - born) / 0.18)
        top = y + (1 - pop) * row_h * 0.6
        rounded(img, (x, top, x + tw, top + row_h), 10, fill=(*((255, 255, 255) if i % 2 == 0 else (214, 222, 232)), int(250 * a)), outline=(120, 135, 152, int(200 * a)))
        put_text(img, label, f, (x + tw / 2, top + row_h / 2), (40, 54, 72), a, shadow=False)
        x += tw + unit * 0.014


def ov_chips(img, o, t, size, assets):
    w, h = size
    start, end = o.get("from", 0), o.get("to", 99)
    a = window(t, start, end, 0.1, 0.2)
    if a <= 0:
        return
    unit = min(w, h)
    f = font(assets, int(unit * 0.04), 800, True)
    items = o["items"]
    widths = [f.getlength(s) + unit * 0.09 for s in items]
    total = sum(widths) + unit * 0.025 * (len(items) - 1)
    x, y = (w - total) / 2, h * float(o.get("y", 0.86))
    gap = float(o.get("step", 0.25))
    for i, label in enumerate(items):
        born = start + i * gap
        if t < born:
            x += widths[i] + unit * 0.025
            continue
        pop = ease_out_back((t - born) / 0.22)
        ch = unit * 0.075
        top = y + (1 - pop) * ch * 0.5
        rounded(img, (x, top, x + widths[i], top + ch), int(ch / 2), fill=(*DEEP, int(240 * a)), outline=(*CYAN, int(255 * a)), width=3)
        put_text(img, label, f, (x + widths[i] / 2, top + ch / 2), WHITE, a, shadow=False)
        x += widths[i] + unit * 0.025


def ov_timeline(img, o, t, size, assets):
    w, h = size
    start, end = o.get("from", 0), o.get("to", 99)
    a = window(t, start, end, 0.2, 0.2)
    if a <= 0:
        return
    unit = min(w, h)
    x0, x1, y = w * 0.08, w * 0.92, h * float(o.get("y", 0.62))
    bar = unit * 0.05
    reveal = ease_out((t - start) / float(o.get("draw", 1.2)))
    draw = ImageDraw.Draw(img, "RGBA")
    f = font(assets, int(unit * 0.03), 800, False)
    for seg in o["segments"]:
        sx0, sx1 = lerp(x0, x1, seg["start"]), lerp(x0, x1, seg["end"])
        sx1 = lerp(sx0, sx1, clamp((reveal * 1.0 - seg["start"] * 0.6) / max(seg["end"] - seg["start"], 0.05)))
        if sx1 <= sx0:
            continue
        hot = seg.get("hot")
        fill = (*CYAN, int(255 * a)) if hot else (60, 84, 112, int(230 * a))
        draw.rectangle((sx0, y, sx1, y + bar), fill=fill)
        if reveal > 0.55:
            label, size = seg["label"].upper(), int(unit * 0.03)
            fit = font(assets, size, 800, False)
            while fit.getlength(label) > (sx1 - sx0) * 0.92 and size > int(unit * 0.016):  # shrink to the segment, never overlap the next
                size -= 1
                fit = font(assets, size, 800, False)
            if fit.getlength(label) <= (sx1 - sx0) * 0.92:
                put_text(img, label, fit, ((sx0 + sx1) / 2, y + bar / 2), NAVY if hot else (200, 214, 230), a, shadow=False)
    mk = o.get("marker")
    if mk and t >= start + float(mk.get("at_time", 0.6)):
        mx = lerp(x0, x1, mk["at"])
        pop = ease_out_back((t - start - float(mk.get("at_time", 0.6))) / 0.3)
        draw.line([(mx, y - unit * 0.07 * pop), (mx, y + bar + unit * 0.025)], fill=(*WHITE, int(255 * a)), width=4)
        put_text(img, mk["label"].upper(), font(assets, int(unit * 0.028), 800, False), (mx, y - unit * 0.1 * pop), WHITE, a)
    if o.get("caption"):
        cf = font(assets, int(unit * 0.034), 700, False)
        for i, row in enumerate(wrap(o["caption"], cf, int(w * 0.86))):
            put_text(img, row, cf, (w / 2, y + bar + unit * 0.09 + i * cf.size * 1.25), (216, 227, 245), a)


def ov_logo(img, o, t, size, assets, root):
    w, h = size
    a = window(t, o.get("from", 0), o.get("to", 99), 0.3, 0.2) * float(o.get("opacity", 0.95))
    if a <= 0:
        return
    logo = Image.open(root / o["file"]).convert("RGBA")
    width = int(w * float(o.get("width", 0.2)))
    logo = logo.resize((width, int(logo.height * width / logo.width)), Image.LANCZOS)
    alpha = logo.getchannel("A").point(lambda v: int(v * a))
    pos = o.get("pos", "top-left")
    x = int(w * 0.04) if "left" in pos else (w - width) // 2 if "center" in pos else int(w * 0.96) - width
    y = int(h * 0.05) if "top" in pos else int(h * float(o.get("y", 0.4)))
    if o.get("plate"):  # a dark pill behind the logo so it reads on a bright sky
        pad = int(width * 0.09)
        rounded(img, (x - pad, y - pad * 0.7, x + width + pad, y + logo.height + pad * 0.7), int(logo.height * 0.7), fill=hex_alpha(NAVY, 0.72 * a))
    img.paste(logo.convert("RGB"), (x, y), alpha)


def ov_endcard(img, o, t, size, assets, root):
    w, h = size
    start = o.get("from", 0)
    a = clamp((t - start) / 0.35)
    if a <= 0:
        return
    unit = min(w, h)
    dim = Image.new("RGB", img.size, NAVY)
    img.paste(Image.blend(img, dim, 0.78 * a))
    pop = ease_out_back((t - start) / 0.5)
    if o.get("logo"):
        ov_logo(img, {"file": o["logo"], "width": 0.5 if w > h else 0.68, "pos": "center", "y": 0.2, "from": start, "to": 999, "opacity": 1.0}, t + 0.3, size, assets, root)
    if o.get("tagline"):
        f = font(assets, int(unit * 0.042), 800, True)
        for i, line in enumerate(wrap(o["tagline"], f, int(w * 0.86))):
            put_text(img, line, f, (w / 2, h * 0.56 + i * f.size * 1.2), WHITE, a)
    if o.get("button"):
        f = font(assets, int(unit * 0.041), 800, False)
        bw, bh = f.getlength(o["button"]) + unit * 0.14, unit * 0.1
        bx, by = (w - bw) / 2, h * 0.70 + (1 - pop) * 30
        rounded(img, (bx, by, bx + bw, by + bh), int(bh / 2), fill=(*DEEP, int(255 * a)), outline=(*CYAN, int(255 * a)), width=3)
        put_text(img, o["button"], f, (w / 2, by + bh / 2), WHITE, a, shadow=False)
    if o.get("url"):
        put_text(img, o["url"], font(assets, int(unit * 0.036), 700, False), (w / 2, h * 0.70 + unit * 0.17), CYAN, a)
    if o.get("small"):
        put_text(img, o["small"], font(assets, int(unit * 0.024), 600, False), (w / 2, h * 0.93), (170, 190, 210), a, shadow=False)


def ov_subtitle(img, o, t, size, assets):
    """A caption for words spoken on camera, placed and weighted like the burned-in captions."""
    w, h = size
    a = window(t, o.get("from", 0), o.get("to", 99), 0.08, 0.08)
    if a <= 0:
        return
    f = font(assets, int(h * 0.04), 700, False)
    rows = wrap(o["text"], f, int(w * 0.86))
    base = h * 0.78 - (len(rows) - 1) * f.size * 1.25
    for i, row in enumerate(rows):
        put_text(img, row, f, (w / 2, base + i * f.size * 1.25), WHITE, a)


OVERLAYS = {"subtitle": ov_subtitle, "headline": ov_headline, "bubble": ov_bubble, "tabs": ov_tabs, "chips": ov_chips, "timeline": ov_timeline}
NEEDS_ROOT = {"logo": ov_logo, "endcard": ov_endcard}


# --- product screens --------------------------------------------------------------------------------


class Screen:
    """A product screenshot shown as a card that slowly pushes in, with optional glowing highlights."""

    def __init__(self, spec: dict, size, root: Path):
        self.spec, self.size = spec, size
        src = Image.open(root / spec["file"]).convert("RGB")
        x0, y0, x1, y1 = spec.get("crop") or (0, 0, src.width, src.height)
        self.src = src.crop((x0, y0, x1, y1))
        self.origin = (x0, y0)
        w, h = size
        # `scale` shrinks a card that would otherwise reach down into the caption zone (a tall crop of a wide screen).
        self.card_w = int(min(int(w * (0.84 if w > h else 0.9)), int(h * 0.76 * self.src.width / self.src.height)) * float(spec.get("scale", 1.0)))
        self.card_h = int(self.card_w * self.src.height / self.src.width)
        self.cx, self.cy = w / 2, h * float(spec.get("y", 0.47))

    def draw(self, img: Image.Image, t: float, seconds: float, highlights: list[dict], assets: Path) -> None:
        z0, z1 = self.spec.get("zoom", [1.0, 1.12])
        zoom = lerp(z0, z1, ease_out(t / max(seconds, 0.1)) * 0.6 + (t / max(seconds, 0.1)) * 0.4)
        fx, fy = self.spec.get("focus", [0.5, 0.5])
        vw, vh = self.src.width / zoom, self.src.height / zoom
        vx = clamp(fx * self.src.width - vw / 2, 0, self.src.width - vw)
        vy = clamp(fy * self.src.height - vh / 2, 0, self.src.height - vh)
        view = self.src.resize((self.card_w, self.card_h), Image.LANCZOS, box=(vx, vy, vx + vw, vy + vh))
        x, y = int(self.cx - self.card_w / 2), int(self.cy - self.card_h / 2)
        shadow = Image.new("L", (self.card_w + 80, self.card_h + 80), 0)
        ImageDraw.Draw(shadow).rounded_rectangle((40, 50, self.card_w + 40, self.card_h + 50), 26, fill=150)
        shadow = shadow.filter(ImageFilter.GaussianBlur(22))
        img.paste(Image.new("RGB", shadow.size, (0, 0, 0)), (x - 40, y - 40), shadow)
        round_mask = Image.new("L", view.size, 0)
        ImageDraw.Draw(round_mask).rounded_rectangle((0, 0, view.width - 1, view.height - 1), 22, fill=255)
        img.paste(view, (x, y), round_mask)
        sx, sy = self.card_w / vw, self.card_h / vh
        for hl in highlights:
            a = window(t, hl.get("from", 0), hl.get("to", 99), 0.15, 0.15)
            if a <= 0:
                continue
            bx0, by0, bx1, by1 = hl["box"]
            px0, py0 = x + (bx0 - self.origin[0] - vx) * sx, y + (by0 - self.origin[1] - vy) * sy
            px1, py1 = x + (bx1 - self.origin[0] - vx) * sx, y + (by1 - self.origin[1] - vy) * sy
            glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(glow).rounded_rectangle((px0, py0, px1, py1), 14, outline=(*CYAN, int(255 * a)), width=7)
            glow = glow.filter(ImageFilter.GaussianBlur(9))
            img.paste(glow, (0, 0), glow)
            rounded(img, (px0, py0, px1, py1), 14, outline=(*CYAN, int(255 * a)), width=4)
            if hl.get("label"):
                put_text(img, hl["label"], font(assets, int(min(img.size) * 0.034), 800, False), ((px0 + px1) / 2, py0 - min(img.size) * 0.035), CYAN, a)


# --- one scene ----------------------------------------------------------------------------------------


def render_scene(scene: dict, out: Path, size, root: Path, assets: Path, clip: Path | None, background: Path | None,
                 start_offset: float = 0.0) -> None:
    """Render a scene to an MP4 (silent) of exactly scene['seconds'] at FPS."""
    w, h = size
    count = max(1, round(float(scene["seconds"]) * FPS))
    kind = scene["kind"]
    push = scene.get("push", [1.0, 1.06]) if kind == "veo" else None
    stream = Stream(size)
    mask = vignette(size, 0.45, float(scene.get("caption_shade", 0.5)))
    base_src = None
    if kind == "veo":
        if clip is None:
            raise ComposeError(f"scene {scene['id']} has no clip yet")
        base_src = decode(clip, size, count, start_offset, float(scene.get("speed", 1.0)))
    elif background is not None:
        base_src = decode(background, size, count, 0.0)
    screen = Screen(scene["image"], size, root) if kind == "image" else None
    overlays = scene.get("overlays", [])
    highlights = [o for o in overlays if o["type"] == "highlight"]
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    try:
        for i in range(count):
            t = i / FPS
            if base_src is not None:
                frame = next(base_src).copy()
                # Footage behind a screen is a mood, not the subject, so it is dimmed; a title scene can ask for the same (`dim`)
                # when bright streaks would run through its headline.
                dim = float(scene.get("dim", 0.45 if kind == "image" else 0.0))
                if dim > 0:
                    frame = Image.blend(frame, Image.new("RGB", size, NAVY), dim)
                if push:
                    z = lerp(push[0], push[1], t / max(float(scene["seconds"]), 0.1))
                    if z > 1.001:
                        cw, ch = w / z, h / z
                        frame = frame.resize((w, h), Image.BILINEAR, box=((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2))
                frame = graded(frame, mask)
            else:
                frame = stream.frame(t + float(scene.get("stream_phase", 0)))
            if screen:
                screen.draw(frame, t, float(scene["seconds"]), highlights, assets)
            for o in overlays:
                kind_o = o["type"]
                if kind_o in OVERLAYS:
                    OVERLAYS[kind_o](frame, o, t, size, assets)
                elif kind_o in NEEDS_ROOT:
                    NEEDS_ROOT[kind_o](frame, o, t, size, assets, root)
            enc.stdin.write(frame.tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode:
        raise ComposeError(f"ffmpeg failed encoding scene {scene['id']}")
