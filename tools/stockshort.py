#!/usr/bin/env python3
"""stockshort — turn a researched stock-move script (JSON) into a 9:16 explainer short.

    python stockshort.py examples/ORCL-2026-11-02.json            # -> out/ORCL-2026-11-02.mp4 + .txt
    python stockshort.py spec.json --voice am_adam --music bed.mp3

VOICE (most natural first):
  1. ElevenLabs   if ELEVENLABS_API_KEY is set (paid)
  2. Kokoro       free neural voice that runs locally (pip install kokoro-onnx soundfile; the
                  ~350 MB model downloads once from GitHub into ~/.cache/stockshort)
  3. macOS `say` / espeak-ng   robotic fallbacks
  Kokoro voices: am_michael (default), am_adam, am_onyx, am_eric, af_heart, af_bella, bm_george ...

SYNC: every on-screen element can carry "at": a word or phrase from that beat's narration.
The element animates in at the moment that phrase is spoken. Without "at", elements are
spread evenly across the beat.

Spec: ticker, company, move_pct, title, description, hashtags, keywords, sources, and
"beats": [{"say": narration, "visual": {"type": ..., ...}}]
Visual types: hook, headline, stat, bars, compare, line, bullets, scorecard, text, outro,
question (explainer/recap opener), rank (countdown card) (see examples/). Specs without ticker/move_pct
(explainers, weekly recap) set "slug", "chip" and "topic" instead.
Extras: "loop": true makes the last line flow into the first; "tape": [...] fills the scrolling ticker
tape; sound effects and a generated music bed are synced to the same cues (--music to use your own
licensed track, --no-music for voice + effects only).
Every number on screen must come from the sources listed in the spec.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
SR = 24000
C = {
    "bg1": (8, 12, 24), "bg2": (16, 24, 46), "text": (242, 245, 250), "muted": (148, 160, 184),
    "up": (30, 215, 130), "down": (255, 72, 88), "accent": (255, 200, 0),
}
BEAT_GAP = 0.08      # pause between beats (s)
SENT_GAP = 0.06      # pause between sentences inside a beat (s)
# Platform safe zone: TikTok/Shorts cover the top ~210px (tabs), the right ~180px from y≈700 (buttons)
# and the bottom ~400px (caption, username, nav). Scenes are drawn on a 1080x1920 canvas, then scaled
# into the safe box: canvas point (540, 800) lands at (SAFE_CX, SAFE_CY), scaled by SAFE_S.
# Centered for TikTok, Reels and Shorts (2026-11-07): content and captions center on x~520, clear of the right button column (x>940).
SAFE_S, SAFE_CX, SAFE_CY = 0.90, 520, 850
CACHE = Path.home() / ".cache" / "stockshort"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

# ================================================================== fonts
_FC: dict = {}
FONT_CANDIDATES = {
    "bold": ["Poppins-ExtraBold.ttf", "Poppins-Bold.ttf", "Inter-Bold.otf", "Montserrat-ExtraBold.ttf",
             "Arial Bold.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
    "medium": ["Poppins-Medium.ttf", "Inter-Medium.otf", "Montserrat-Medium.ttf", "Arial.ttf", "arial.ttf",
               "DejaVuSans.ttf"],
}
FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts"),
             str(Path.home() / "Library/Fonts"), "/Library/Fonts", "/System/Library/Fonts/Supplemental",
             "C:/Windows/Fonts"]


def _find_font(weight: str) -> str | None:
    for name in FONT_CANDIDATES[weight]:
        for d in FONT_DIRS:
            if Path(d).exists():
                for p in Path(d).rglob(name):
                    return str(p)
    if shutil.which("fc-match"):
        q = "Poppins:bold" if weight == "bold" else "Poppins:medium"
        out = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True).stdout
        if out:
            return out
    return None


def font(weight: str, size: int):
    key = (weight, size)
    if key not in _FC:
        if weight + "_path" not in _FC:
            _FC[weight + "_path"] = _find_font(weight)
        path = _FC[weight + "_path"]
        _FC[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    return _FC[key]


# ================================================================== voice
_KOKORO = None


def _kokoro():
    global _KOKORO
    if _KOKORO is None:
        from kokoro_onnx import Kokoro  # noqa
        CACHE.mkdir(parents=True, exist_ok=True)
        for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            p = CACHE / f
            if not p.exists() or p.stat().st_size < 1_000_000:
                print(f"  downloading voice model {f} (one time) ...", flush=True)
                urllib.request.urlretrieve(KOKORO_URL + f, p)
        _KOKORO = Kokoro(str(CACHE / "kokoro-v1.0.onnx"), str(CACHE / "voices-v1.0.bin"))
    return _KOKORO


def _have_kokoro() -> bool:
    try:
        import kokoro_onnx  # noqa
        import soundfile  # noqa
        return True
    except ImportError:
        return False


def _write_wav(path: Path, samples, sr: int):
    import numpy as np
    a = np.clip(np.asarray(samples, dtype="float32"), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((a * 32767).astype("<i2").tobytes())


def _read_wav(path: Path):
    import numpy as np
    with wave.open(str(path)) as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2" if sw == 2 else "u1").astype("float32")
    a = a / 32768 if sw == 2 else (a - 128) / 128
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def synth(text: str, out_wav: Path, voice: str, speed: float) -> str:
    """Synthesize one sentence to out_wav. Returns the engine used."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        import requests
        vid = os.environ.get("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
        r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format=pcm_24000",
                          headers={"xi-api-key": key, "Content-Type": "application/json"},
                          json={"text": text, "model_id": "eleven_multilingual_v2",
                                "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.3}},
                          timeout=120)
        r.raise_for_status()
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(r.content)
        return "elevenlabs"
    if _have_kokoro():
        audio, sr = _kokoro().create(text, voice=voice, speed=speed, lang="en-us")
        _write_wav(out_wav, audio, sr)
        return "kokoro"
    if shutil.which("say"):
        aiff = out_wav.with_suffix(".aiff")
        subprocess.run(["say", "-r", "185", "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(aiff), "-ac", "1", str(out_wav)], check=True)
        return "say"
    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if not engine:
        raise SystemExit("No voice available: pip install kokoro-onnx soundfile (free), or set ELEVENLABS_API_KEY.")
    subprocess.run([engine, "-v", "en-us+m3", "-s", "172", "-w", str(out_wav), text], check=True)
    return "espeak"


def speech_bounds(a, sr) -> tuple[float, float]:
    """Start/end of actual speech (trims leading/trailing silence)."""
    import numpy as np
    hop = int(sr * 0.01)
    if len(a) < hop * 3:
        return 0.0, len(a) / sr
    frames = a[: len(a) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    thr = max(rms.max() * 0.06, 1e-4)
    idx = np.where(rms > thr)[0]
    if not len(idx):
        return 0.0, len(a) / sr
    return idx[0] * hop / sr, (idx[-1] + 1) * hop / sr


def _syllables(word: str) -> float:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1.0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(1, n) + (0.6 if re.search(r"[,;:—–-]$", word) else 0)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# ================================================================== drawing helpers
def clamp01(x):
    return max(0.0, min(1.0, x))


def ease(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def pop(x):
    x = clamp01(x)
    return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def text_w(d, s, f):
    return int(d.textlength(s, font=f))


def ctext(d, y, s, f, fill, x=W // 2, anchor="mm"):
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def wrap(d, s, f, max_w):
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_w(d, trial, f) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_font(d, s, weight, start, max_w, min_size=40):
    size = start
    while size > min_size and text_w(d, s, font(weight, size)) > max_w:
        size -= 4
    return font(weight, size)


def card(img, box, radius=36, fill=(255, 255, 255, 16)):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(box, radius=radius, fill=fill)
    img.alpha_composite(layer)


def background():
    img = Image.new("RGBA", (W, H), C["bg1"] + (255,))
    d = ImageDraw.Draw(img)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=mix(C["bg1"], C["bg2"], (y / H) ** 1.4) + (255,))
    g = ImageDraw.Draw(img)
    for x in range(0, W, 90):
        g.line([(x, 0), (x, H)], fill=(20, 28, 48, 255))
    for y in range(0, H, 90):
        g.line([(0, y), (W, y)], fill=(20, 28, 48, 255))
    return img


def colr(name, default="accent"):
    return C.get(name or default, C[default])


def move_color(p):
    return C["up"] if p >= 0 else C["down"]


def fmt_pct(p):
    return f"{'+' if p >= 0 else '−'}{abs(p):.1f}%"


def A(x):
    """alpha from 0..1"""
    return int(255 * clamp01(x))


# ================================================================== cue timing
def _norm(s):
    return re.sub(r"[^\w$%.']", "", s.lower()).strip(".'")


class Cues:
    """Maps 'at' phrases to times (seconds from beat start) using the beat's word timings."""

    def __init__(self, words: list[dict], dur: float):
        self.words, self.dur = words, dur
        self.toks = [_norm(w["word"]) for w in words]

    def at(self, phrase, default: float) -> float:
        if phrase is None:
            return default
        if isinstance(phrase, (int, float)):
            return float(phrase)
        target = [_norm(x) for x in str(phrase).split() if _norm(x)]
        if not target:
            return default
        n = len(target)
        for i in range(len(self.toks) - n + 1):
            if all(self.toks[i + k].startswith(target[k]) for k in range(n)):
                return max(0.0, self.words[i]["start"] - 0.08)  # land a hair before the word
        print(f"  ! cue '{phrase}' not found in narration; using default", file=sys.stderr)
        return default

    def spread(self, k: int, n: int, lo: float = 0.1, hi: float = 0.75) -> float:
        return self.dur * (lo + (hi - lo) * (k / max(1, n - 1) if n > 1 else 0))


# ================================================================== scenes
# each scene: f(img, v, t, spec, cues) ; t = seconds since beat start

def _candle(d, cx, top, bot, btop, bbot, w, col, a=255, wick=16):
    d.rounded_rectangle((cx - wick // 2, top, cx + wick // 2, bot), radius=wick // 2, fill=col + (a,))
    d.rounded_rectangle((cx - w // 2, btop, cx + w // 2, bbot), radius=max(4, w // 8), fill=col + (a,))


def scene_hook(img, v, t, spec, cues):
    """Frame 1 already shows the move, the candle and the question. Nothing to wait for."""
    d = ImageDraw.Draw(img)
    pct = float(v.get("move_pct", spec["move_pct"]))
    col = move_color(pct)
    ctext(d, 400, v.get("label", "TODAY").upper(), font("bold", 40), C["muted"])
    k = ease(t / 0.15)  # candle drops the last few pixels into place: frame 1 already reads clearly
    off = int((1 - k) * (-40 if pct < 0 else 40))
    _candle(d, W // 2, 700 + off, 1010 + off, 740 + off, 960 + off, 160, col, a=255, wick=24)
    f = font("bold", int(250 * (0.92 + 0.08 * pop(t / 0.3))))
    ctext(d, 540, fmt_pct(pct), f, col)  # real number from frame 1 (a count-up shows wrong values)
    q = v.get("question") or v.get("note")
    if q:
        f2 = font("bold", 76)
        parts = [x.strip() for x in re.split(r"(?<=[.?!])\s+", q.upper()) if x.strip()]
        lines = [ln for part in parts for ln in wrap(d, part, f2, 980)]
        # keep the whole question above the caption band (captions start near y 1290)
        for i, line in enumerate(lines):
            ctext(d, 1070 + i * 88, line, f2, C["text"] if i < len(lines) - 1 else C["accent"])
    if v.get("to") is not None:
        tp = cues.at(v.get("price_at"), 0.6)
        if t > tp:
            fr = f"${float(v['from']):,.2f} → " if v.get("from") is not None else "now "
            ctext(d, 1060, f"{fr}${float(v['to']):,.2f}", font("medium", 50), C["muted"] + (A((t - tp) / 0.25),))


def scene_headline(img, v, t, spec, cues):
    """A recreated news headline (text only, never a screenshot) with a marker sweep on the key phrase."""
    t0 = cues.at(v.get("at"), 0.0)
    if t < t0:
        return
    p = ease((t - t0) / 0.35)
    d = ImageDraw.Draw(img)
    ctext(d, 420, v.get("kicker", "THE NEWS").upper(), font("bold", 40), C["muted"] + (A(p),))
    x0, y0 = 80, 500 + int((1 - p) * 80)
    fh = font("bold", 68)
    lines = wrap(d, v["headline"], fh, 840)
    hgt = 210 + 84 * len(lines)
    card(img, (x0, y0, W - 80, y0 + hgt), radius=30, fill=(245, 247, 251, A(p)))
    d = ImageDraw.Draw(img)
    src = v.get("source", "")
    d.rounded_rectangle((x0 + 50, y0 + 46, x0 + 50 + text_w(d, src.upper(), font("bold", 32)) + 40, y0 + 100),
                        radius=12, fill=(16, 24, 46, A(p)))
    d.text((x0 + 70, y0 + 73), src.upper(), font=font("bold", 32), fill=(255, 255, 255, A(p)), anchor="lm")
    if v.get("date"):
        d.text((W - 130, y0 + 73), v["date"], font=font("medium", 32), fill=(110, 120, 140, A(p)), anchor="rm")
    hl = (v.get("highlight") or "").lower()
    th = cues.at(v.get("highlight_at"), t0 + 0.6)
    yy = y0 + 150
    for line in lines:
        if hl and hl in line.lower() and t > th:
            i = line.lower().index(hl)
            pre_w = text_w(d, line[:i], fh)
            hw = text_w(d, line[i:i + len(hl)], fh)
            sweep = ease((t - th) / 0.3)
            d.rectangle((x0 + 50 + pre_w - 6, yy + 6, x0 + 50 + pre_w + int((hw + 12) * sweep), yy + 80),
                        fill=C["accent"] + (A(p),))
        d.text((x0 + 50, yy), line, font=fh, fill=(14, 18, 30, A(p)))
        yy += 84
    if v.get("sub"):
        ts = cues.at(v.get("sub_at"), t0 + 1.0)
        for i, line in enumerate(wrap(d, v["sub"], font("medium", 48), 900)):
            ctext(d, y0 + hgt + 80 + i * 60, line, font("medium", 48), C["text"] + (A((t - ts) / 0.3),))


def scene_stat(img, v, t, spec, cues):
    d = ImageDraw.Draw(img)
    color = colr(v.get("color"), "text")
    t0 = cues.at(v.get("at"), 0.05)
    s = t - t0
    if s > 0:
        f = fit_font(d, str(v["value"]), "bold", int(290 * max(0.3, pop(s / 0.4))), 980)
        ctext(d, 650, str(v["value"]), f, color)
    if v.get("label"):
        tl = cues.at(v.get("label_at"), t0 + 0.35)
        for i, line in enumerate(wrap(d, v["label"], font("bold", 70), 940)):
            ctext(d, 880 + i * 84 + int(30 * (1 - ease((t - tl) / 0.3))), line, font("bold", 70),
                  C["text"] + (A((t - tl) / 0.3),))
    if v.get("sub"):
        ts = cues.at(v.get("sub_at"), t0 + 0.7)
        for i, line in enumerate(wrap(d, v["sub"], font("medium", 48), 900)):
            ctext(d, 1080 + i * 60, line, font("medium", 48), C["muted"] + (A((t - ts) / 0.3),))


def scene_bars(img, v, t, spec, cues):
    d = ImageDraw.Draw(img)
    items = v["items"][:4]
    if v.get("title"):
        ctext(d, 420, v["title"], fit_font(d, v["title"], "bold", 66, 980), C["text"])
    mx = max(float(i["value"]) for i in items) or 1
    for k, it in enumerate(items):
        tk = cues.at(it.get("at"), cues.spread(k, len(items), 0.05, 0.55))
        if t < tk:
            continue
        p = ease((t - tk) / 0.5)
        y = 560 + k * 190
        d.text((90, y), it["label"], font=font("medium", 52), fill=C["muted"] + (A((t - tk) / 0.2),))
        bw = int(740 * float(it["value"]) / mx * p)
        d.rounded_rectangle((90, y + 70, 90 + max(bw, 8), y + 150), radius=20, fill=colr(it.get("color")))
        if p > 0.2:
            d.text((90 + bw + 24, y + 110), it.get("display", str(it["value"])), font=font("bold", 62),
                   fill=C["text"], anchor="lm")


def scene_compare(img, v, t, spec, cues):
    d = ImageDraw.Draw(img)
    if v.get("title"):
        for i, line in enumerate(wrap(d, v["title"], font("bold", 64), 980)):
            ctext(d, 420 + i * 76, line, font("bold", 64), C["text"])
    shown = 0
    for k, side in enumerate(("left", "right")):
        it = v[side]
        tk = cues.at(it.get("at"), cues.spread(k, 2, 0.05, 0.45))
        if t < tk:
            continue
        shown += 1
        p = ease((t - tk) / 0.35)
        x0, y0 = 70 + k * 500 + int((1 - p) * (-80 if k == 0 else 80)), 600
        card(img, (x0, y0, x0 + 440, y0 + 470), fill=(255, 255, 255, int(22 * p)))
        d = ImageDraw.Draw(img)
        ctext(d, y0 + 110, it["label"].upper(), font("bold", 40), C["muted"] + (A(p),), x=x0 + 220)
        ctext(d, y0 + 260, it["value"], fit_font(d, it["value"], "bold", int(120 * (0.8 + 0.2 * pop((t - tk) / 0.4))), 400),
              colr(it.get("color"), "text") + (A(p),), x=x0 + 220)
        for i, line in enumerate(wrap(d, it.get("sub", ""), font("medium", 38), 380)):
            ctext(d, y0 + 370 + i * 46, line, font("medium", 38), C["muted"] + (A(p),), x=x0 + 220)
    if shown == 2:
        ctext(ImageDraw.Draw(img), 835, "vs", font("bold", 56), C["accent"])


def scene_line(img, v, t, spec, cues):
    d = ImageDraw.Draw(img)
    pts = v["points"]
    vals = [float(p["value"]) for p in pts]
    if v.get("title"):
        ctext(d, 400, v["title"], fit_font(d, v["title"], "bold", 64, 980), C["text"])
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.18 or 1
    lo, hi = lo - pad, hi + pad
    x0, x1, y0, y1 = 190, 890, 520, 1120
    for gy in range(5):
        yy = y0 + (y1 - y0) * gy / 4
        d.line([(x0, yy), (x1, yy)], fill=(40, 52, 80, 255), width=2)
    xy = [(x0 + (x1 - x0) * i / max(1, len(pts) - 1), y1 - (y1 - y0) * (vals[i] - lo) / (hi - lo))
          for i in range(len(pts))]
    times = [cues.at(p.get("at"), cues.spread(i, len(pts), 0.05, 0.7)) for i, p in enumerate(pts)]
    drawn = [xy[0]] if t >= times[0] else []
    for i in range(1, len(xy)):
        if t >= times[i]:
            drawn.append(xy[i])
        elif t > times[i] - 0.5 and drawn:
            f = ease((t - (times[i] - 0.5)) / 0.5)
            drawn.append((xy[i - 1][0] + (xy[i][0] - xy[i - 1][0]) * f, xy[i - 1][1] + (xy[i][1] - xy[i - 1][1]) * f))
            break
        else:
            break
    col = move_color(vals[-1] - vals[0])
    if len(drawn) > 1:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(drawn + [(drawn[-1][0], y1), (drawn[0][0], y1)], fill=col + (40,))
        img.alpha_composite(layer)
        d = ImageDraw.Draw(img)
        d.line(drawn, fill=col, width=10, joint="curve")
    pre = v.get("prefix", "$")
    for i, (x, y) in enumerate(xy):
        if t >= times[i]:
            a = A((t - times[i]) / 0.25)
            r = 14 * (0.6 + 0.4 * pop((t - times[i]) / 0.3))
            d.ellipse((x - r, y - r, x + r, y + r), fill=C["text"] + (a,))
            lab = f"{pre}{vals[i]:,.0f}" if vals[i] >= 100 else f"{pre}{vals[i]:,.2f}"
            d.text((x, y - 52 if y > y0 + 80 else y + 60), lab, font=font("bold", 50), fill=C["text"] + (a,), anchor="mm")
            d.text((x, y1 + 50), pts[i]["label"], font=font("medium", 40), fill=C["muted"] + (a,), anchor="mm")


def scene_bullets(img, v, t, spec, cues):
    d = ImageDraw.Draw(img)
    if v.get("title"):
        ctext(d, 410, v["title"], fit_font(d, v["title"], "bold", 66, 980), C["text"])
    items = [i if isinstance(i, dict) else {"text": i} for i in v["items"][:4]]
    times = [cues.at(it.get("at"), cues.spread(k, len(items), 0.12, 0.7)) for k, it in enumerate(items)]
    active = max([k for k, tk in enumerate(times) if t >= tk], default=-1)
    y = 540
    for k, it in enumerate(items):
        lines = wrap(d, it["text"], font("bold", 56), 800)
        hgt = 40 + 68 * len(lines)
        if t >= times[k]:
            p = ease((t - times[k]) / 0.35)
            x = 90 - int((1 - p) * 60)
            on = k == active
            card(img, (x, y, x + 900, y + hgt), radius=28, fill=(255, 255, 255, int((30 if on else 14) * p)))
            d = ImageDraw.Draw(img)
            d.ellipse((x + 34, y + 38, x + 62, y + 66), fill=(C["accent"] if on else C["muted"]) + (A(p),))
            for i, line in enumerate(lines):
                d.text((x + 92, y + 22 + i * 68), line, font=font("bold", 56),
                       fill=(C["text"] if on else C["muted"]) + (A(p),))
        y += hgt + 34


VERDICT = {"good": "up", "bad": "down", "unknown": "accent", "beat": "up", "miss": "down", "inline": "accent"}


def _mark(d, cx, cy, r, kind, scale=1.0, alpha=255):
    col = C[VERDICT.get(kind, "accent")]
    r = int(r * scale)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col + (alpha,))
    w = max(6, int(r * 0.2))
    ink = (10, 14, 26, alpha)
    if kind in ("good", "beat"):
        d.line([(cx - r * 0.42, cy + r * 0.02), (cx - r * 0.1, cy + r * 0.34), (cx + r * 0.45, cy - r * 0.3)], fill=ink, width=w, joint="curve")
    elif kind in ("bad", "miss"):
        k = r * 0.36
        d.line([(cx - k, cy - k), (cx + k, cy + k)], fill=ink, width=w)
        d.line([(cx - k, cy + k), (cx + k, cy - k)], fill=ink, width=w)
    else:
        d.text((cx, cy + 2), "?" if kind == "unknown" else "=", font=font("bold", int(r * 1.3)), fill=ink, anchor="mm")


def scene_scorecard(img, v, t, spec, cues):
    """Signature format: each line stamps a verdict (good / bad / unknown) as it's spoken."""
    d = ImageDraw.Draw(img)
    title = v.get("title", "THE SCORECARD")
    ctext(d, 405, title.upper(), fit_font(d, title.upper(), "bold", 72, 900), C["text"])
    d.rounded_rectangle((W // 2 - 70, 455, W // 2 + 70, 465), radius=5, fill=C["accent"])
    items = v["items"][:4]
    for k, it in enumerate(items):
        tk = cues.at(it.get("at"), cues.spread(k, len(items), 0.1, 0.75))
        y = 520 + k * 175
        if t < tk:
            card(img, (70, y, W - 70, y + 150), radius=26, fill=(255, 255, 255, 8))
            continue
        p = ease((t - tk) / 0.25)
        card(img, (70, y, W - 70, y + 150), radius=26, fill=(255, 255, 255, int(10 + 12 * p)))
        d = ImageDraw.Draw(img)
        d.text((120, y + 50), it["label"], font=font("bold", 56), fill=C["text"] + (A(p),), anchor="lm")
        if it.get("detail"):
            d.text((120, y + 108), it["detail"], font=font("medium", 38), fill=C["muted"] + (A(p),), anchor="lm")
        ts = t - tk - 0.12  # the stamp lands just after the label
        if ts > 0:
            sc = 1 + 1.2 * (1 - ease(ts / 0.18))
            _mark(d, W - 160, y + 75, 50, it.get("verdict", "unknown"), scale=sc, alpha=A(ts / 0.1))


def scene_text(img, v, t, spec, cues):
    d = ImageDraw.Draw(img)
    t0 = cues.at(v.get("at"), 0.0)
    s = t - t0
    if s <= 0:
        return
    f = font("bold", int(96 * (0.85 + 0.15 * pop(s / 0.4))))
    lines = wrap(d, v["big"], f, 960)
    y = 700 - (len(lines) - 1) * 60
    th = cues.at(v.get("highlight_at") or v.get("highlight"), t0)
    for i, line in enumerate(lines):
        hl = v.get("highlight") and v["highlight"].lower() in line.lower()
        col = mix(C["text"], C["accent"], ease((t - th) / 0.3)) if hl else C["text"]
        ctext(d, y + i * 118, line, f, col + (A(s / 0.25),))
    if v.get("small"):
        ts = cues.at(v.get("small_at"), t0 + 0.5)
        for i, line in enumerate(wrap(d, v["small"], font("medium", 50), 900)):
            ctext(d, y + len(lines) * 118 + 40 + i * 62, line, font("medium", 50), C["muted"] + (A((t - ts) / 0.3),))


def scene_outro(img, v, t, spec, cues):
    scene_text(img, {k: v.get(k) for k in ("big", "highlight", "highlight_at", "at")}, t, spec, cues)
    tc = cues.at(v.get("cta_at"), cues.dur * 0.5)
    if t > tc:
        d = ImageDraw.Draw(img)
        p = pop((t - tc) / 0.4)
        cta = v.get("cta") or f"Follow {spec.get('brand', '')}".strip()
        f = font("bold", 54)
        w = text_w(d, cta, f) + 90
        x0 = (W - w) // 2
        d.rounded_rectangle((x0, 1060, x0 + w, 1060 + int(110 * max(0.2, p))), radius=55, fill=C["accent"])
        if p > 0.5:
            ctext(d, 1115, cta, f, (10, 10, 10))
        if v.get("cta_sub") and t > tc + 0.25:
            fs = font("medium", 44)
            ctext(d, 1215, v["cta_sub"], fs, C["muted"] + (A((t - tc - 0.25) / 0.3),))


def scene_question(img, v, t, spec, cues):
    """Opener for explainers and recaps: a big question/promise readable on frame 1."""
    d = ImageDraw.Draw(img)
    if v.get("kicker"):
        ctext(d, 420, v["kicker"].upper(), font("bold", 40), C["muted"])
    f = font("bold", int(108 * (0.94 + 0.06 * pop(t / 0.3))))
    lines = wrap(d, v["big"], font("bold", 108), 960)
    y = 700 - (len(lines) - 1) * 66
    hl = (v.get("highlight") or "").lower()
    for i, line in enumerate(lines):
        ctext(d, y + i * 128, line, f, C["accent"] if hl and hl in line.lower() else C["text"])
    if v.get("sub"):
        ts = cues.at(v.get("sub_at"), 0.8)
        for i, line in enumerate(wrap(d, v["sub"], font("medium", 50), 920)):
            ctext(d, y + len(lines) * 128 + 50 + i * 62, line, font("medium", 50), C["muted"] + (A((t - ts) / 0.3),))


def scene_rank(img, v, t, spec, cues):
    """Countdown card: rank, ticker, move and the one-line reason (weekly recap)."""
    d = ImageDraw.Draw(img)
    pct = float(v["move_pct"])
    col = move_color(pct)
    p = ease(t / 0.25)
    ctext(d, 430, f"#{v['rank']}", font("bold", int(150 * (0.8 + 0.2 * pop(t / 0.35)))), C["accent"])
    fk = font("bold", 76)
    chip_w = text_w(d, v["ticker"], fk) + 70
    x0 = (W - chip_w) // 2
    card(img, (x0, 540, x0 + chip_w, 650), radius=26, fill=col + (int(80 * p),))
    d = ImageDraw.Draw(img)
    ctext(d, 595, v["ticker"], fk, C["text"] + (A(p),))
    ctext(d, 700, v.get("company", ""), font("bold", 54), C["text"] + (A(p),))
    tm = cues.at(v.get("move_at"), 0.25)
    if t > tm:
        ctext(d, 840, fmt_pct(pct), font("bold", int(170 * (0.85 + 0.15 * pop((t - tm) / 0.35)))), col)
    if v.get("reason"):
        tr = cues.at(v.get("reason_at"), cues.dur * 0.4)
        for i, line in enumerate(wrap(d, v["reason"], font("medium", 52), 920)):
            ctext(d, 1000 + i * 64, line, font("medium", 52), C["text"] + (A((t - tr) / 0.3),))
    if v.get("when"):
        ctext(d, 1180, v["when"], font("medium", 38), C["muted"] + (A(p),))


def _arrow_down(d, x, y0, y1, prog, col, width=8):
    """Vertical arrow drawn from y0 toward y1 by prog (0..1)."""
    if prog <= 0:
        return
    ye = y0 + (y1 - y0) * prog
    d.line([(x, y0), (x, ye)], fill=col, width=width)
    if prog > 0.85:
        h = 22
        d.polygon([(x - h, ye - h), (x + h, ye - h), (x, ye + 4)], fill=col)


def scene_chain(img, v, t, spec, cues):
    """Cause -> effect domino. Each link drops in when spoken; the arrow draws just before it."""
    d = ImageDraw.Draw(img)
    if v.get("title"):
        ctext(d, 410, v["title"].upper(), fit_font(d, v["title"].upper(), "bold", 60, 960), C["muted"])
    items = v["items"][:4]
    n = len(items)
    hgt, gap = 132, 72 if n == 4 else 96
    y = 480 if n == 4 else 520
    times = [cues.at(it.get("at"), cues.spread(k, n, 0.05, 0.75)) for k, it in enumerate(items)]
    active = max([k for k, tk in enumerate(times) if t >= tk], default=-1)
    for k, it in enumerate(items):
        tk = times[k]
        if k > 0:
            _arrow_down(ImageDraw.Draw(img), W // 2, y - gap + 8, y - 10, ease((t - (tk - 0.3)) / 0.3), C["muted"] + (255,))
        if t >= tk:
            p = ease((t - tk) / 0.3)
            sc = 0.9 + 0.1 * pop((t - tk) / 0.35)
            last = k == n - 1
            fillc = colr(it.get("color")) if it.get("color") else None
            bw = int(880 * sc)
            x0 = (W - bw) // 2
            if fillc:
                card(img, (x0, y, x0 + bw, y + hgt), radius=32, fill=fillc + (int(235 * p),))
            else:
                card(img, (x0, y, x0 + bw, y + hgt), radius=32, fill=(255, 255, 255, int((34 if k == active else 16) * p)))
            d = ImageDraw.Draw(img)
            if k == active and not fillc:
                d.rounded_rectangle((x0, y, x0 + bw, y + hgt), radius=32, outline=C["accent"] + (A(p),), width=5)
            ink = (10, 14, 26) if fillc else C["text"]
            f = fit_font(d, it["text"], "bold", 64 if last else 58, bw - 80)
            ctext(d, y + hgt // 2, it["text"], f, ink + (A(p),))
        y += hgt + gap


def scene_beatmiss(img, v, t, spec, cues):
    """Expected vs actual. The 'actual' number and the verdict stamp land when spoken."""
    d = ImageDraw.Draw(img)
    title = v.get("title", "Expected vs. actual")
    ctext(d, 410, title.upper(), fit_font(d, title.upper(), "bold", 64, 960), C["text"])
    d.rounded_rectangle((W // 2 - 70, 455, W // 2 + 70, 465), radius=5, fill=C["accent"])
    items = v["items"][:3]
    for k, it in enumerate(items):
        tk = cues.at(it.get("at"), cues.spread(k, len(items), 0.05, 0.6))
        y = 510 + k * 245
        if t < tk:
            continue
        p = ease((t - tk) / 0.25)
        card(img, (70, y, W - 70, y + 215), radius=28, fill=(255, 255, 255, int(10 + 12 * p)))
        d = ImageDraw.Draw(img)
        d.text((115, y + 48), it["label"], font=font("bold", 54), fill=C["text"] + (A(p),), anchor="lm")
        d.text((115, y + 112), "EXPECTED", font=font("bold", 30), fill=C["muted"] + (A(p),), anchor="lm")
        d.text((115, y + 160), it["expected"], font=fit_font(d, it["expected"], "bold", 56, 300), fill=C["muted"] + (A(p),), anchor="lm")
        ta = cues.at(it.get("actual_at"), tk + 0.6)
        verdict = it.get("verdict", "inline")
        if t >= ta:
            q = t - ta
            col = C[VERDICT.get(verdict, "accent")]
            d.text((470, y + 112), "ACTUAL", font=font("bold", 30), fill=C["muted"] + (A(q / 0.2),), anchor="lm")
            f = fit_font(d, it["actual"], "bold", int(64 * (0.85 + 0.15 * pop(q / 0.35))), 300)
            d.text((470, y + 160), it["actual"], font=f, fill=col + (A(q / 0.2),), anchor="lm")
            ts = q - 0.12
            if ts > 0:
                sc = 1 + 1.2 * (1 - ease(ts / 0.18))
                _mark(d, W - 165, y + 108, 52, verdict, scale=sc, alpha=A(ts / 0.1))


def scene_timeline(img, v, t, spec, cues):
    """Clock-stamped sequence of events (after-hours stories, multi-day slides)."""
    d = ImageDraw.Draw(img)
    if v.get("title"):
        ctext(d, 410, v["title"].upper(), fit_font(d, v["title"].upper(), "bold", 60, 960), C["muted"])
    items = v["items"][:4]
    n = len(items)
    times = [cues.at(it.get("at"), cues.spread(k, n, 0.05, 0.75)) for k, it in enumerate(items)]
    row = 250
    y0 = 540
    lx = 150
    # spine grows to the newest visible item
    shown = [k for k in range(n) if t >= times[k]]
    if shown:
        k = shown[-1]
        grow = ease((t - times[k]) / 0.3)
        yend = y0 + 40 + row * (k - 1) + row * grow if k > 0 else y0 + 40
        d.line([(lx, y0 + 40), (lx, yend)], fill=(70, 84, 120, 255), width=8)
    active = shown[-1] if shown else -1
    for k, it in enumerate(items):
        if t < times[k]:
            continue
        p = ease((t - times[k]) / 0.3)
        y = y0 + k * row
        on = k == active
        col = colr(it.get("color")) if it.get("color") else (C["accent"] if on else C["muted"])
        r = int(22 * (0.7 + 0.3 * pop((t - times[k]) / 0.3)))
        d.ellipse((lx - r, y + 40 - r, lx + r, y + 40 + r), fill=col + (A(p),))
        x = 220 + int((1 - p) * 50)
        d.text((x, y + 40), it["time"].upper(), font=font("bold", 48), fill=col + (A(p),), anchor="lm")
        for i, line in enumerate(wrap(d, it["text"], font("bold", 62), 760)[:2]):
            d.text((x, y + 112 + i * 70), line, font=font("bold", 62),
                   fill=(C["text"] if on else C["muted"]) + (A(p),), anchor="lm")


def scene_peers(img, v, t, spec, cues):
    """Who else moved: ticker tiles tinted by move, the subject outlined."""
    d = ImageDraw.Draw(img)
    title = v.get("title", "Who else moved")
    ctext(d, 410, title.upper(), fit_font(d, title.upper(), "bold", 60, 960), C["text"])
    items = v["items"][:6]
    n = len(items)
    cols = 2
    tw, th, gx, gy = 430, 210, 40, 36
    x_start = (W - (cols * tw + (cols - 1) * gx)) // 2
    rows = (n + 1) // 2
    y_start = 500 if rows == 3 else 560
    for k, it in enumerate(items):
        tk = cues.at(it.get("at"), cues.spread(k, n, 0.05, 0.6))
        if t < tk:
            continue
        p = ease((t - tk) / 0.25)
        sc = 0.85 + 0.15 * pop((t - tk) / 0.35)
        pct = float(it["pct"])
        col = move_color(pct)
        inten = 0.35 + 0.65 * min(1.0, abs(pct) / 12)
        cx = x_start + (k % cols) * (tw + gx) + tw // 2
        cy = y_start + (k // cols) * (th + gy) + th // 2
        w2, h2 = int(tw * sc) // 2, int(th * sc) // 2
        card(img, (cx - w2, cy - h2, cx + w2, cy + h2), radius=28, fill=col + (int(200 * inten * p),))
        d = ImageDraw.Draw(img)
        if it.get("main"):
            d.rounded_rectangle((cx - w2, cy - h2, cx + w2, cy + h2), radius=28, outline=C["accent"] + (A(p),), width=7)
        d.text((cx, cy - 40), it["ticker"], font=font("bold", 64), fill=C["text"] + (A(p),), anchor="mm")
        d.text((cx, cy + 45), fmt_pct(pct), font=font("bold", 56), fill=C["text"] + (A(p),), anchor="mm")
    if v.get("sub"):
        ts = cues.at(v.get("sub_at"), cues.dur * 0.7)
        yy = y_start + rows * (th + gy) + 30
        for i, line in enumerate(wrap(d, v["sub"], font("medium", 48), 920)):
            ctext(d, yy + i * 60, line, font("medium", 48), C["muted"] + (A((t - ts) / 0.3),))


def _strike(d, x0, x1, y, prog, col, width=10):
    if prog > 0:
        d.line([(x0, y), (x0 + (x1 - x0) * prog, y)], fill=col, width=width)


def scene_mythfact(img, v, t, spec, cues):
    """MYTH card gets struck through, then the FACT card slides in. Contrast hook + payoff."""
    d = ImageDraw.Draw(img)
    tm = cues.at(v.get("myth_at"), 0.0)
    tf = cues.at(v.get("fact_at"), cues.dur * 0.45)
    fm, ff = font("bold", 60), font("bold", 60)
    # MYTH
    pm = ease((t - tm) / 0.3)
    if t >= tm:
        lines = wrap(d, v["myth"], fm, 800)
        hgt = 150 + 74 * len(lines)
        y0 = 450
        dim = 1 - 0.5 * ease((t - tf) / 0.3)
        card(img, (70, y0, W - 70, y0 + hgt), radius=32, fill=C["down"] + (int(60 * pm * dim),))
        d = ImageDraw.Draw(img)
        d.text((120, y0 + 55), "MYTH", font=font("bold", 44), fill=C["down"] + (A(pm * dim),), anchor="lm")
        for i, line in enumerate(lines):
            d.text((120, y0 + 130 + i * 74), line, font=fm, fill=C["text"] + (A(pm * dim),), anchor="lm")
            _strike(d, 120, 120 + text_w(d, line, fm), y0 + 130 + i * 74, ease((t - tf + 0.1) / 0.25), C["down"] + (255,))
        ybottom = y0 + hgt
    else:
        return
    # FACT
    if t >= tf:
        pf = ease((t - tf - 0.1) / 0.3)
        lines = wrap(d, v["fact"], ff, 800)
        hgt2 = 150 + 74 * len(lines)
        y1 = ybottom + 60 + int((1 - pf) * 70)
        card(img, (70, y1, W - 70, y1 + hgt2), radius=32, fill=C["up"] + (int(70 * pf),))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((70, y1, W - 70, y1 + hgt2), radius=32, outline=C["up"] + (A(pf),), width=5)
        d.text((120, y1 + 55), "FACT", font=font("bold", 44), fill=C["up"] + (A(pf),), anchor="lm")
        for i, line in enumerate(lines):
            d.text((120, y1 + 130 + i * 74), line, font=ff, fill=C["text"] + (A(pf),), anchor="lm")
        if v.get("sub"):
            ts = cues.at(v.get("sub_at"), tf + 1.0)
            for i, line in enumerate(wrap(d, v["sub"], font("medium", 46), 900)):
                ctext(d, y1 + hgt2 + 70 + i * 58, line, font("medium", 46), C["muted"] + (A((t - ts) / 0.3),))


def scene_quiz(img, v, t, spec, cues):
    """Guess-then-reveal. Options appear on their words; at the reveal the right one lights up."""
    d = ImageDraw.Draw(img)
    fq = font("bold", 66)
    qlines = wrap(d, v["question"], fq, 940)
    for i, line in enumerate(qlines):
        ctext(d, 440 + i * 80, line, fq, C["text"])
    opts = v["options"][:3]
    n = len(opts)
    tr = cues.at(v.get("reveal_at"), cues.dur * 0.7)
    y = 440 + len(qlines) * 80 + 60
    for k, o in enumerate(opts):
        tk = cues.at(o.get("at"), cues.spread(k, n, 0.12, 0.5))
        if t < tk:
            y += 190
            continue
        p = ease((t - tk) / 0.25)
        correct = k == int(v["answer"])
        rev = t >= tr
        col = C["up"] if (rev and correct) else C["muted"]
        a = 1.0 if not rev else (1.0 if correct else 0.45)
        fillc = (C["up"] + (int(70 * p),)) if (rev and correct) else (255, 255, 255, int(16 * p))
        card(img, (70, y, W - 70, y + 160), radius=30, fill=fillc)
        d = ImageDraw.Draw(img)
        if rev and correct:
            d.rounded_rectangle((70, y, W - 70, y + 160), radius=30, outline=C["up"] + (255,), width=6)
        d.ellipse((115, y + 40, 195, y + 120), fill=(col if rev and correct else C["accent"]) + (A(p * a),))
        d.text((155, y + 80), "ABC"[k], font=font("bold", 50), fill=(10, 14, 26, A(p * a)), anchor="mm")
        d.text((235, y + 80), o["text"], font=fit_font(d, o["text"], "bold", 54, 640), fill=C["text"] + (A(p * a),), anchor="lm")
        if rev and correct:
            q = t - tr - 0.1
            if q > 0:
                _mark(d, W - 165, y + 80, 46, "good", scale=1 + 1.2 * (1 - ease(q / 0.18)), alpha=A(q / 0.1))
        y += 190
    if v.get("reveal_sub") and t >= tr:
        for i, line in enumerate(wrap(d, v["reveal_sub"], font("medium", 48), 920)):
            ctext(d, y + 40 + i * 60, line, font("medium", 48), C["muted"] + (A((t - tr - 0.2) / 0.3),))


def scene_poll(img, v, t, spec, cues):
    """Either/or opinion prompt that ends on a comment call-to-action. No advice, no 'would you buy'."""
    d = ImageDraw.Draw(img)
    fq = font("bold", 70)
    for i, line in enumerate(wrap(d, v["question"], fq, 940)):
        ctext(d, 470 + i * 84, line, fq, C["text"] + (A(t / 0.25),))
    tl = cues.at(v["left"].get("at"), 0.4)
    tr = cues.at(v["right"].get("at"), 0.9)
    for k, (side, tk) in enumerate((("left", tl), ("right", tr))):
        if t < tk:
            continue
        it = v[side]
        p = ease((t - tk) / 0.3)
        sc = 0.9 + 0.1 * pop((t - tk) / 0.35)
        w2, h2 = int(430 * sc), int(480 * sc)
        cx = 70 + 220 + k * 500
        cy = 880
        col = colr(it.get("color"), "accent")
        card(img, (cx - w2 // 2, cy - h2 // 2, cx + w2 // 2, cy + h2 // 2), radius=36, fill=(255, 255, 255, int(24 * p)))
        d = ImageDraw.Draw(img)
        d.ellipse((cx - 44, cy - h2 // 2 + 30, cx + 44, cy - h2 // 2 + 118), fill=col + (A(p),))
        d.text((cx, cy - h2 // 2 + 74), "AB"[k], font=font("bold", 56), fill=(10, 14, 26, A(p)), anchor="mm")
        f = fit_font(d, it["label"], "bold", 64, 380)
        for i, line in enumerate(wrap(d, it["label"], f, 380)[:2]):
            ctext(d, cy + 10 + i * 70, line, f, C["text"] + (A(p),), x=cx)
        for i, line in enumerate(wrap(d, it.get("sub", ""), font("medium", 42), 395)[:3]):
            ctext(d, cy + 118 + i * 50, line, font("medium", 42), C["muted"] + (A(p),), x=cx)
    if t >= min(tl, tr) + 0.4:
        ctext(ImageDraw.Draw(img), 880 - 240 + 74, "OR", font("bold", 40), C["accent"] + (A((t - tr) / 0.2),))
    tc = cues.at(v.get("cta_at"), cues.dur * 0.65)
    if t > tc:
        p = pop((t - tc) / 0.4)
        cta = v.get("cta", "Comment A or B")
        f = font("bold", 56)
        w = text_w(d, cta, f) + 100
        x0 = (W - w) // 2
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((x0, 1200, x0 + w, 1200 + int(116 * max(0.2, p))), radius=58, fill=C["accent"])
        if p > 0.5:
            ctext(d, 1258, cta, f, (10, 10, 10))


def scene_decoder(img, v, t, spec, cues):
    """Jargon decoder: term -> plain English. The explainer beat that non-investors save and share."""
    d = ImageDraw.Draw(img)
    title = v.get("title", "In plain English")
    ctext(d, 420, title.upper(), fit_font(d, title.upper(), "bold", 60, 960), C["muted"])
    items = v["items"][:3]
    n = len(items)
    y = 500
    for k, it in enumerate(items):
        tk = cues.at(it.get("at"), cues.spread(k, n, 0.05, 0.7))
        tp = cues.at(it.get("plain_at"), tk + 0.7)
        if t < tk:
            y += 270
            continue
        p = ease((t - tk) / 0.3)
        card(img, (70, y, W - 70, y + 240), radius=32, fill=(255, 255, 255, int(18 * p)))
        d = ImageDraw.Draw(img)
        d.text((120, y + 62), it["term"].upper(), font=fit_font(d, it["term"].upper(), "bold", 58, 860), fill=C["accent"] + (A(p),), anchor="lm")
        if t >= tp:
            q = ease((t - tp) / 0.3)
            d.text((120, y + 128), "=", font=font("bold", 50), fill=C["muted"] + (A(q),), anchor="lm")
            for i, line in enumerate(wrap(d, it["plain"], font("bold", 50), 760)[:2]):
                d.text((180, y + 128 + i * 56), line, font=font("bold", 50), fill=C["text"] + (A(q),), anchor="lm")
        y += 270


def scene_shift(img, v, t, spec, cues):
    """Before -> after: two big numbers with an arrow. The 'what changed' beat."""
    d = ImageDraw.Draw(img)
    if v.get("title"):
        ctext(d, 440, v["title"].upper(), fit_font(d, v["title"].upper(), "bold", 52, 960), C["muted"] + (A(t / 0.25),))
    t0 = cues.at(v["from"].get("at"), 0.05)
    t1 = cues.at(v["to"].get("at"), cues.dur * 0.5)
    y0, h = 560, 400
    boxes = ((70, 470, v["from"], t0, "muted"), (610, 1010, v["to"], t1, v["to"].get("color", "accent")))
    for k, (x0, x1, it, tk, c) in enumerate(boxes):
        if t < tk:
            continue
        p = ease((t - tk) / 0.3)
        sc = 0.92 + 0.08 * pop((t - tk) / 0.35)
        cx, cy = (x0 + x1) // 2, y0 + h // 2
        w2, h2 = int((x1 - x0) * sc), int(h * sc)
        col = colr(c, "muted") if k == 0 else colr(c, "accent")
        card(img, (cx - w2 // 2, cy - h2 // 2, cx + w2 // 2, cy + h2 // 2), radius=36,
             fill=(col + (int(46 * p),)) if k else (255, 255, 255, int(20 * p)))
        d = ImageDraw.Draw(img)
        if k:
            d.rounded_rectangle((cx - w2 // 2, cy - h2 // 2, cx + w2 // 2, cy + h2 // 2), radius=36, outline=col + (A(p),), width=5)
        val = str(it["value"])
        ctext(d, cy - 30, val, fit_font(d, val, "bold", 130, (x1 - x0) - 50), (C["text"] if k else C["muted"]) + (A(p),), x=cx)
        for i, line in enumerate(wrap(d, it.get("label", ""), font("medium", 40), (x1 - x0) - 60)[:2]):
            ctext(d, cy + 90 + i * 50, line, font("medium", 40), C["muted"] + (A(p),), x=cx)
    if t >= t1:
        q = ease((t - t1 + 0.15) / 0.25)
        ax = 540
        d = ImageDraw.Draw(img)
        d.polygon([(ax - 38, y0 + h // 2 - 36), (ax + 14, y0 + h // 2 - 36), (ax + 14, y0 + h // 2 - 62),
                   (ax + 60, y0 + h // 2), (ax + 14, y0 + h // 2 + 62), (ax + 14, y0 + h // 2 + 36),
                   (ax - 38, y0 + h // 2 + 36)], fill=C["accent"] + (A(q),))
    if v.get("sub"):
        ts = cues.at(v.get("sub_at"), t1 + 0.9)
        for i, line in enumerate(wrap(d, v["sub"], font("bold", 54), 940)):
            ctext(d, y0 + h + 110 + i * 68, line, font("bold", 54), C["text"] + (A((t - ts) / 0.3),))


def scene_receipts(img, v, t, spec, cues):
    """'What the sources say': up to 3 source cards with a check mark. Builds trust in the sourcing."""
    d = ImageDraw.Draw(img)
    title = v.get("title", "What the sources say")
    ctext(d, 430, title.upper(), fit_font(d, title.upper(), "bold", 56, 960), C["muted"] + (A(t / 0.25),))
    items = v["items"][:3]
    n = len(items)
    y = 510
    for k, it in enumerate(items):
        tk = cues.at(it.get("at"), cues.spread(k, n, 0.05, 0.7))
        if t < tk:
            y += 270
            continue
        p = ease((t - tk) / 0.3)
        card(img, (70, y + int((1 - p) * 40), W - 70, y + 240 + int((1 - p) * 40)), radius=32, fill=(255, 255, 255, int(20 * p)))
        d = ImageDraw.Draw(img)
        yy = y + int((1 - p) * 40)
        src = it["source"].upper()
        fs = font("bold", 34)
        d.rounded_rectangle((110, yy + 28, 110 + text_w(d, src, fs) + 40, yy + 82), radius=14, fill=(245, 247, 251, A(p)))
        d.text((130, yy + 55), src, font=fs, fill=(14, 18, 30, A(p)), anchor="lm")
        fb = font("bold", 48)
        for i, line in enumerate(wrap(d, it["fact"], fb, 700)[:2]):
            d.text((110, yy + 128 + i * 56), line, font=fb, fill=C["text"] + (A(p),), anchor="lm")
        q = t - tk - 0.25
        if q > 0:
            _mark(d, W - 150, yy + 120, 44, "good", scale=1 + 1.2 * (1 - ease(q / 0.18)), alpha=A(q / 0.1))
        y += 270


def scene_qa(img, v, t, spec, cues):
    """Question / answer pairs (generic labels, never invented people). 1-2 pairs."""
    d = ImageDraw.Draw(img)
    y = 430
    fq, fa = font("bold", 44), font("bold", 44)
    for k, it in enumerate(v["items"][:2]):
        tq = cues.at(it.get("q_at"), cues.spread(k, len(v["items"][:2]), 0.05, 0.6))
        ta = cues.at(it.get("a_at"), tq + 1.0)
        ql = wrap(d, it["q"], fq, 760)
        al = wrap(d, it["a"], fa, 760)
        hq, ha = 100 + 52 * len(ql), 100 + 52 * len(al)
        if t >= tq:
            p = ease((t - tq) / 0.3)
            card(img, (70, y, 70 + 880, y + hq), radius=32, fill=(255, 255, 255, int(20 * p)))
            d = ImageDraw.Draw(img)
            d.text((110, y + 40), "THE QUESTION", font=font("bold", 30), fill=C["muted"] + (A(p),), anchor="lm")
            for i, line in enumerate(ql):
                d.text((110, y + 82 + i * 52), line, font=fq, fill=C["text"] + (A(p),), anchor="lm")
        y += hq + 24
        if t >= ta:
            p = ease((t - ta) / 0.3)
            card(img, (130, y, W - 70, y + ha), radius=32, fill=C["accent"] + (int(46 * p),))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((130, y, W - 70, y + ha), radius=32, outline=C["accent"] + (A(p),), width=4)
            d.text((170, y + 40), "THE ANSWER", font=font("bold", 30), fill=C["accent"] + (A(p),), anchor="lm")
            for i, line in enumerate(al):
                d.text((170, y + 82 + i * 52), line, font=fa, fill=C["text"] + (A(p),), anchor="lm")
        y += ha + 36



# ---- format-creator additions (2026-11-08): "balance" (tug of war) and "split" (donut breakdown) ----
def scene_balance(img, v, t, spec, cues):
    """TUG OF WAR: items pulling the stock up (left pan) vs down (right pan). The beam tips as each item is
    spoken; heavier = bigger 'weight' (1-3, an editorial size, never a number on screen). Ends on a 'net' line."""
    import math
    d = ImageDraw.Draw(img)
    title = v.get("title", "Two forces, one stock")
    ctext(d, 405, title.upper(), fit_font(d, title.upper(), "bold", 56, 940), C["muted"] + (A(t / 0.25),))
    sides = (("up", v.get("up", [])[:3], 300), ("down", v.get("down", [])[:3], 780))
    n_all = sum(len(x[1]) for x in sides)
    plan, k = [], 0
    for key, items, cx in sides:
        for it in items:
            tk = cues.at(it.get("at"), cues.spread(k, n_all, 0.08, 0.7))
            plan.append((key, it, tk, cx))
            k += 1
    net = 0.0
    for key, it, tk, cx in plan:
        w = max(1, min(3, int(it.get("weight", 1))))
        net += (w if key == "down" else -w) * ease((t - tk) / 0.55)
    ang = math.radians(10 * math.tanh(net * 0.7))
    px, py, L = 540, 585, 240
    ends = {"up": (px - L * math.cos(ang), py - L * math.sin(ang)), "down": (px + L * math.cos(ang), py + L * math.sin(ang))}
    d.rounded_rectangle((px - 8, py, px + 8, 1095), radius=8, fill=(70, 84, 120, 255))
    d.rounded_rectangle((px - 110, 1090, px + 110, 1112), radius=11, fill=(70, 84, 120, 255))
    d.line([ends["up"], ends["down"]], fill=(205, 214, 232, 255), width=14)
    for e in ends.values():
        d.ellipse((e[0] - 10, e[1] - 10, e[0] + 10, e[1] + 10), fill=(205, 214, 232, 255))
    d.ellipse((px - 20, py - 20, px + 20, py + 20), fill=C["accent"] + (255,))
    for key, label in (("up", "PULLING UP"), ("down", "PULLING DOWN")):
        cx = 300 if key == "up" else 780
        ctext(d, 478, label, font("bold", 38), C[key] + (A(t / 0.3),), x=cx)
    for key, items, cx in sides:
        ex, ey = ends[key]
        y = ey + 50
        d.line([(ex, ey), (ex, y)], fill=(120, 132, 160, 255), width=5)
        for it in items:
            tk = [x[2] for x in plan if x[1] is it][0]
            if t < tk:
                continue
            p = ease((t - tk) / 0.3)
            w = max(1, min(3, int(it.get("weight", 1))))
            h = 112 + 22 * (w - 1)
            yy = y + (1 - p) * -70
            col = C[key]
            card(img, (ex - 220, yy, ex + 220, yy + h), radius=26, fill=col + (int(46 * p),))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((ex - 220, yy, ex + 220, yy + h), radius=26, outline=col + (A(p),), width=4)
            ft = fit_font(d, it["text"], "bold", 42, 400, 28)
            if it.get("sub"):
                d.text((ex, yy + h * 0.38), it["text"], font=ft, fill=C["text"] + (A(p),), anchor="mm")
                fs = fit_font(d, it["sub"], "medium", 32, 400, 24)
                d.text((ex, yy + h * 0.72), it["sub"], font=fs, fill=C["muted"] + (A(p),), anchor="mm")
            else:
                d.text((ex, yy + h / 2), it["text"], font=ft, fill=C["text"] + (A(p),), anchor="mm")
            y += h + 14
        if y > ey + 50:
            d.rounded_rectangle((ex - 230, y - 6, ex + 230, y + 8), radius=7, fill=(205, 214, 232, 255))
    if v.get("net"):
        tn = cues.at(v.get("net_at"), cues.dur * 0.8)
        if t >= tn:
            q = ease((t - tn) / 0.3)
            col = colr(v.get("net_color"), "text")
            for i, line in enumerate(wrap(d, v["net"], font("bold", 50), 940)[:2]):
                ctext(d, 1165 + i * 58, line, font("bold", 50), col + (A(q),))


SPLIT_COLORS = [(96, 165, 250), (167, 139, 250), (45, 212, 191), (148, 163, 184)]


def scene_split(img, v, t, spec, cues):
    """BREAKDOWN: a donut split into parts (revenue mix, where a raise comes from). Slices sweep in as each
    part is spoken; the newest slice is lifted and its 'pill' (a change such as +89%) shows in the hole."""
    d = ImageDraw.Draw(img)
    title = v.get("title", "Where it came from")
    ctext(d, 405, title.upper(), fit_font(d, title.upper(), "bold", 56, 940), C["muted"] + (A(t / 0.25),))
    items = v["items"][:4]
    n = len(items)
    tot = sum(float(i["value"]) for i in items) or 1.0
    tks = [cues.at(it.get("at"), cues.spread(k, n, 0.05, 0.7)) for k, it in enumerate(items)]
    cx, cy, R, r = 540, 715, 240, 148
    SS = 2
    half = (R + 24) * SS
    lay = Image.new("RGBA", (half * 2, half * 2), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lay)
    ld.pieslice((half - (R + 0) * SS, half - (R + 0) * SS, half + (R + 0) * SS, half + (R + 0) * SS), 0, 360, fill=(30, 40, 68, 255))
    a0 = -90.0
    acts = []
    for k, it in enumerate(items):
        ext = 360.0 * float(it["value"]) / tot
        prog = ease((t - tks[k]) / 0.55)
        nxt = ease((t - tks[k + 1]) / 0.3) if k + 1 < n else 0.0
        act = ease((t - tks[k]) / 0.3) * (1 - nxt)
        acts.append(act)
        if prog > 0:
            rr = (R + 16 * act) * SS
            col = colr(it.get("color"), "text") if it.get("color") in ("up", "down") else SPLIT_COLORS[k % 4]
            e = a0 + ext * prog - 0.7
            if e > a0 + 0.7:
                ld.pieslice((half - rr, half - rr, half + rr, half + rr), a0 + 0.7, e, fill=col + (255,))
        a0 += ext
    ld.ellipse((half - r * SS, half - r * SS, half + r * SS, half + r * SS), fill=(0, 0, 0, 0))
    lay = lay.resize((half * 2 // SS, half * 2 // SS), Image.LANCZOS)
    img.alpha_composite(lay, (cx - half // SS, cy - half // SS))
    d = ImageDraw.Draw(img)
    shown = [k for k in range(n) if t >= tks[k]]
    if shown:
        k = shown[-1]
        it = items[k]
        p = ease((t - tks[k]) / 0.3)
        big = it.get("pill") or it.get("display", "")
        pc = colr(it.get("pill_color"), "text") if it.get("pill") else C["text"]
        ctext(d, cy - 18, big, fit_font(d, big, "bold", 92, 270, 48), pc + (A(p),), x=cx)
        if it.get("pill_sub"):
            ctext(d, cy + 52, it["pill_sub"], font("medium", 32), C["muted"] + (A(p),), x=cx)
    for k, it in enumerate(items):
        if t < tks[k]:
            continue
        p = ease((t - tks[k]) / 0.3)
        y = 1015 + k * 72
        col = colr(it.get("color"), "text") if it.get("color") in ("up", "down") else SPLIT_COLORS[k % 4]
        d.ellipse((120, y - 17, 154, y + 17), fill=col + (A(p),))
        on = acts[k]
        tc = mix(C["muted"], C["text"], on)
        d.text((180, y), it["label"], font=font("bold", 44), fill=tc + (A(p),), anchor="lm")
        d.text((960, y), it.get("display", ""), font=font("bold", 48), fill=tc + (A(p),), anchor="rm")


SCENES = {"hook": scene_hook, "headline": scene_headline, "stat": scene_stat, "bars": scene_bars, "compare": scene_compare,
          "line": scene_line, "bullets": scene_bullets, "scorecard": scene_scorecard,
          "text": scene_text, "outro": scene_outro, "question": scene_question, "rank": scene_rank,
          "chain": scene_chain, "beatmiss": scene_beatmiss, "timeline": scene_timeline, "peers": scene_peers,
          "mythfact": scene_mythfact, "quiz": scene_quiz, "poll": scene_poll, "decoder": scene_decoder, "shift": scene_shift, "receipts": scene_receipts, "qa": scene_qa,
          "balance": scene_balance, "split": scene_split}


# ================================================================== frame furniture
def candle_strip(seed: str, width: int, height: int):
    """Faint scrolling candlestick chart behind the content."""
    import random
    rnd = random.Random(seed)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    price, xs = 100.0, 0
    vals = []
    while xs < width:
        o = price
        c = o + rnd.gauss(0, 2.2)
        hi, lo = max(o, c) + abs(rnd.gauss(0, 1.2)), min(o, c) - abs(rnd.gauss(0, 1.2))
        vals.append((xs, o, c, hi, lo))
        price = c
        xs += 40
    allp = [v for _, o, c, hi, lo in vals for v in (hi, lo)]
    pmin, pmax = min(allp), max(allp)
    sy = lambda p: int(height - 20 - (p - pmin) / (pmax - pmin + 1e-6) * (height - 40))
    for x, o, c, hi, lo in vals:
        col = C["up"] if c >= o else C["down"]
        d.line([(x + 13, sy(hi)), (x + 13, sy(lo))], fill=col + (60,), width=4)
        d.rectangle((x, sy(max(o, c)), x + 26, max(sy(min(o, c)), sy(max(o, c)) + 4)), fill=col + (60,))
    return img


def tape_strip(items: list[str]):
    """Scrolling ticker tape; each item measured and placed on its own, dot separators drawn."""
    f = font("medium", 34)
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    gap = 90
    widths = [text_w(probe, it, f) for it in items]
    uw = sum(widths) + gap * len(items)
    reps = max(2, (W * 2) // max(1, uw) + 2)
    img = Image.new("RGBA", (uw * reps, 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 0
    for _ in range(reps):
        for it, w in zip(items, widths):
            d.text((x, 30), it, font=f, fill=C["muted"] + (255,), anchor="lm")
            x += w + gap
            d.ellipse((x - gap // 2 - 5, 25, x - gap // 2 + 5, 35), fill=C["accent"] + (255,))
    return img, uw


def header(img, spec, now, total):
    d = ImageDraw.Draw(img)
    has_move = spec.get("ticker") and spec.get("move_pct") is not None
    pct = float(spec["move_pct"]) if has_move else 0.0
    col = move_color(pct) if has_move else C["accent"]
    chip = spec["ticker"] if has_move else spec.get("chip", "EXPLAINED")
    f = font("bold", 60 if has_move else 44)
    chip_w = text_w(d, chip, f) + 64
    y0 = 236
    card(img, (60, y0, 60 + chip_w, y0 + 90), radius=22, fill=col + (70 if has_move else 255,))
    d = ImageDraw.Draw(img)
    d.text((60 + chip_w // 2, y0 + 45), chip, font=f, fill=C["text"] if has_move else (10, 14, 26), anchor="mm")
    d.text((88 + chip_w, y0 + 22), spec.get("company") or spec.get("topic", ""), font=font("bold", 44), fill=C["text"], anchor="lm")
    sub = f"{fmt_pct(pct)}  ·  {spec.get('date_label', '')}" if has_move else spec.get("date_label", "")
    d.text((88 + chip_w, y0 + 70), sub, font=font("medium", 36), fill=col if has_move else C["muted"], anchor="lm")
    d.text((W - 60, y0 + 24), "WHY IT", font=font("bold", 32), fill=C["accent"], anchor="rm")
    d.text((W - 60, y0 + 62), "MOVED?", font=font("bold", 32), fill=C["accent"], anchor="rm")
    d.text((60, y0 + 122), "Educational only · not financial advice", font=font("medium", 26), fill=C["muted"], anchor="lm")
    d.rectangle((0, 214, W, 220), fill=(30, 40, 66))
    d.rectangle((0, 214, int(W * now / total), 220), fill=C["accent"])


def brand_sting(img, now):
    """0.0–0.9s: WHY IT MOVED? stamps in, then flies to the corner."""
    if now > 0.9:
        return
    d = ImageDraw.Draw(img)
    if now < 0.55:
        p = pop(now / 0.3)
        a = 255
        cx, cy, sc = W // 2, 930, 1.0
    else:
        k = ease((now - 0.55) / 0.35)
        a = int(255 * (1 - k))
        cx, cy, sc = int(W // 2 + (W - 220 - W // 2) * k), int(930 + (170 - 930) * k), 1 - 0.7 * k
        p = 1
    size = int(150 * sc * (0.7 + 0.3 * p))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    if now < 0.55:
        ld.rectangle((0, 0, W, H), fill=(6, 9, 18, int(200 * (1 - now / 0.55))))
    ld.text((cx, cy - size * 0.55), "WHY IT", font=font("bold", size), fill=C["text"] + (a,), anchor="mm")
    ld.text((cx, cy + size * 0.55), "MOVED?", font=font("bold", size), fill=C["accent"] + (a,), anchor="mm")
    img.alpha_composite(layer)


# ================================================================== sound design (generated, royalty-free)
def _onepole(x, cut, sr):
    import numpy as np
    y = np.empty_like(x)
    acc = 0.0
    cut = np.broadcast_to(np.asarray(cut, dtype="float64"), x.shape)
    a = 1 - np.exp(-2 * np.pi * cut / sr)
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def sfx(kind: str, sr: int, rng):
    import numpy as np
    if kind.startswith("whoosh"):
        var = int(kind[-1]) if kind[-1].isdigit() else 0
        dur, peak = [(0.42, 3800), (0.3, 5200), (0.5, 2600)][var % 3]
        n = int(dur * sr); tt = np.linspace(0, 1, n)
        cut = 300 + peak * np.sin(np.pi * tt) ** 2
        x = _onepole(rng.standard_normal(n), cut, sr)
        return (x / (np.abs(x).max() + 1e-9)) * np.sin(np.pi * tt) ** 1.5
    if kind == "pop":
        n = int(0.09 * sr); tt = np.arange(n) / sr
        f = 1100 - 4500 * tt
        return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 45)
    if kind == "stamp":
        n = int(0.28 * sr); tt = np.arange(n) / sr
        thump = np.sin(2 * np.pi * np.cumsum(95 - 150 * tt) / sr) * np.exp(-tt * 18)
        click = np.diff(rng.standard_normal(n + 1)) * np.exp(-tt * 160) * 0.35
        return thump + click
    if kind == "impact":
        n = int(0.7 * sr); tt = np.arange(n) / sr
        boom = np.sin(2 * np.pi * np.cumsum(62 - 40 * tt) / sr) * np.exp(-tt * 6)
        rumble = _onepole(rng.standard_normal(n), 400, sr)
        rumble = rumble / (np.abs(rumble).max() + 1e-9) * np.exp(-tt * 9) * 0.5
        return boom + rumble
    if kind == "riser":
        n = int(1.1 * sr); tt = np.linspace(0, 1, n)
        x = _onepole(rng.standard_normal(n), 300 + 5000 * tt ** 2, sr)
        x = x / (np.abs(x).max() + 1e-9)
        tone = np.sin(2 * np.pi * np.cumsum(220 + 900 * tt ** 2) / sr) * 0.3
        return (x * 0.7 + tone) * tt ** 2
    raise ValueError(kind)


def music_bed(total: float, sr: int, rng):
    """Low, tense synth pad + soft pulse, ~100 bpm. Royalty-free (generated)."""
    import numpy as np
    n = int(total * sr)
    out = np.zeros(n)
    bar = 2.4
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    t_all = np.arange(n) / sr
    for i in range(int(total / bar) + 1):
        s0, s1 = int(i * bar * sr), min(n, int((i + 1) * bar * sr) + int(0.3 * sr))
        if s0 >= n:
            break
        tt = t_all[s0:s1] - i * bar
        env = np.minimum(1, tt / 0.35) * np.minimum(1, np.maximum(0, (bar + 0.3 - tt) / 0.5))
        seg = np.zeros(s1 - s0)
        for m in chords[i % 4]:
            for det in (-0.08, 0.08):
                seg += np.sin(2 * np.pi * hz(m - 12) * (1 + det / 100) * tt)
            seg += 0.25 * np.sin(2 * np.pi * hz(m) * tt)
        seg += 0.6 * np.sin(2 * np.pi * hz(chords[i % 4][0] - 24) * tt)  # bass
        out[s0:s1] += seg * env
    # soft pulse on every beat
    beat = 0.6
    for b in range(int(total / beat)):
        s0 = int(b * beat * sr)
        m = min(n - s0, int(0.12 * sr))
        if m <= 0:
            break
        tt = np.arange(m) / sr
        out[s0:s0 + m] += 1.4 * np.sin(2 * np.pi * (60 - 120 * tt) * tt) * np.exp(-tt * 30)
    out = _onepole(out, 1800, sr) if n < sr * 120 else out
    return out / (np.abs(out).max() + 1e-9)


def collect_events(beats: list[dict]) -> list[tuple[float, str]]:
    """Sound effects synced to the same cues that drive the visuals."""
    ev = []
    for i, b in enumerate(beats):
        v, cues, st = b["visual"], b["cues"], b["start"]
        if i > 0:
            ev.append((st - 0.1, f"whoosh{i % 3}"))
        kind = v["type"]
        if kind == "hook":
            ev.append((0.0, "impact"))
            if v.get("price_at"):
                ev.append((st + cues.at(v["price_at"], 0.6), "pop"))
            continue
        if kind == "question":
            if i == 0:
                ev.append((0.0, "impact"))
            if v.get("sub_at"):
                ev.append((st + cues.at(v["sub_at"], 0.8), "pop"))
            continue
        if kind == "rank":
            ev.append((st + 0.02, "pop"))
            ev.append((st + cues.at(v.get("move_at"), 0.25) + 0.05, "stamp"))
            continue
        if kind == "headline":
            ev.append((st + cues.at(v.get("at"), 0.0), "whoosh1"))
            if v.get("highlight_at"):
                ev.append((st + cues.at(v["highlight_at"], 0.6), "pop"))
            continue
        if kind == "beatmiss":
            ev.append((st - 1.1, "riser"))
            for k, it in enumerate(v["items"]):
                tk = cues.at(it.get("at"), cues.spread(k, len(v["items"]), 0.05, 0.6))
                ev.append((st + tk, "pop"))
                ev.append((st + cues.at(it.get("actual_at"), tk + 0.6) + 0.12, "stamp"))
            continue
        if kind == "mythfact":
            ev.append((st + cues.at(v.get("myth_at"), 0.0), "pop"))
            tf = cues.at(v.get("fact_at"), cues.dur * 0.45)
            ev.append((st + tf, "whoosh1"))
            ev.append((st + tf + 0.2, "stamp"))
            continue
        if kind == "quiz":
            ev.append((st - 0.9, "riser"))
            for k, o in enumerate(v["options"][:3]):
                ev.append((st + cues.at(o.get("at"), cues.spread(k, len(v["options"][:3]), 0.12, 0.5)), "pop"))
            ev.append((st + cues.at(v.get("reveal_at"), cues.dur * 0.7) + 0.1, "stamp"))
            continue
        if kind == "poll":
            for side in ("left", "right"):
                ev.append((st + cues.at(v[side].get("at"), 0.4 if side == "left" else 0.9), "pop"))
            if v.get("cta_at"):
                ev.append((st + cues.at(v["cta_at"], cues.dur * 0.65), "pop"))
            continue
        if kind == "shift":
            ev.append((st + cues.at(v["from"].get("at"), 0.05), "pop"))
            t1 = cues.at(v["to"].get("at"), cues.dur * 0.5)
            ev.append((st + t1, "whoosh1"))
            ev.append((st + t1 + 0.2, "stamp"))
            continue
        if kind == "qa":
            for k, it in enumerate(v["items"][:2]):
                tq = cues.at(it.get("q_at"), cues.spread(k, len(v["items"][:2]), 0.05, 0.6))
                ev.append((st + tq, "pop"))
                ev.append((st + cues.at(it.get("a_at"), tq + 1.0), "stamp"))
            continue
        if kind == "receipts":
            for k, it in enumerate(v["items"][:3]):
                tk = cues.at(it.get("at"), cues.spread(k, len(v["items"][:3]), 0.05, 0.7))
                ev.append((st + tk, "pop"))
                ev.append((st + tk + 0.3, "stamp"))
            continue
        if kind == "balance":
            allit = [x for x in v.get("up", [])[:3]] + [x for x in v.get("down", [])[:3]]
            for k, it in enumerate(allit):
                ev.append((st + cues.at(it.get("at"), cues.spread(k, len(allit), 0.08, 0.7)), "pop"))
            if v.get("net"):
                ev.append((st + cues.at(v.get("net_at"), cues.dur * 0.8), "stamp"))
            continue
        if kind == "split":
            its = v["items"][:4]
            for k, it in enumerate(its):
                tk = cues.at(it.get("at"), cues.spread(k, len(its), 0.05, 0.7))
                ev.append((st + tk, "pop"))
                if it.get("pill"):
                    ev.append((st + tk + 0.25, "stamp"))
            continue
        if kind == "decoder":
            for k, it in enumerate(v["items"][:3]):
                tk = cues.at(it.get("at"), cues.spread(k, len(v["items"][:3]), 0.05, 0.7))
                ev.append((st + tk, "pop"))
                ev.append((st + cues.at(it.get("plain_at"), tk + 0.7), "stamp"))
            continue
        if kind in ("chain", "timeline", "peers"):
            items = v["items"]
            lo, hi = (0.05, 0.6) if kind == "peers" else (0.05, 0.75)
            for k, it in enumerate(items):
                ev.append((st + cues.at(it.get("at"), cues.spread(k, len(items), lo, hi)), "pop"))
            if kind == "chain" and items:
                ev.append((st + cues.at(items[-1].get("at"), cues.spread(len(items) - 1, len(items), lo, hi)) + 0.05, "stamp"))
            continue
        if kind == "scorecard":
            ev.append((st - 1.1, "riser"))
            for k, it in enumerate(v["items"]):
                ev.append((st + cues.at(it.get("at"), cues.spread(k, len(v["items"]), 0.1, 0.75)) + 0.12, "stamp"))
            continue
        for key in ("at", "label_at"):
            if key in v and kind != "hook":
                ev.append((st + cues.at(v[key], 0.05), "pop"))
        for coll in ("items", "points"):
            for k, it in enumerate(v.get(coll, []) if isinstance(v.get(coll), list) else []):
                if isinstance(it, dict) and it.get("at"):
                    ev.append((st + cues.at(it["at"], 0.1), "pop"))
        for side in ("left", "right"):
            if isinstance(v.get(side), dict) and v[side].get("at"):
                ev.append((st + cues.at(v[side]["at"], 0.1), "pop"))
        if kind == "outro" and v.get("cta_at"):
            ev.append((st + cues.at(v["cta_at"], 0.5), "pop"))
    return sorted((max(0.0, t), k) for t, k in ev)


# ================================================================== captions
def _ass_time(t):
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def build_ass(words, keywords, fontname):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{fontname},80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,8,3,2,60,100,545,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev, chunks, cur = [], [], []
    for w in words:
        cur.append(w)
        if len(cur) == 3 or re.search(r"[.!?]$", w["word"]):
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    for ci, chunk in enumerate(chunks):
        nxt = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else None
        for j, w in enumerate(chunk):
            st = w["start"]
            en = chunk[j + 1]["start"] if j + 1 < len(chunk) else (min(nxt, w["end"] + 0.35) if nxt else w["end"] + 0.4)
            parts = []
            for k, o in enumerate(chunk):
                txt = o["word"].upper().replace("{", "").replace("}", "")
                kw = _norm(o["word"]) in keywords
                if k == j:
                    col = "&H0000C8FF" if kw else "&H00FFFFFF"
                    parts.append(f"{{\\c{col}\\fscx114\\fscy114}}{txt}{{\\c&H00FFFFFF\\fscx100\\fscy100}}")
                else:
                    parts.append(f"{{\\c&H0000C8FF}}{txt}{{\\c&H00FFFFFF}}" if kw else f"{{\\alpha&H40&}}{txt}{{\\alpha&H00&}}")
            anim = r"{\fscx80\fscy80\t(0,120,\fscx100\fscy100)}" if j == 0 else ""
            ev.append(f"Dialogue: 1,{_ass_time(st)},{_ass_time(max(en, st + 0.05))},Cap,,0,0,0,,{anim}{' '.join(parts)}")
    return head + "\n".join(ev) + "\n"


# ================================================================== render
OUT_SR = 48000


def _measure_loudness(path: Path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr
    try:
        return json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])
    except ValueError:
        return None


def render(spec_path: Path, out_dir: Path, music: Path | None = None, voice: str = "am_michael",
           speed: float = 1.08, no_music: bool = False) -> Path:
    import numpy as np

    spec = json.loads(spec_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = spec.get("slug") or f"{spec.get('ticker', 'clip')}-{spec.get('date', 'clip')}"
    work = Path(tempfile.mkdtemp(prefix="stockshort-"))
    voice = spec.get("voice", voice)
    loop = bool(spec.get("loop"))
    lead = 0.12  # first word starts almost immediately; frame 1 is already the hook

    # 1) voice: one take per sentence, trimmed, with measured word timings
    track = [np.zeros(int(lead * SR), dtype="float32")]
    words, beats, t, engine = [], [], lead, None
    for bi, b in enumerate(spec["beats"]):
        bstart, bwords = (0.0 if bi == 0 else t), []
        for si, sent in enumerate(split_sentences(b["say"])):
            wav = work / f"b{bi}_{si}.wav"
            engine = synth(sent, wav, voice, speed)
            a, sr = _read_wav(wav)
            if sr != SR:
                a = np.interp(np.linspace(0, len(a), int(len(a) * SR / sr), endpoint=False), np.arange(len(a)), a)
            s0, s1 = speech_bounds(a, SR)
            a = a[max(0, int((s0 - 0.03) * SR)): int((s1 + 0.06) * SR)].astype("float32")
            speak = s1 - s0
            toks = sent.split()
            wts = [_syllables(x) for x in toks]
            tot, acc = sum(wts), 0.0
            for tok, wt in zip(toks, wts):
                ws = t + 0.03 + speak * acc / tot
                acc += wt
                ent = {"word": tok, "start": round(ws, 3), "end": round(t + 0.03 + speak * acc / tot, 3)}
                words.append(ent)
                bwords.append({**ent, "start": ent["start"] - bstart, "end": ent["end"] - bstart})
            track.append(a)
            t += len(a) / SR
            gap = SENT_GAP
            track.append(np.zeros(int(gap * SR), dtype="float32"))
            t += gap
        last = bi == len(spec["beats"]) - 1
        gap = 0.0 if (last and loop) else BEAT_GAP
        track.append(np.zeros(int(gap * SR), dtype="float32"))
        t += gap
        beats.append({"start": bstart, "dur": t - bstart, "cues": Cues(bwords, t - bstart), **b})
    tail = 0.15 if loop else 0.5
    track.append(np.zeros(int(tail * SR), dtype="float32"))
    total = t + tail
    v24 = np.concatenate(track)
    vo = np.interp(np.linspace(0, len(v24), int(len(v24) * OUT_SR / SR), endpoint=False), np.arange(len(v24)), v24)
    vo = vo / (np.abs(vo).max() + 1e-9) * 0.9
    print(f"  voice: {engine}{' (' + voice + ')' if engine == 'kokoro' else ''}, {total:.1f}s")

    # 2) sound design + music, ducked under the voice
    rng = np.random.default_rng(7)
    mixd = vo.copy()
    gains = {"whoosh0": 0.2, "whoosh1": 0.16, "whoosh2": 0.13, "pop": 0.14, "stamp": 0.5, "impact": 0.45, "riser": 0.16}
    cache = {}
    for tt_, kind in collect_events(beats):
        s = cache.setdefault(kind, sfx(kind, OUT_SR, rng))
        i0 = int(tt_ * OUT_SR)
        i1 = min(len(mixd), i0 + len(s))
        if i0 < len(mixd):
            mixd[i0:i1] += s[: i1 - i0] * gains[kind]
    if music is None and not no_music and not spec.get("no_music"):
        bed = music_bed(total, OUT_SR, rng)
    elif music:
        tmp = work / "music.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", str(music), "-t", f"{total:.2f}",
                        "-ac", "1", "-ar", str(OUT_SR), str(tmp)], check=True)
        bed, _ = _read_wav(tmp)
    else:
        bed = None
    if bed is not None:
        bed = np.pad(bed, (0, max(0, len(mixd) - len(bed))))[: len(mixd)]
        envv = np.convolve(np.abs(vo), np.ones(int(0.25 * OUT_SR)) / int(0.25 * OUT_SR), mode="same")
        duck = 1 - 0.55 * np.clip(envv / 0.08, 0, 1)
        fade = np.ones(len(bed))
        if not loop:
            nf = int(1.2 * OUT_SR)
            fade[-nf:] = np.linspace(1, 0, nf)
        mixd += bed * 0.075 * duck * fade
    mixd = mixd / max(1.0, np.abs(mixd).max() / 0.98)
    audio_wav = work / "mix.wav"
    _write_wav(audio_wav, mixd, OUT_SR)

    # 3) captions
    kws = {_norm(k) for k in spec.get("keywords", [])}
    fontname = Path(_find_font("bold") or "Poppins").stem.split("-")[0]
    ass = work / "caps.ass"
    ass.write_text(build_ass(words, kws, fontname), encoding="utf-8")

    # 4) frames
    bg = background()
    strip = candle_strip(spec.get("ticker") or stem, W * 3, 560)
    default_tape = ([f"{spec['ticker']} {fmt_pct(float(spec['move_pct']))}"] if spec.get("ticker") and spec.get("move_pct") is not None
                    else [spec.get("topic", "Why It Moved")])
    tape, tape_w = tape_strip(spec.get("tape") or default_tape)
    raw = work / "video.mp4"
    enc = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-vf", f"ass='{ass.as_posix()}'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-pix_fmt", "yuv420p", str(raw)], stdin=subprocess.PIPE)
    import random
    shake_rng = random.Random(3)
    bi = 0
    hook_hit = 0.0 if beats[0]["visual"]["type"] in ("hook", "question") else -9
    kw_times = [w["start"] for w in words if _norm(w["word"]) in kws]
    for n in range(int(total * FPS)):
        now = n / FPS
        while bi + 1 < len(beats) and now >= beats[bi + 1]["start"]:
            bi += 1
        b = beats[bi]
        tb = now - b["start"]
        img = bg.copy()
        # animated background: drifting candles + ticker tape (tape sits above the app's bottom text)
        off = int(now * 45) % (W * 2)
        img.alpha_composite(strip.crop((off, 0, off + W, 560)), (0, 980))
        img.alpha_composite(Image.new("RGBA", (W, 64), (6, 9, 18, 235)), (0, 1440))
        toff = int(now * 110) % tape_w
        img.alpha_composite(tape.crop((toff, 0, toff + W, 60)), (0, 1442))
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        SCENES[b["visual"]["type"]](layer, b["visual"], tb, spec, b["cues"])
        z = 1 + 0.04 * ease(tb / max(1.0, b["dur"]))                    # slow camera push
        punch = max([0.0] + [max(0.0, 1 - abs(now - kt - 0.08) / 0.14) for kt in kw_times if -0.1 < now - kt < 0.3])
        z *= 1 + 0.035 * punch                                          # zoom punch on key words
        dx = dy = 0.0
        if bi > 0 and tb < 0.2:                                         # whip-in from the right
            dx = (1 - ease(tb / 0.2)) * 300
            layer.putalpha(layer.getchannel("A").point(lambda a, k=ease(tb / 0.2): int(a * k)))
        if 0 <= now - hook_hit < 0.28:                                  # impact shake on frame 1
            amp = 16 * (1 - (now - hook_hit) / 0.28)
            dx += shake_rng.uniform(-amp, amp); dy += shake_rng.uniform(-amp, amp)
        sc = SAFE_S * z
        cx, cy = SAFE_CX + dx, SAFE_CY + dy
        layer = layer.transform((W, H), Image.AFFINE, (1 / sc, 0, 540 - cx / sc, 0, 1 / sc, 800 - cy / sc),
                                resample=Image.BILINEAR)
        img.alpha_composite(layer)
        header(img, spec, now, total)
        enc.stdin.write(img.tobytes())
        if n % (FPS * 10) == 0:
            print(f"  rendering {now:4.0f}s / {total:.0f}s", flush=True)
    enc.stdin.close()
    enc.wait()

    # 5) mux + two-pass loudness (one even gain change, no pumping)
    out = out_dir / f"{stem}.mp4"
    limited = work / "mix_lim.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(audio_wav), "-af",
                    "alimiter=limit=0.3:attack=4:release=60:level=false", str(limited)], check=True)
    audio_wav = limited
    m = _measure_loudness(audio_wav)
    norm = (f"loudnorm=I=-14:TP=-1.5:LRA=11:linear=true:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}"
            if m else "loudnorm=I=-14:TP=-1.5:LRA=11")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-i", str(audio_wav), "-map", "0:v", "-map", "1:a",
                    "-af", norm, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)

    tags = " ".join(f"#{h}" for h in spec.get("hashtags", []))
    src = "\n".join(f"- {s['title']}: {s['url']}" for s in spec.get("sources", []))
    (out_dir / f"{stem}.txt").write_text(
        f"TITLE:\n{spec['title']}\n\nDESCRIPTION:\n{spec.get('description', '')}\n\nSources:\n{src}\n\n"
        f"Educational content only, not financial advice. Do your own research.\n\n{tags}\n\n"
        f"UPLOAD NOTE: narration is an AI voice — tick YouTube's 'altered or synthetic content' box and "
        f"TikTok's 'AI-generated' label.\n")
    shutil.rmtree(work, ignore_errors=True)
    print(f"Saved {out} ({total:.1f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--music", type=Path, help="licensed music track to use instead of the generated bed")
    ap.add_argument("--no-music", action="store_true", help="voice and sound effects only")
    ap.add_argument("--voice", default=os.environ.get("STOCKSHORT_VOICE", "am_michael"), help="Kokoro voice id")
    ap.add_argument("--speed", type=float, default=1.08)
    a = ap.parse_args()
    render(a.spec, a.out, a.music, a.voice, a.speed, a.no_music)


if __name__ == "__main__":
    main()
