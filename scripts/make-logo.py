#!/usr/bin/env python3
"""Render the vllm.cpp logo assets from one source-of-truth geometry.

The mark ("Steps") is the V1 scheduler's own timeline: requests as bars crossing
step boundaries, joining and retiring mid-flight, which is exactly what
continuous batching does and exactly what a static-batch engine cannot draw.

Palette is lifted from the project's own figure renderer
(benchmarks/demo/concurrency_race.py) so the logo, the race clip and the
footprint figure read as one identity: ground #0d1117, ink #d7dde5, teal
#3ec8e0. The light-ground variant darkens the teal to #0f7f96, which the
dark-ground teal cannot carry on white.

Two mark variants ship, because the full 7-bar timeline flattens into a barcode
below about 24px:
  full  - 7 bars + 2 step-boundary rules, for the README lockup
  small - 3 bars, no rules, for the favicon and any <=32px use

Outputs (all written to assets/):
  logo-dark.png / logo-light.png  the README lockup, transparent background
  logo.svg / logo-light.svg       vector lockup (text needs a mono face present)
  logo-mark.svg                   mark only, full variant
  logo-mark-small.svg             mark only, small variant
  favicon.png                     32px small variant

Usage: python3 scripts/make-logo.py
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

# (x, y, width, weight) on a 64x64 viewBox, bar height 6.5, corner radius 2.
# Rows are the batch slots; the x stagger is requests arriving and retiring at
# different steps rather than a synchronized static batch. "weight" is NOT alpha:
# it indexes a solid teal ramp (see _ramp). Translucent bars let the step-boundary
# rules bleed through and read as scratches, so every bar is opaque and the rules
# show only in the gutters between rows.
BARS_FULL = (
    (4, 9, 26, 0.90),
    (34, 9, 26, 0.45),
    (4, 20, 42, 1.00),
    (12, 31, 48, 0.68),
    (4, 42, 16, 0.45),
    (24, 42, 36, 0.90),
    (20, 53, 32, 0.60),
)
RULES_FULL = (23.5, 42.5)  # step boundaries

# (x, y, width, height, weight), 3 bars, readable at 16px.
BARS_SMALL = (
    (3, 11, 40, 14, 1.00),
    (14, 27, 47, 14, 0.70),
    (3, 43, 29, 14, 0.45),
)

THEMES = {
    "dark": {"accent": "#3ec8e0", "ground": "#0d1117", "ink": "#d7dde5"},
    "light": {"accent": "#0f7f96", "ground": "#ffffff", "ink": "#131a20"},
}


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))


def _blend(a: str, b: str, t: float) -> str:
    """Blend hex colour a toward b by t, returning hex."""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(round(ca + (cb - ca) * t) for ca, cb in zip(ra, rb)))


def _ramp(theme: dict[str, str], weight: float) -> str:
    """Solid colour for a bar weight: accent blended toward the ground."""
    return _blend(theme["accent"], theme["ground"], (1.0 - weight) * 0.72)


def _rule_colour(theme: dict[str, str]) -> str:
    """Step-boundary rule: a hairline lifted just off the ground."""
    return _blend(theme["ground"], theme["ink"], 0.22)

MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
MONO_FAMILY = "DejaVu Sans Mono, ui-monospace, SFMono-Regular, Menlo, monospace"


def mark_svg(theme: dict[str, str], variant: str = "full") -> str:
    """Return a standalone 64x64 SVG for the mark."""
    body: list[str] = []
    if variant == "full":
        for x in RULES_FULL:
            body.append(
                f'<path d="M{x} 6 V58" stroke="{_rule_colour(theme)}" stroke-width="1.4"/>'
            )
        for x, y, w, weight in BARS_FULL:
            body.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="6.5" rx="2" '
                f'fill="{_ramp(theme, weight)}"/>'
            )
    else:
        for x, y, w, h, weight in BARS_SMALL:
            body.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" '
                f'fill="{_ramp(theme, weight)}"/>'
            )
    inner = "\n  ".join(body)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
        'width="64" height="64" role="img" aria-label="vllm.cpp">\n  '
        f"{inner}\n</svg>\n"
    )


def lockup_svg(theme: dict[str, str]) -> str:
    """Return the horizontal lockup: mark, then the wordmark."""
    body: list[str] = []
    for x in RULES_FULL:
        body.append(f'<path d="M{x} 6 V58" stroke="{_rule_colour(theme)}" stroke-width="1.4"/>')
    for x, y, w, weight in BARS_FULL:
        body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="6.5" rx="2" '
            f'fill="{_ramp(theme, weight)}"/>'
        )
    inner = "\n    ".join(body)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 210 64" '
        'width="210" height="64" role="img" aria-label="vllm.cpp">\n'
        f"  <g>\n    {inner}\n  </g>\n"
        f'  <text x="76" y="41" font-family="{MONO_FAMILY}" font-size="27" letter-spacing="-1.6" '
        f'font-weight="700" fill="{theme["ink"]}">vllm<tspan '
        f'fill="{theme["accent"]}">.cpp</tspan></text>\n'
        "</svg>\n"
    )


def rasterize(svg: str, px: int) -> Image.Image:
    """Rasterize an SVG string to a square RGBA image of px on a side."""
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=px, output_height=px)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def lockup_png(theme: dict[str, str], scale: int = 2) -> Image.Image:
    """Compose the lockup with Pillow so the wordmark never silently falls back.

    cairosvg resolves <text> against whatever fonts the host happens to have;
    the PNG is what the README actually renders, so its type is drawn from the
    DejaVu Sans Mono Bold file directly.
    """
    mark_px = 130 * scale
    size = 54 * scale
    gap = 15 * scale
    pad = 4 * scale
    # DejaVu Sans Mono ships at a wide default advance; a logo wants it tighter
    # than running code, so the wordmark is drawn glyph by glyph with negative
    # tracking rather than in one d.text() call.
    track = -0.058 * size

    font = ImageFont.truetype(MONO_BOLD, size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    word = "vllm.cpp"
    advances = [probe.textlength(ch, font=font) + track for ch in word]
    text_w = int(sum(advances) - track)

    width = pad + mark_px + gap + text_w + pad
    height = pad + mark_px + pad
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.alpha_composite(rasterize(mark_svg(theme, "full"), mark_px), (pad, pad))

    d = ImageDraw.Draw(canvas)
    # Centre the drawn ink of the wordmark on the mark's centre line.
    box = d.textbbox((0, 0), word, font=font)
    ty = pad + (mark_px - (box[3] + box[1])) // 2
    x = float(pad + mark_px + gap)
    ext = word.index(".")  # ".cpp" onward is the accent-coloured extension
    for i, (ch, adv) in enumerate(zip(word, advances)):
        d.text((x, ty), ch, font=font,
               fill=theme["accent"] if i >= ext else theme["ink"])
        x += adv
    return canvas


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    written: list[str] = []

    for name, theme in THEMES.items():
        out = ASSETS / f"logo-{name}.png"
        lockup_png(theme).save(out)
        written.append(f"{out.name} ({Image.open(out).size[0]}x{Image.open(out).size[1]})")

    (ASSETS / "logo.svg").write_text(lockup_svg(THEMES["dark"]), encoding="utf-8")
    (ASSETS / "logo-light.svg").write_text(lockup_svg(THEMES["light"]), encoding="utf-8")
    (ASSETS / "logo-mark.svg").write_text(mark_svg(THEMES["dark"], "full"), encoding="utf-8")
    (ASSETS / "logo-mark-small.svg").write_text(
        mark_svg(THEMES["dark"], "small"), encoding="utf-8"
    )
    written += ["logo.svg", "logo-light.svg", "logo-mark.svg", "logo-mark-small.svg"]

    rasterize(mark_svg(THEMES["dark"], "small"), 32).save(ASSETS / "favicon.png")
    written.append("favicon.png (32x32)")

    for w in written:
        print(f"wrote assets/{w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
