#!/usr/bin/env python3
r"""footprint - render the "what you actually install" contrast.

This is the one axis where the gap is enormous rather than a few percent, so it
is the honest hero visual: vLLM needs a Python/PyTorch/CUDA stack on disk before
it serves a token; vllm.cpp is a shared library plus a server binary.

Every number is MEASURED and its provenance is recorded in the spec's
"_source" fields. Nothing is estimated. Where a figure is not yet measured the
spec must omit the row rather than guess, and the renderer will simply not draw
it.

  python3 footprint.py                     # 16:9 still (PNG)
  python3 footprint.py --layout square     # 1:1 for social

Spec JSON (see footprint_gb10.json):
  {
    "title": "...", "note": "...",
    "groups": [
      {"label": "vLLM 0.26.0.dev0 venv", "accent": "slate", "total_mib": 9728,
       "parts": [{"label": "nvidia CUDA libs", "mib": 3174}, ...]},
      {"label": "vllm.cpp", "accent": "teal", "total_mib": 53, "parts": [...]}
    ]
  }
"""
import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

BG = (13, 17, 23)
INK = (215, 221, 229)
DIM = (110, 118, 129)
PANEL = (22, 27, 34)
ACCENTS = {
    "teal": (62, 200, 224),
    "slate": (148, 163, 178),
    "green": (126, 231, 135),
    "amber": (255, 207, 86),
    "violet": (175, 145, 245),
    "rose": (244, 130, 150),
}
LAYOUTS = {"cols": (1280, 720), "square": (1080, 1080)}


def font(sz, bold=True, mono=False):
    fam = "DejaVuSansMono" if mono else "DejaVuSans"
    suffix = "-Bold" if bold else ""
    try:
        return ImageFont.truetype(
            f"/usr/share/fonts/truetype/dejavu/{fam}{suffix}.ttf", sz
        )
    except OSError:
        return ImageFont.load_default()


def human(mib):
    return f"{mib / 1024:.1f} GiB" if mib >= 1024 else f"{mib:.0f} MiB"


def render(spec, layout, out):
    W, H = LAYOUTS[layout]
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)

    d.text((40, 30), spec["title"], fill=INK, font=font(30))
    d.text((40, 72), spec["subtitle"], fill=DIM, font=font(17, False))

    groups = spec["groups"]
    peak = max(g["total_mib"] for g in groups)
    top = 140
    slot = (H - top - 150) // len(groups)

    for gi, g in enumerate(groups):
        y = top + gi * slot
        accent = ACCENTS.get(g.get("accent", "teal"), ACCENTS["teal"])
        d.text((40, y), g["label"], fill=accent, font=font(22))
        d.text(
            (40, y + 30), g.get("detail", ""), fill=DIM, font=font(15, False)
        )

        bar_y = y + 58
        bar_h = 42
        track_w = W - 80 - 190
        d.rounded_rectangle([40, bar_y, 40 + track_w, bar_y + bar_h], 6, fill=PANEL)

        # Stacked parts, so the bulk is attributable rather than asserted.
        x = 40
        for pi, part in enumerate(g.get("parts", [])):
            pw = int(track_w * part["mib"] / peak)
            if pw < 1:
                continue
            shade = tuple(
                max(0, min(255, int(c * (1.0 - 0.16 * (pi % 4))))) for c in accent
            )
            d.rectangle([x, bar_y, x + pw, bar_y + bar_h], fill=shade)
            if pw > 92:
                d.text(
                    (x + 8, bar_y + 6),
                    part["label"],
                    fill=BG,
                    font=font(13, False),
                )
                d.text(
                    (x + 8, bar_y + 22),
                    human(part["mib"]),
                    fill=BG,
                    font=font(13, True, mono=True),
                )
            x += pw

        total_w = int(track_w * g["total_mib"] / peak)
        d.rounded_rectangle(
            [40, bar_y, 40 + max(total_w, 3), bar_y + bar_h], 6, outline=accent, width=2
        )
        d.text(
            (40 + track_w + 20, bar_y + 8),
            human(g["total_mib"]),
            fill=INK,
            font=font(26, mono=True),
        )

    # The headline is authored in the spec, never computed: a ratio across two
    # differently-configured builds would read as a like-for-like claim it is not.
    d.text((40, H - 116), spec["headline"], fill=ACCENTS["green"], font=font(22))
    for i, note in enumerate(spec.get("footnotes", [])[:3]):
        d.text((40, H - 78 + i * 21), note, fill=DIM, font=font(14, False))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(HERE / "footprint_gb10.json"))
    ap.add_argument("--layout", default="cols", choices=sorted(LAYOUTS))
    ap.add_argument("--out", default=str(ROOT / "benchmarks/media/footprint.png"))
    a = ap.parse_args()
    render(json.loads(Path(a.spec).read_text()), a.layout, a.out)


if __name__ == "__main__":
    main()
