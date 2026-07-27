#!/usr/bin/env python3
r"""concurrency_race - render the vllm.cpp vs vLLM concurrency sweep.

The honest-timing rule (inherited from locate-anything.cpp's image_race.py): the
numbers drawn on screen are the REAL measured binding numbers from
docs/BENCHMARKS.md. Nothing here simulates an engine or invents a datapoint; the
animation only paces how those measured values are revealed, and --dilate sets
playback speed alone.

Why a SWEEP and not a finish-line race: at a single concurrency our lead is
1.007x to 1.045x, which is real but invisible as a "who crosses first" clip. The
story the data actually tells is that we are ahead at EVERY concurrency while
using less host memory and no Python stack, so the sweep is the honest visual.

  python3 concurrency_race.py                    # 16:9 hero
  python3 concurrency_race.py --layout square    # 1:1 for social
  python3 concurrency_race.py --still            # final frame only (PNG)

Spec JSON (see qwen36_27b_c1_c32.json):
  {
    "title": "...", "model": "...", "hardware": "...", "workload": "...",
    "ours":  {"label": "vllm.cpp", "accent": "teal"},
    "theirs":{"label": "vLLM 0.25.0", "accent": "slate"},
    "points": [{"c": 1, "ours": 86.05, "theirs": 82.32}, ...],
    "footnotes": ["peak host memory 24.88 GiB vs 28.18 GiB", ...],
    "links": ["localai.io", "github.com/mudler/LocalAI", "github.com/mudler/vllm.cpp"]
  }
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

BG = (13, 17, 23)
INK = (215, 221, 229)
DIM = (110, 118, 129)
GOLD = (240, 200, 90)
PANEL = (22, 27, 34)
ACCENTS = {
    "teal": (62, 200, 224),
    "slate": (148, 163, 178),
    "green": (126, 231, 135),
    "amber": (255, 207, 86),
}
LAYOUTS = {"cols": (1280, 720), "square": (1080, 1080), "vertical": (1080, 1920)}
FPS = 25


def font(sz, bold=True, mono=False):
    fam = "DejaVuSansMono" if mono else "DejaVuSans"
    suffix = "-Bold" if bold else ""
    try:
        return ImageFont.truetype(
            f"/usr/share/fonts/truetype/dejavu/{fam}{suffix}.ttf", sz
        )
    except OSError:
        return ImageFont.load_default()


def ease(t):
    """Ease-out cubic, so a bar settles instead of snapping."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    for side in ("ours", "theirs"):
        acc = spec[side].get("accent", "teal")
        spec[side]["_c"] = ACCENTS.get(acc, ACCENTS["teal"])
    return spec


def draw_header(cv, W, spec):
    d = ImageDraw.Draw(cv)
    d.text((40, 30), spec["title"], fill=INK, font=font(30))
    sub = f"{spec['model']}  ·  {spec['hardware']}"
    d.text((40, 70), sub, fill=DIM, font=font(17, False))
    d.text((40, 92), spec["workload"], fill=DIM, font=font(15, False))


def draw_row(cv, x, y, w, point, spec, reveal, peak):
    """One concurrency level: two bars, measured tok/s, and the ratio badge."""
    d = ImageDraw.Draw(cv)
    bar_h = 16
    gap = 6
    label_w = 62
    num_w = 130          # "1095.01 tok/s" at 15px mono
    badge_w = 84         # "1.045x" pill
    val_w = num_w + badge_w + 24

    d.text((x, y + 4), f"c{point['c']}", fill=INK, font=font(19, mono=True))

    track_x = x + label_w
    track_w = w - label_w - val_w
    for i, side in enumerate(("ours", "theirs")):
        val = point[side]
        accent = spec[side]["_c"]
        by = y + i * (bar_h + gap)
        d.rounded_rectangle([track_x, by, track_x + track_w, by + bar_h], 4, fill=PANEL)
        shown = val * reveal
        bw = int(track_w * (shown / peak)) if peak else 0
        if bw > 3:
            d.rounded_rectangle([track_x, by, track_x + bw, by + bar_h], 4, fill=accent)
        d.text(
            (track_x + track_w + 14, by - 1),
            f"{shown:7.2f} tok/s",
            fill=INK if side == "ours" else DIM,
            font=font(15, mono=True),
        )

    if reveal >= 0.999:
        ratio = point["ours"] / point["theirs"]
        bx = track_x + track_w + 14 + num_w + 14
        badge = f"{ratio:.3f}x"
        fb = font(15, mono=True)
        tw = d.textlength(badge, font=fb)
        by0 = y + (2 * bar_h + gap) // 2 - 14
        d.rounded_rectangle([bx, by0, bx + tw + 14, by0 + 28], 6, fill=(24, 48, 44))
        d.text((bx + 7, by0 + 5), badge, fill=ACCENTS["green"], font=fb)


def draw_legend(cv, W, y, spec):
    d = ImageDraw.Draw(cv)
    x = 40
    for side in ("ours", "theirs"):
        c = spec[side]["_c"]
        d.rounded_rectangle([x, y + 3, x + 26, y + 15], 3, fill=c)
        lab = spec[side]["label"]
        d.text((x + 34, y), lab, fill=INK if side == "ours" else DIM, font=font(17))
        x += 34 + int(d.textlength(lab, font=font(17))) + 34


def draw_footer(cv, W, H, spec, show):
    if not show:
        return
    d = ImageDraw.Draw(cv)
    y = H - 78
    for i, note in enumerate(spec.get("footnotes", [])[:2]):
        d.text((40, y + i * 22), note, fill=DIM, font=font(15, False))


def compose(spec, W, H, t, total_reveal):
    """One frame at animation time t (0..1 over the reveal phase)."""
    cv = Image.new("RGB", (W, H), BG)
    draw_header(cv, W, spec)

    points = spec["points"]
    peak = max(max(p["ours"], p["theirs"]) for p in points)
    n = len(points)

    top = 140
    row_h = (H - top - 130) // n
    per = 1.0 / n
    for i, p in enumerate(points):
        # Each row reveals in sequence, so the sweep reads left-to-right in time.
        local = (t - i * per) / per
        draw_row(cv, 40, top + i * row_h, W - 80, p, spec, ease(local), peak)

    draw_legend(cv, W, H - 108, spec)
    draw_footer(cv, W, H, spec, t >= 0.999)
    return cv


def end_card(spec, W, H):
    """The LocalAI CTA card the house style requires on every demo."""
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    logo_p = ROOT / "assets/localai_logo.png"
    y = H // 2 - 210
    if logo_p.exists():
        logo = Image.open(logo_p).convert("RGBA")
        lw = min(300, W // 3)
        logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
        cv.paste(logo, ((W - logo.width) // 2, y), logo)
        y += logo.height + 26

    def center(text, fnt, fill, dy):
        tw = d.textlength(text, font=fnt)
        d.text(((W - tw) // 2, dy), text, fill=fill, font=fnt)
        return dy

    center("from the LocalAI team", font(20, False), DIM, y)
    y += 44
    center(spec["headline"], font(34), ACCENTS["teal"], y)
    y += 48
    center(spec["subline"], font(18, False), INK, y)
    y += 54
    for link in spec.get("links", []):
        center(link, font(17, mono=True), DIM, y)
        y += 26
    return cv


def render(spec, layout, out, dilate, still):
    W, H = LAYOUTS[layout]
    if still:
        compose(spec, W, H, 1.0, 1.0).save(out)
        print(f"wrote {out}")
        return

    reveal_s = 6.0 * dilate
    hold_s = 2.5
    card_s = 3.0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx = 0
        n_reveal = int(reveal_s * FPS)
        for f in range(n_reveal):
            compose(spec, W, H, f / max(1, n_reveal - 1), 1.0).save(
                td / f"f{idx:05d}.png"
            )
            idx += 1
        final = compose(spec, W, H, 1.0, 1.0)
        for _ in range(int(hold_s * FPS)):
            final.save(td / f"f{idx:05d}.png")
            idx += 1
        card = end_card(spec, W, H)
        for _ in range(int(card_s * FPS)):
            card.save(td / f"f{idx:05d}.png")
            idx += 1

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        mp4 = out.with_suffix(".mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", str(td / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(mp4)],
            check=True,
        )
        pal = td / "pal.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
             "-vf", "fps=12,scale=900:-1:flags=lanczos,palettegen", str(pal)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(pal),
             "-lavfi", "fps=12,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse",
             str(out.with_suffix(".gif"))],
            check=True,
        )
        final.save(out.with_suffix(".png"))
        print(f"wrote {mp4}, {out.with_suffix('.gif')}, {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(HERE / "qwen36_27b_c1_c32.json"))
    ap.add_argument("--layout", default="cols", choices=sorted(LAYOUTS))
    ap.add_argument("--out", default=str(ROOT / "benchmarks/media/concurrency_race"))
    ap.add_argument("--dilate", type=float, default=1.0,
                    help="playback pacing only; never changes a drawn number")
    ap.add_argument("--still", action="store_true", help="write the final frame only")
    a = ap.parse_args()
    render(load_spec(a.spec), a.layout, a.out, a.dilate, a.still)


if __name__ == "__main__":
    main()
