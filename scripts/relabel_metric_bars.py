"""Overlay more explicit titles on existing outputs/eval/metric_bars_*.png.

The per-image metric values are baked into the existing PNGs and are not
persisted anywhere else, so this script rewrites only the text (suptitle and
subplot titles) rather than recomputing bars. Run from repo root:

    uv run python scripts/relabel_metric_bars.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

OUT_DIR = Path("outputs/eval")


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def _paint_text(
    img: Image.Image,
    text: str,
    band: tuple[int, int, int, int],
    *,
    font: ImageFont.FreeTypeFont,
    fill: str = "black",
) -> None:
    """White-out a band (l, t, r, b) and centre `text` inside it."""
    draw = ImageDraw.Draw(img)
    draw.rectangle(band, fill="white")
    left, top, right, bottom = band
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (left + right) // 2 - tw // 2 - bbox[0]
    cy = (top + bottom) // 2 - th // 2 - bbox[1]
    draw.text((cx, cy), text, font=font, fill=fill)


def _relabel_grid(
    path: Path,
    suptitle: str,
    subtitles: list[str],
    *,
    suptitle_band: tuple[int, int],
    subtitle_band: tuple[int, int],
    suptitle_size: int = 15,
    subtitle_size: int = 13,
) -> None:
    """Overlay new text on a horizontal grid of subplots.

    `suptitle_band`, `subtitle_band` are (top, bottom) y-pixel bands.
    """
    img = Image.open(path).convert("RGB")
    w, _ = img.size
    n = len(subtitles)
    col_w = w / n

    _paint_text(
        img,
        suptitle,
        (0, suptitle_band[0], w, suptitle_band[1]),
        font=_font(suptitle_size),
    )
    sub_font = _font(subtitle_size)
    for i, title in enumerate(subtitles):
        left = int(round(i * col_w))
        right = int(round((i + 1) * col_w))
        _paint_text(
            img,
            title,
            (left, subtitle_band[0], right, subtitle_band[1]),
            font=sub_font,
        )

    img.save(path)
    print(f"wrote {path}")


def main() -> None:
    _relabel_grid(
        OUT_DIR / "metric_bars_depth.png",
        suptitle="Depth metrics per image: DA3 vs SSM-3D  (ETH3D terrains, median-aligned, 42 views)",
        subtitles=[
            "|relative_depth_error| = mean(|d̂−d|/d)   ↓ lower is better",
            "δ<1.25 = frac{max(d̂/d, d/d̂) < 1.25}   ↑ higher is better",
            "rmse = √mean(d̂−d)²  [m]   ↓ lower is better",
            "log10 = mean|log₁₀d̂ − log₁₀d|   ↓ lower is better",
        ],
        suptitle_band=(0, 34),
        subtitle_band=(36, 100),
    )

    _relabel_grid(
        OUT_DIR / "metric_bars_repr.png",
        suptitle="Representation metrics per image: DA3 vs SSM-3D  (ETH3D terrains, 42 views)",
        subtitles=[
            "feat_cos_mean  (mean pairwise token cosine, ↓ = less collapse)",
            "effective_rank  (exp(entropy of SVD spectrum), ↑ richer)",
            "cross_view_nn_agreement  (GT-warped NN match frac, ↑ better)",
        ],
        suptitle_band=(0, 34),
        subtitle_band=(36, 100),
    )

    _relabel_grid(
        OUT_DIR / "metric_bars_memory.png",
        suptitle="Memory footprint: DA3 vs SSM-3D  (backbone-only params; peak deltas per inference call)",
        subtitles=[
            "Parameters (backbone only)",
            "Peak host RSS delta (one fwd)",
            "Peak CUDA memory (one fwd)",
        ],
        suptitle_band=(0, 34),
        subtitle_band=(36, 100),
    )


if __name__ == "__main__":
    main()
