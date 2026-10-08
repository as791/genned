"""Finds the picture inside a phone screenshot. Python mirror of the app's
ScreenshotCrop.kt (app/src/main/kotlin/com/genned/app/data/image/); keep the two in sync.

A shared screenshot of a social app holds a photo between flat UI: status bar, app bar,
captions, buttons, navigation. Classifying the whole screenshot mixes that UI into the
model input. The picture is the tallest band of rows that are *not* mostly the app's flat
background colour; within that band, the widest run of non-background columns.

The work happens on a small grid (GRID_WIDTH wide) so it is cheap on a phone.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

GRID_WIDTH = 108
BACKGROUND_TOLERANCE = 24      # max per-channel distance to count as background
UI_FRACTION = 0.6              # a row/column is UI when this share of it is background
GAP_FRACTION = 0.02            # bridge non-picture gaps up to 2% of the side
MIN_BAND_FRACTION = 0.25       # the picture must be at least 25% of the side
MIN_UI_FRACTION = 0.15         # at least 15% of the rows must be UI, or it isn't a UI screenshot
FALLBACK_TOP, FALLBACK_BOTTOM = 0.04, 0.05  # status/nav bar share of height for the fallback


def _round(x: float) -> int:
    """Half-up rounding, as Kotlin's roundToInt (Python's round() rounds half to even)."""
    return math.floor(x + 0.5)


def _longest_run(is_content: np.ndarray, max_gap: int) -> tuple[int, int]:
    """[start, end) of the longest run of True, bridging False gaps of up to max_gap."""
    best, start, last_true = (0, 0), None, None
    for i, value in enumerate(is_content):
        if value:
            if start is None or i - last_true - 1 > max_gap:
                start = i
            last_true = i
            if last_true + 1 - start > best[1] - best[0]:
                best = (start, last_true + 1)
    return best


def _background(grid: np.ndarray) -> np.ndarray:
    """Mean colour of the most common 4-bit-per-channel colour bucket."""
    q = grid >> 4
    keys = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    values, counts = np.unique(keys, return_counts=True)
    mode = values[np.argmax(counts)]
    return grid[keys == mode].sum(axis=0) // int((keys == mode).sum())  # integer mean, as Kotlin


def picture_band(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) of the picture in grid cells, or None if not confident."""
    gh, gw = grid.shape[:2]
    near = np.abs(grid - _background(grid)).max(axis=2) <= BACKGROUND_TOLERANCE
    ui_rows = near.mean(axis=1) >= UI_FRACTION
    top, bottom = _longest_run(~ui_rows, max(1, _round(GAP_FRACTION * gh)))
    band = bottom - top
    if band < MIN_BAND_FRACTION * gh or ui_rows.sum() < MIN_UI_FRACTION * gh or band >= gh:
        return None
    ui_cols = near[top:bottom].mean(axis=0) >= UI_FRACTION
    left, right = _longest_run(~ui_cols, max(1, _round(GAP_FRACTION * gw)))
    if right - left < MIN_BAND_FRACTION * gw:
        left, right = 0, gw
    return left, top, right, bottom


def content_square(width: int, height: int, top_inset: int, bottom_inset: int) -> tuple[int, int, int]:
    """Mirror of CaptureCrop.contentSquare: (left, top, size)."""
    top = min(max(top_inset, 0), height - 1)
    bottom = min(max(bottom_inset, 0), height - 1 - top)
    content_h = height - top - bottom
    size = min(width, content_h)
    return (width - size) // 2, top + (content_h - size) // 2, size


def classifier_region(image: Image.Image) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the part of a screenshot the app classifies."""
    w, h = image.size
    gw = min(GRID_WIDTH, w)
    gh = max(1, _round(h * gw / w))
    grid = np.asarray(image.convert("RGB").resize((gw, gh), Image.BILINEAR), dtype=np.int32)
    found = picture_band(grid)
    if found is None:
        left, top, size = content_square(w, h, _round(FALLBACK_TOP * h), _round(FALLBACK_BOTTOM * h))
        return left, top, left + size, top + size
    gl, gt, gr, gb = found
    return (gl * w // gw, gt * h // gh, min(w, -(-gr * w // gw)), min(h, -(-gb * h // gh)))
