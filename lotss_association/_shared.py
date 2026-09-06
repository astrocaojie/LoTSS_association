"""Shared frame and pixel-plane helpers for the association modules.

These helpers were previously duplicated across the association, measurement,
local-sanity, graph-merge, and parent-link modules.  They live here once so
that identical quantities (bounding boxes, LAS, sampled S/N lines, table
schemas) cannot drift between stages.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import POSITION_ANGLE_ALIGNMENT_SCALE_DEG, POSITION_ANGLE_MODULUS_DEG

# Support-mask sampling is capped identically in every module so the same mask
# always yields the same LAS.
SUPPORT_LAS_MAX_SAMPLES = 1000


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy with every named column present, keeping extra columns."""

    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = pd.Series(dtype=object)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy projected onto exactly the named columns in order."""

    out = ensure_columns(df, columns)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out[columns]


def line_samples(x1: float, y1: float, x2: float, y2: float, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    distance = float(np.hypot(x2 - x1, y2 - y1))
    n_samples = n or max(3, int(np.ceil(distance)) + 1)
    return np.linspace(x1, x2, n_samples), np.linspace(y1, y2, n_samples)


def sample_image_nearest(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape
    xi = np.rint(xs).astype(int)
    yi = np.rint(ys).astype(int)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    values = np.full(len(xs), np.nan, dtype=float)
    values[valid] = np.asarray(image, dtype=float)[yi[valid], xi[valid]]
    return values


def bbox_from_points(x: np.ndarray, y: np.ndarray, padding: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    if len(x) == 0:
        return 0, 0, max(0, width - 1), max(0, height - 1)
    return (
        int(max(0, np.floor(np.nanmin(x) - padding))),
        int(max(0, np.floor(np.nanmin(y) - padding))),
        int(min(width - 1, np.ceil(np.nanmax(x) + padding))),
        int(min(height - 1, np.ceil(np.nanmax(y) + padding))),
    )


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def las_from_points(x: np.ndarray, y: np.ndarray, pixel_scale_arcsec: float) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0
    coords = np.column_stack([x, y])
    diff = coords[:, None, :] - coords[None, :, :]
    las_pix = float(np.sqrt(np.sum(diff * diff, axis=-1)).max())
    return las_pix, las_pix * pixel_scale_arcsec


def support_las(mask: np.ndarray, pixel_scale_arcsec: float, current_las_pix: float) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) <= 1:
        return current_las_pix, current_las_pix * pixel_scale_arcsec
    sample = np.column_stack([xs, ys])
    if len(sample) > SUPPORT_LAS_MAX_SAMPLES:
        idx = np.linspace(0, len(sample) - 1, SUPPORT_LAS_MAX_SAMPLES).astype(int)
        sample = sample[idx]
    diff = sample[:, None, :] - sample[None, :, :]
    support_las = float(np.sqrt(np.sum(diff * diff, axis=-1)).max())
    las_pix = max(current_las_pix, support_las)
    return las_pix, las_pix * pixel_scale_arcsec


def second_moments(
    image: np.ndarray,
    mask: np.ndarray,
    fallback_x: np.ndarray | None = None,
    fallback_y: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(major, minor, PA, axis_ratio)`` from intensity-weighted moments.

    Masks with fewer than three pixels fall back to unweighted component
    positions when at least two are supplied, and otherwise report NaN axes.
    """

    ys, xs = np.where(mask)
    if len(xs) < 3:
        if fallback_x is None or len(fallback_x) < 2:
            return float("nan"), float("nan"), float("nan"), 1.0
        xs = fallback_x
        ys = fallback_y
        weights = np.ones_like(xs, dtype=float)
    else:
        weights = np.asarray(image[ys, xs], dtype=float)
        weights = weights - np.nanmin(weights)
        weights[~np.isfinite(weights)] = 0.0
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
    x0 = float(np.average(xs, weights=weights))
    y0 = float(np.average(ys, weights=weights))
    dx = xs - x0
    dy = ys - y0
    cov = np.array(
        [
            [float(np.average(dx * dx, weights=weights)), float(np.average(dx * dy, weights=weights))],
            [float(np.average(dx * dy, weights=weights)), float(np.average(dy * dy, weights=weights))],
        ]
    )
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    major = float(np.sqrt(max(vals[0], 0.0)))
    minor = float(np.sqrt(max(vals[1], 0.0)))
    pa = float((np.rad2deg(np.arctan2(vec[1], vec[0])) + 180.0) % 180.0)
    axis_ratio = float(major / max(minor, 1e-6)) if major > 0 else 1.0
    return major, minor, pa, axis_ratio


def angle_delta_deg(a: float, b: float) -> float:
    """Smallest separation between two undirected angles; 90 when invalid."""

    if not np.isfinite(a) or not np.isfinite(b):
        return 90.0
    half_modulus = POSITION_ANGLE_MODULUS_DEG / 2.0
    return float(abs((a - b + half_modulus) % POSITION_ANGLE_MODULUS_DEG - half_modulus))


def alignment_score(delta_deg: float, scale_deg: float = POSITION_ANGLE_ALIGNMENT_SCALE_DEG) -> float:
    return float(np.clip(1.0 - delta_deg / max(scale_deg, 1e-6), 0.0, 1.0))


def balance_ratio_score(ratio: float, max_ratio: float, missing_score: float = 0.0) -> float:
    """Score a flux/size balance ratio; missing ratios receive ``missing_score``."""

    if not np.isfinite(ratio):
        return float(missing_score)
    if ratio >= max_ratio:
        return 0.0
    return float(np.clip(1.0 - np.log(ratio) / max(np.log(max_ratio), 1e-6), 0.0, 1.0))


def beam_area_pix(pixel_scale: float, beam_arcsec: float) -> float:
    """Return the Gaussian beam area in pixels for a circular beam scale."""

    return float(np.pi * (beam_arcsec / max(pixel_scale, 1e-6)) ** 2 / (4.0 * np.log(2.0)))


def with_columns(df: Any, columns: list[str], *, project: bool = False) -> pd.DataFrame:
    """Backward-compatible wrapper dispatching to :func:`ensure_columns`/:func:`select_columns`."""

    return select_columns(df, columns) if project else ensure_columns(df, columns)
