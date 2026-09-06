"""Cutout-to-Gaussian matching helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .catalog import select_catalog_region, validate_gaussian_coordinates
from .config import MATCHING_DEFAULTS
from .segmentation import labels_at_points
from .utils import infer_pixel_scale_arcsec, resolve_pixel_scale_arcsec


def match_gaussians_to_cutout(
    gaussians: pd.DataFrame,
    cutout: Any,
    segmentation: Any | None = None,
    pixel_scale_arcsec: float | None = None,
    preselect_margin_arcsec: float | None = None,
    strict_metadata: bool = False,
    config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Find Gaussian components that fall inside a cutout.

    Returns a normalized component DataFrame and the matching mode name.
    """

    # Prefer sky-coordinate matching and require explicit pixels without WCS.
    if preselect_margin_arcsec is None:
        preselect_margin_arcsec = float((config or {}).get("matching", MATCHING_DEFAULTS).get("preselect_margin_arcsec", MATCHING_DEFAULTS["preselect_margin_arcsec"]))
    if not np.isfinite(preselect_margin_arcsec) or preselect_margin_arcsec < 0:
        raise ValueError("preselect_margin_arcsec must be finite and non-negative")
    height, width = cutout.image.shape
    if pixel_scale_arcsec is None:
        pixel_scale = infer_pixel_scale_arcsec(
            cutout.wcs,
            default=(config or {}).get("pixel_scale_arcsec"),
            strict=strict_metadata,
        )
    else:
        pixel_scale = resolve_pixel_scale_arcsec(pixel_scale_arcsec, config, context="matching pixel scale")
    radius_arcsec = 0.5 * np.hypot(width, height) * pixel_scale + preselect_margin_arcsec

    source = gaussians.copy()
    source["_matching_row_position"] = np.arange(len(source), dtype=int)
    sky_valid, pixel_valid = validate_gaussian_coordinates(source, context="Gaussian matching input")
    if cutout.wcs is None:
        validate_gaussian_coordinates(
            source,
            context="Gaussian matching input for a cutout without WCS",
            require_pixel=True,
        )

    has_sky_coordinates = cutout.wcs is not None and bool(sky_valid.any())
    has_pixel_coordinates = bool(pixel_valid.any())
    if not has_sky_coordinates and not has_pixel_coordinates:
        raise ValueError(
            "Gaussian matching requires a cutout WCS with finite RA/DEC or "
            "finite Gaussian pixel x/y coordinates"
        )

    matched_frames: list[pd.DataFrame] = []
    matching_modes: list[str] = []
    n_outside = 0
    if has_sky_coordinates:
        sky_rows = source.loc[sky_valid]
        candidates = select_catalog_region(sky_rows, cutout.ra, cutout.dec, radius_arcsec)
        if not candidates.empty:
            x, y = cutout.wcs.celestial.world_to_pixel_values(
                candidates["_ra"].to_numpy(float),
                candidates["_dec"].to_numpy(float),
            )
            sky_out = candidates.copy()
            sky_out["x"] = x
            sky_out["y"] = y
            keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            n_outside += int((~keep).sum())
            matched_frames.append(sky_out.loc[keep].copy())
        matching_modes.append("sky")

    # Rows lacking a complete RA/DEC pair are matched through their explicit
    # pixel coordinates.  Complete sky rows are intentionally not duplicated.
    pixel_rows_mask = pixel_valid & ~sky_valid if has_sky_coordinates else pixel_valid
    if pixel_rows_mask.any():
        pixel_out = source.loc[pixel_rows_mask].copy()
        pixel_out["x"] = pixel_out["_x"].to_numpy(float)
        pixel_out["y"] = pixel_out["_y"].to_numpy(float)
        x = pixel_out["x"].to_numpy(float)
        y = pixel_out["y"].to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        n_outside += int((~keep).sum())
        matched_frames.append(pixel_out.loc[keep].copy())
        matching_modes.append("pixel")

    mode = "mixed" if len(matching_modes) > 1 else matching_modes[0]
    if not matched_frames or sum(len(frame) for frame in matched_frames) == 0:
        empty = _empty_components(cutout.cutout_id)
        empty.attrs["n_outside_after_projection"] = n_outside
        return empty, mode

    out = pd.concat(matched_frames, ignore_index=False)
    out = out.sort_values("_matching_row_position", kind="mergesort").drop(columns="_matching_row_position")
    out = out.reset_index(drop=True)
    out.attrs["n_outside_after_projection"] = n_outside
    out["cutout_id"] = cutout.cutout_id
    out["cutout_index"] = cutout.index
    out["component_index"] = np.arange(len(out), dtype=int)
    out["pixel_scale_arcsec"] = pixel_scale

    if segmentation is not None:
        # Attach threshold labels for direct use during pair scoring.
        labels = labels_at_points(
            segmentation.labels_by_threshold,
            segmentation.thresholds,
            out["x"].to_numpy(float),
            out["y"].to_numpy(float),
        )
        for key, values in labels.items():
            out[key] = values

    return out, mode


def _empty_components(cutout_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cutout_id",
            "cutout_index",
            "component_index",
            "_source_id",
            "_island_id",
            "_gaussian_id",
            "_ra",
            "_dec",
            "_total_flux",
            "_peak_flux",
            "_maj",
            "_min",
            "_pa",
            "x",
            "y",
            "pixel_scale_arcsec",
        ]
    ).assign(cutout_id=cutout_id)
