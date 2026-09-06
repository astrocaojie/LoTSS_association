"""Beam-aware Gaussian component association for radio structures."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ._shared import (
    alignment_score as _alignment_score,
    bbox_from_mask as _bbox_from_mask,
    bbox_from_points as _bbox_from_points,
    ensure_columns as _with_columns,
    las_from_points as _las_from_points,
    line_samples as _line_samples,
    sample_image_nearest as _sample_image_nearest,
    second_moments as _second_moments,
    support_las as _support_las,
)
from .beam import (
    angle_delta_180,
    beam_area_arcsec2,
    beam_axes_from_config,
    beam_axis_components,
    beam_covariance_from_config,
    direction_angle_from_delta,
    elliptical_beam_distance,
    projected_beam_distance_from_delta,
    projected_beam_fwhm,
)
from .config import (
    ARTIFACT_DEEP_VALLEY_OFFSET,
    ARTIFACT_DEEP_VALLEY_SPAN,
    ARTIFACT_FLAG_SCORE_MIN,
    ARTIFACT_LARGE_LABEL_AREA_BEAMS,
    ARTIFACT_LARGE_LABEL_DISTANCE_SCALE,
    ARTIFACT_LARGE_LABEL_FRACTION,
    ARTIFACT_LARGE_LABEL_SPAN,
    ARTIFACT_NEGATIVE_SNR_FLOOR,
    ARTIFACT_SIDEBAND_BRIDGE_SCORE_MAX,
    ARTIFACT_SIDEBAND_DEEP_VALLEY_MIN,
    ARTIFACT_SIDEBAND_DISTANCE_BEAM,
    ARTIFACT_SIDEBAND_FLUX_RATIO,
    ARTIFACT_SIDEBAND_NEGATIVE_BOWL_MIN,
    ARTIFACT_SIDEBAND_PA_SCORE_MAX,
    ARTIFACT_SIDEBAND_RIDGE_SCORE_MAX,
    ARTIFACT_SIDEBAND_SCORE_CLOSE,
    ARTIFACT_SIDEBAND_SCORE_NEAR,
    ARTIFACT_TOO_FAR_FLAG_MIN,
    ASSOCIATION_DEFAULTS,
    ASSOCIATION_GROUP_SUPPORT_MIN_PADDING_PIX,
    ASSOCIATION_GROUP_SUPPORT_PADDING_BEAM,
    ASSOCIATION_HIGH_QUALITY_STRONG_EDGE_FRACTION,
    ASSOCIATION_INDEPENDENT_SUPPORT_LENGTH_FACTOR,
    ASSOCIATION_INDEPENDENT_SUPPORT_WIDTH_FACTOR,
    ASSOCIATION_MULTI_GAUSSIAN_EXTENDED_MIN_LAS_BEAM,
    ASSOCIATION_ONLY_2SIGMA_FLAG_FRACTION,
    ASSOCIATION_PA_SUPPORT_SCORE_MIN,
    ASSOCIATION_RESIDUAL_BRIDGE_SCORE_FACTOR,
    ASSOCIATION_SEVERE_SCORE_SPREAD,
    ASSOCIATION_SEVERE_SUPPORT_FRACTION,
    ASSOCIATION_STRONG_SUPPORT_SCORE_MIN,
    ASSOCIATION_SUPPORT_SCORE_MIN,
    ASSOCIATION_TOO_FAR_DISTANCE_FRACTION,
    ASSOCIATION_TOO_FAR_PENALTY_SPAN,
    ASSOCIATION_TOO_FAR_SUPPORT_FRACTION,
    ASSOCIATION_TYPE_DEFAULTS,
    ASSOCIATION_WEIGHT_DEFAULTS,
    BRIDGE_LENGTH_SHORT_SCORE,
    BRIDGE_SCORE_WEIGHTS,
    FORMAL_SNR_LEVELS,
    RESIDUAL_BRIDGE_MIN_WIDTH_BEAM,
    RESIDUAL_BRIDGE_NEGATIVE_MEAN_FRACTION,
    RESIDUAL_BRIDGE_PEAK_SCORE_SPAN,
    RESIDUAL_BRIDGE_SCORE_WEIGHTS,
    RIDGE_SCORE_WEIGHTS,
    SCIENTIFIC_DEFAULTS,
)
from .feature_flags import feature_enabled
from .morphology import (
    add_morphology_columns,
    classify_gaussian_component,
    effective_component_pa_pixel,
    effective_pa_weight,
    intrinsic_axes_from_component,
)
from .segmentation import component_support_mask, threshold_key
from .utils import json_dumps_safe, resolve_pixel_scale_arcsec, safe_float, validate_integer_indices


@dataclass
class AssociationResult:
    """Container returned by the association pipeline."""

    graph: nx.Graph
    edges: pd.DataFrame
    components: pd.DataFrame
    groups: pd.DataFrame
    clusters: list[list[int]]
    local_sanity_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    local_needs_visual_check: pd.DataFrame = field(default_factory=pd.DataFrame)


# Edge fields follow candidate generation, morphology, connectivity, penalties,
# and the final decision in that order.
EDGE_COLUMNS = [
    "cutout_id",
    "gaussian_id_1",
    "gaussian_id_2",
    "component_index_1",
    "component_index_2",
    "pair_separation_arcsec",
    "distance_arcsec",
    "pair_angle_deg",
    "distance_beam",
    "elliptical_beam_distance",
    "directional_beam_arcsec",
    "beam_parallel_separation_arcsec",
    "beam_perpendicular_separation_arcsec",
    "beam_parallel_arcsec",
    "beam_perpendicular_arcsec",
    "pair_angle_relative_to_bpa_deg",
    "pair_angle_relative_bpa_deg",
    "candidate_generation_reason",
    "morphology_class_1",
    "morphology_class_2",
    "resolved_probability_1",
    "resolved_probability_2",
    "beam_like_score_1",
    "beam_like_score_2",
    "classification_reason_1",
    "classification_reason_2",
    "observed_ellipse_overlap_score",
    "intrinsic_ellipse_overlap_score",
    "ellipse_overlap_score",
    "ellipse_gap_beam",
    "pa_weight_1",
    "pa_weight_2",
    "raw_pa_alignment_score",
    "effective_pa_alignment_score",
    "raw_line_to_pa_alignment_score",
    "effective_line_to_pa_alignment_score",
    "pa_alignment_score",
    "line_to_pa_alignment_score",
    "size_similarity_score",
    "flux_ratio",
    "flux_continuity_score",
    "connected_at_3sigma",
    "connected_at_2p5sigma",
    "connected_at_2sigma",
    "only_2sigma_connected",
    "same_label_3sigma",
    "same_label_2p5sigma",
    "same_label_2sigma",
    "common_envelope_area_beam",
    "bridge_mean_snr",
    "bridge_min_snr",
    "bridge_max_snr",
    "bridge_width_pix",
    "bridge_width_beam",
    "bridge_length_pix",
    "bridge_length_beam",
    "bridge_area_pix",
    "bridge_area_beam",
    "bridge_score",
    "residual_bridge_peak_snr",
    "residual_bridge_mean_snr",
    "residual_bridge_integrated_snr",
    "residual_bridge_area_beams",
    "residual_bridge_length_fraction",
    "residual_bridge_width_beams",
    "residual_bridge_contiguous_fraction",
    "multi_threshold_bridge_persistence",
    "residual_bridge_score",
    "ridge_mean_snr",
    "ridge_gap_fraction",
    "ridge_continuity_score",
    "ridge_gradient_smoothness",
    "closeness_score",
    "flow_alignment_score",
    "deep_valley_penalty",
    "only_2sigma_penalty",
    "negative_bowl_penalty",
    "sidelobe_risk_penalty",
    "too_far_penalty",
    "large_mask_swallow_penalty",
    "unresolved_pair_veto",
    "unresolved_pair_veto_reason",
    "association_score",
    "edge_type",
    "association_decision",
    "rejection_reason",
    "artifact_risk_flags",
    "penalties",
    "debug_info",
]

GROUP_COLUMNS = [
    "cutout_id",
    "association_group_id",
    "original_association_group_id",
    "association_group_index",
    "component_ids",
    "n_gaussians",
    "gaussian_ids",
    "ra",
    "dec",
    "centroid_x",
    "centroid_y",
    "bounding_box",
    "LAS_arcsec",
    "LAS_beam",
    "total_flux_gaussian",
    "peak_flux",
    "group_PA",
    "axis_ratio",
    "association_score_mean",
    "association_score_min",
    "association_score_max",
    "n_strong_edges",
    "n_weak_edges",
    "n_only_2sigma_edges",
    "association_quality",
    "association_type",
    "morphology_class",
    "resolved_probability",
    "beam_like_score",
    "classification_reason",
    "artifact_risk_flags",
    "debug_info",
]


def _association_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(ASSOCIATION_DEFAULTS)
    out = dict(defaults)
    out.update(config.get("association", {}) or {})
    out["quality_thresholds"] = {
        **defaults["quality_thresholds"],
        **(config.get("association", {}).get("quality_thresholds", {}) if config.get("association") else {}),
    }
    return out


def _max_pair_distance_arcsec(config: dict[str, Any]) -> float | None:
    """Return the configured absolute pair-search cap in arcsec, if present."""

    value = _association_config(config).get("max_pair_distance_arcsec")
    if value is None:
        return None
    cap = safe_float(value, float("nan"))
    return float(cap) if np.isfinite(cap) and cap > 0 else None


def _association_weights(config: dict[str, Any]) -> dict[str, float]:
    out = dict(ASSOCIATION_WEIGHT_DEFAULTS)
    out.update(config.get("weights_association", {}) or {})
    return {key: float(value) for key, value in out.items()}


def compute_beam_size_arcsec(config: dict[str, Any]) -> float:
    """Return the effective beam size used for distance normalization."""

    beam = config.get("beam", {}) or {}
    major = safe_float(beam.get("major_arcsec"), float("nan"))
    minor = safe_float(beam.get("minor_arcsec"), float("nan"))
    if not np.isfinite(major) or major <= 0 or not np.isfinite(minor) or minor <= 0:
        raise ValueError("beam major/minor must be provided and positive")
    return float(np.sqrt(major * minor))


def _beam_area_arcsec2(config: dict[str, Any]) -> float:
    return beam_area_arcsec2(config=config)


def _thresholds_from_labels(labels_by_threshold: Any, config: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray]:
    if hasattr(labels_by_threshold, "labels_by_threshold"):
        labels = np.asarray(labels_by_threshold.labels_by_threshold)
        thresholds = np.asarray(labels_by_threshold.thresholds, dtype=float)
        return labels, thresholds
    if labels_by_threshold is None:
        return None, np.asarray(config.get("snr_thresholds", SCIENTIFIC_DEFAULTS["snr_thresholds"]), dtype=float)
    return np.asarray(labels_by_threshold), np.asarray(config.get("snr_thresholds", SCIENTIFIC_DEFAULTS["snr_thresholds"]), dtype=float)


def _label_from_row(row: pd.Series, threshold: float) -> int:
    col = f"label_at_{threshold_key(threshold)}"
    if col not in row:
        return 0
    try:
        return int(row[col])
    except (TypeError, ValueError, OverflowError):
        return 0


def _label_from_map(row: pd.Series, labels: np.ndarray | None, thresholds: np.ndarray, threshold: float) -> int:
    if labels is None:
        return 0
    idx = int(np.argmin(np.abs(thresholds - float(threshold))))
    label_map = labels[idx]
    height, width = label_map.shape
    x = safe_float(row.get("x"))
    y = safe_float(row.get("y"))
    if not np.isfinite(x) or not np.isfinite(y):
        return 0
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or xi >= width or yi < 0 or yi >= height:
        return 0
    return int(label_map[yi, xi])


def _shared_label(
    row_i: pd.Series,
    row_j: pd.Series,
    labels: np.ndarray | None,
    thresholds: np.ndarray,
    threshold: float,
) -> int:
    left = _label_from_row(row_i, threshold) or _label_from_map(row_i, labels, thresholds, threshold)
    right = _label_from_row(row_j, threshold) or _label_from_map(row_j, labels, thresholds, threshold)
    return int(left) if left > 0 and left == right else 0


_LABEL_COUNT_CACHE: dict[int, tuple[weakref.ReferenceType[np.ndarray], dict[int, np.ndarray]]] = {}


def _label_count_cache(labels: np.ndarray | None, thresholds: np.ndarray, config: dict[str, Any] | None = None) -> dict[int, np.ndarray]:
    """Cache per-threshold label-pixel counts for repeated edge scoring."""

    if labels is None:
        return {}
    key = id(labels)
    cached_entry = _LABEL_COUNT_CACHE.get(key)
    if cached_entry is not None and cached_entry[0]() is labels:
        return cached_entry[1]
    out: dict[int, np.ndarray] = {}
    for idx in range(len(thresholds)):
        label_map = np.asarray(labels[idx])
        out[int(idx)] = np.bincount(label_map.ravel())
    def _cleanup(_ref: weakref.ReferenceType[np.ndarray], cache_key: int = key) -> None:
        _LABEL_COUNT_CACHE.pop(cache_key, None)

    _LABEL_COUNT_CACHE[key] = (weakref.ref(labels, _cleanup), out)
    return out


def _label_area_fraction(
    labels: np.ndarray | None,
    thresholds: np.ndarray,
    threshold: float,
    label_id: int,
    config: dict[str, Any] | None = None,
) -> float:
    if labels is None or label_id <= 0:
        return 0.0
    idx = int(np.argmin(np.abs(thresholds - float(threshold))))
    label_map = labels[idx]
    if config is not None:
        counts = _label_count_cache(labels, thresholds, config).get(idx)
        if counts is not None and label_id < len(counts):
            return float(counts[int(label_id)] / max(label_map.size, 1))
    return float(np.count_nonzero(label_map == label_id) / max(label_map.size, 1))


def _label_area_beam(
    labels: np.ndarray | None,
    thresholds: np.ndarray,
    threshold: float,
    label_id: int,
    pixel_scale_arcsec: float,
    config: dict[str, Any],
) -> float:
    if labels is None or label_id <= 0:
        return 0.0
    idx = int(np.argmin(np.abs(thresholds - float(threshold))))
    label_map = labels[idx]
    counts = _label_count_cache(labels, thresholds, config).get(idx)
    if counts is not None and label_id < len(counts):
        n_pixels = int(counts[int(label_id)])
    else:
        n_pixels = int(np.count_nonzero(label_map == label_id))
    area_arcsec2 = float(n_pixels) * pixel_scale_arcsec * pixel_scale_arcsec
    return area_arcsec2 / max(_beam_area_arcsec2(config), 1e-6)


def _corridor_values(
    snr_map: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    half_width_pix: float,
) -> np.ndarray:
    height, width = snr_map.shape
    pad = int(np.ceil(half_width_pix + 2))
    xmin = max(0, int(np.floor(min(x1, x2))) - pad)
    xmax = min(width - 1, int(np.ceil(max(x1, x2))) + pad)
    ymin = max(0, int(np.floor(min(y1, y2))) - pad)
    ymax = min(height - 1, int(np.ceil(max(y1, y2))) + pad)
    if xmax < xmin or ymax < ymin:
        return np.asarray([], dtype=float)
    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    dx = x2 - x1
    dy = y2 - y1
    length2 = dx * dx + dy * dy
    if length2 <= 0:
        dist = np.hypot(xx - x1, yy - y1)
        mask = dist <= half_width_pix
    else:
        t = ((xx - x1) * dx + (yy - y1) * dy) / length2
        dist = np.abs((xx - x1) * dy - (yy - y1) * dx) / max(np.sqrt(length2), 1e-6)
        mask = (t >= 0.0) & (t <= 1.0) & (dist <= half_width_pix)
    return np.asarray(snr_map[ymin : ymax + 1, xmin : xmax + 1], dtype=float)[mask]


def _component_major_minor(row: pd.Series, config: dict[str, Any]) -> tuple[float, float]:
    beam = compute_beam_size_arcsec(config)
    major = safe_float(row.get("_dc_maj"), safe_float(row.get("_maj"), beam))
    minor = safe_float(row.get("_dc_min"), safe_float(row.get("_min"), beam))
    if not np.isfinite(major) or major <= 0:
        major = safe_float(row.get("_maj"), beam)
    if not np.isfinite(minor) or minor <= 0:
        minor = safe_float(row.get("_min"), beam)
    if not np.isfinite(major) or major <= 0:
        major = beam
    if not np.isfinite(minor) or minor <= 0:
        minor = beam
    return float(major), float(minor)


def _component_classification(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    if "morphology_class" in row and str(row.get("morphology_class", "")).strip():
        return {
            "morphology_class": row.get("morphology_class"),
            "resolved_probability": safe_float(row.get("resolved_probability"), 0.0),
            "resolved_significance": safe_float(row.get("resolved_significance"), 0.0),
            "beam_like_score": safe_float(row.get("beam_like_score"), 0.0),
            "classification_reason": row.get("classification_reason", ""),
        }
    return classify_gaussian_component(row, config)


def _ellipse_overlap_from_axes(
    distance_arcsec: float,
    major_i: float,
    major_j: float,
    projected_beam_arcsec: float,
) -> tuple[float, float]:
    if not np.isfinite(distance_arcsec) or distance_arcsec < 0:
        return 0.0, float("nan")
    if not np.isfinite(major_i) or not np.isfinite(major_j) or major_i <= 0 or major_j <= 0:
        return 0.0, float("nan")
    support_radius = 0.5 * (major_i + major_j)
    gap_arcsec = float(distance_arcsec - support_radius)
    gap_beam = gap_arcsec / max(projected_beam_arcsec, 1e-6)
    return (1.0 if gap_arcsec <= 0 else float(np.clip(1.0 - gap_beam, 0.0, 1.0))), float(gap_beam)


def _intrinsic_overlap_score(
    component_i: pd.Series,
    component_j: pd.Series,
    distance_arcsec: float,
    projected_beam_arcsec: float,
    config: dict[str, Any],
) -> float:
    class_i = str(component_i.get("morphology_class", "")).strip() or str(_component_classification(component_i, config).get("morphology_class", ""))
    class_j = str(component_j.get("morphology_class", "")).strip() or str(_component_classification(component_j, config).get("morphology_class", ""))
    allowed = {"resolved", "marginally_resolved"}
    if class_i not in allowed or class_j not in allowed:
        return 0.0
    maj_i, _min_i, _pa_i = intrinsic_axes_from_component(component_i, config)
    maj_j, _min_j, _pa_j = intrinsic_axes_from_component(component_j, config)
    score, _gap = _ellipse_overlap_from_axes(distance_arcsec, maj_i, maj_j, projected_beam_arcsec)
    return float(score)


def _flux_ratio(row_i: pd.Series, row_j: pd.Series) -> float:
    for flux_field in ("_total_flux", "_peak_flux"):
        f1 = safe_float(row_i.get(flux_field))
        f2 = safe_float(row_j.get(flux_field))
        if np.isfinite(f1) and np.isfinite(f2) and f1 > 0 and f2 > 0:
            return float(np.clip(min(f1, f2) / max(f1, f2), 0.0, 1.0))
    return float("nan")


def compute_bridge_features(
    component_i: pd.Series,
    component_j: pd.Series,
    snr_map: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """Estimate beam-width bridge evidence between two component centers."""

    # Sample a narrow corridor between the Gaussian centres for bridge support.
    assoc = _association_config(config)
    pixel_scale = resolve_pixel_scale_arcsec(component_i.get("pixel_scale_arcsec"), config, context="association residual-bridge pixel scale")
    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    dx = x2 - x1
    dy = y2 - y1
    cov = beam_covariance_from_config(config)
    axis_angle = direction_angle_from_delta(dx, dy)
    axis_beam = projected_beam_fwhm(axis_angle, cov)
    perp_beam = projected_beam_fwhm(axis_angle + 90.0, cov) if np.isfinite(axis_angle) else compute_beam_size_arcsec(config)
    length_pix = float(np.hypot(x2 - x1, y2 - y1))
    length_beam = float(length_pix * pixel_scale / max(axis_beam, 1e-6))
    min_width_beam = float(assoc["min_bridge_width_beam"])
    half_width_pix = max(1.0, 0.5 * min_width_beam * perp_beam / max(pixel_scale, 1e-6))
    weak = float(assoc["threshold_weak"])
    strong_mid = FORMAL_SNR_LEVELS["intermediate"]

    xs, ys = _line_samples(x1, y1, x2, y2)
    line_values = _sample_image_nearest(snr_map, xs, ys)
    corridor = _corridor_values(snr_map, x1, y1, x2, y2, half_width_pix=half_width_pix)
    finite_line = line_values[np.isfinite(line_values)]
    finite_corridor = corridor[np.isfinite(corridor)]
    finite = finite_corridor if finite_corridor.size else finite_line
    if finite.size == 0:
        return {
            "bridge_mean_snr": 0.0,
            "bridge_min_snr": 0.0,
            "bridge_max_snr": 0.0,
            "bridge_width_pix": 0.0,
            "bridge_width_beam": 0.0,
            "bridge_length_pix": length_pix,
            "bridge_length_beam": length_beam,
            "bridge_area_pix": 0.0,
            "bridge_area_beam": 0.0,
            "bridge_score": 0.0,
        }

    bridge_area_pix = float(np.count_nonzero(finite_corridor >= weak)) if finite_corridor.size else float(np.count_nonzero(finite_line >= weak))
    bridge_width_pix = bridge_area_pix / max(length_pix, 1.0)
    bridge_width_beam = bridge_width_pix * pixel_scale / max(perp_beam, 1e-6)
    beam_area_pix = _beam_area_arcsec2(config) / max(pixel_scale * pixel_scale, 1e-6)
    bridge_area_beam = bridge_area_pix / max(beam_area_pix, 1e-6)
    line_support = float(np.mean(finite_line >= weak)) if finite_line.size else 0.0
    mid_support = float(np.mean(finite_line >= strong_mid)) if finite_line.size else 0.0
    width_score = float(np.clip(bridge_width_beam / max(min_width_beam, 1e-6), 0.0, 1.0))
    mean_snr = float(np.nanmean(finite))
    min_snr = float(np.nanmin(finite_line)) if finite_line.size else float(np.nanmin(finite))
    max_snr = float(np.nanmax(finite))
    mean_score = float(np.clip((mean_snr - weak) / max(FORMAL_SNR_LEVELS["strong"] - weak, 1e-6), 0.0, 1.0))
    length_score = 1.0 if length_beam >= float(assoc["min_bridge_length_beam"]) else BRIDGE_LENGTH_SHORT_SCORE
    bridge_values = (line_support, mid_support, width_score, mean_score)
    bridge_score = length_score * sum(
        weight * bridge_values[index] for index, weight in enumerate(BRIDGE_SCORE_WEIGHTS)
    )
    return {
        "bridge_mean_snr": mean_snr,
        "bridge_min_snr": min_snr,
        "bridge_max_snr": max_snr,
        "bridge_width_pix": float(bridge_width_pix),
        "bridge_width_beam": float(bridge_width_beam),
        "bridge_length_pix": length_pix,
        "bridge_length_beam": length_beam,
        "bridge_area_pix": bridge_area_pix,
        "bridge_area_beam": float(bridge_area_beam),
        "bridge_score": float(np.clip(bridge_score, 0.0, 1.5)),
    }


def _beam_model_on_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    amplitude: float,
    pixel_scale_arcsec: float,
    config: dict[str, Any],
) -> np.ndarray:
    cov_arcsec = beam_covariance_from_config(config)
    cov_pix = cov_arcsec / max(pixel_scale_arcsec * pixel_scale_arcsec, 1e-12)
    inv = np.linalg.pinv(cov_pix)
    dx = xx - float(x0)
    dy = yy - float(y0)
    q = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return float(amplitude) * np.exp(-0.5 * q)


def _max_contiguous_fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    best = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best / max(mask.size, 1))


def compute_residual_bridge_features(
    component_i: pd.Series,
    component_j: pd.Series,
    snr_map: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """Measure bridge residuals after subtracting two elliptical beam models."""

    assoc = _association_config(config)
    # Remove approximate endpoint beam models before measuring bridge residuals.
    if not bool(assoc["enable_residual_bridge"]):
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    pixel_scale = resolve_pixel_scale_arcsec(component_i.get("pixel_scale_arcsec"), config, context="association penalty pixel scale")
    if not np.all(np.isfinite([x1, y1, x2, y2, pixel_scale])):
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    dx = x2 - x1
    dy = y2 - y1
    length_pix = float(np.hypot(dx, dy))
    if length_pix <= 0:
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    cov = beam_covariance_from_config(config)
    axis_angle = direction_angle_from_delta(dx, dy)
    axis_beam = projected_beam_fwhm(axis_angle, cov)
    perp_beam = projected_beam_fwhm(axis_angle + 90.0, cov) if np.isfinite(axis_angle) else compute_beam_size_arcsec(config)
    width_beam = max(float(assoc["min_bridge_width_beam"]), RESIDUAL_BRIDGE_MIN_WIDTH_BEAM)
    half_width_pix = max(1.0, 0.5 * width_beam * perp_beam / max(pixel_scale, 1e-6))
    endpoint_exclusion_beam = float(assoc["residual_bridge_endpoint_exclusion_beam"])
    endpoint_exclusion_pix = endpoint_exclusion_beam * axis_beam / max(pixel_scale, 1e-6)

    height, width = snr_map.shape
    pad = int(np.ceil(half_width_pix + 3.0 * max(axis_beam, perp_beam) / max(pixel_scale, 1e-6)))
    xmin = max(0, int(np.floor(min(x1, x2))) - pad)
    xmax = min(width - 1, int(np.ceil(max(x1, x2))) + pad)
    ymin = max(0, int(np.floor(min(y1, y2))) - pad)
    ymax = min(height - 1, int(np.ceil(max(y1, y2))) + pad)
    if xmax < xmin or ymax < ymin:
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    t = ((xx - x1) * dx + (yy - y1) * dy) / max(length_pix * length_pix, 1e-6)
    dist = np.abs((xx - x1) * dy - (yy - y1) * dx) / max(length_pix, 1e-6)
    along_pix = t * length_pix
    corridor_mask = (
        (t >= 0.0)
        & (t <= 1.0)
        & (along_pix >= endpoint_exclusion_pix)
        & (along_pix <= max(length_pix - endpoint_exclusion_pix, endpoint_exclusion_pix))
        & (dist <= half_width_pix)
    )
    if not corridor_mask.any():
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    sub = np.asarray(snr_map[ymin : ymax + 1, xmin : xmax + 1], dtype=float)
    amp1 = safe_float(_sample_image_nearest(snr_map, np.asarray([x1]), np.asarray([y1]))[0], safe_float(component_i.get("_peak_snr"), 0.0))
    amp2 = safe_float(_sample_image_nearest(snr_map, np.asarray([x2]), np.asarray([y2]))[0], safe_float(component_j.get("_peak_snr"), 0.0))
    amp1 = max(amp1, 0.0)
    amp2 = max(amp2, 0.0)
    null = _beam_model_on_grid(xx, yy, x1, y1, amp1, pixel_scale, config) + _beam_model_on_grid(xx, yy, x2, y2, amp2, pixel_scale, config)
    residual = sub - null
    values = residual[corridor_mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        peak = mean = integrated = area_beams = length_fraction = width_beams = contiguous = persistence = score = 0.0
    else:
        threshold = float(assoc["residual_bridge_threshold_snr"])
        positive = finite >= threshold
        peak = float(np.nanmax(finite))
        mean = float(np.nanmean(finite))
        area_pix = float(np.count_nonzero(positive))
        area_beams = area_pix / max(_beam_area_arcsec2(config) / max(pixel_scale * pixel_scale, 1e-6), 1e-6)
        width_beams = (area_pix / max(length_pix, 1.0)) * pixel_scale / max(perp_beam, 1e-6)
        integrated = float(np.nansum(np.clip(finite, 0.0, None)) / np.sqrt(max(area_beams, 1.0)))
        xs, ys = _line_samples(x1, y1, x2, y2)
        line_resid = _sample_image_nearest(
            residual,
            xs - xmin,
            ys - ymin,
        )
        finite_line = line_resid[np.isfinite(line_resid)]
        if finite_line.size:
            trim = int(round(endpoint_exclusion_pix))
            if trim > 0 and finite_line.size > 2 * trim:
                finite_line = finite_line[trim:-trim]
            line_positive = finite_line >= threshold
            length_fraction = float(np.mean(line_positive)) if line_positive.size else 0.0
            contiguous = _max_contiguous_fraction(line_positive)
            persistence = float(np.mean([np.mean(finite_line >= FORMAL_SNR_LEVELS[name]) for name in ("weak", "intermediate", "strong")]))
        else:
            length_fraction = contiguous = persistence = 0.0
        peak_score = np.clip((peak - threshold) / RESIDUAL_BRIDGE_PEAK_SCORE_SPAN, 0.0, 1.0)
        mean_score = np.clip((mean - RESIDUAL_BRIDGE_NEGATIVE_MEAN_FRACTION * threshold) / max(threshold, 1e-6), 0.0, 1.0)
        area_score = np.clip(area_beams / max(float(assoc["residual_bridge_min_area_beams"]), 1e-6), 0.0, 1.0)
        length_score = np.clip(length_fraction / max(float(assoc["residual_bridge_min_length_fraction"]), 1e-6), 0.0, 1.0)
        score = float(
            np.clip(
                    sum(
                        weight * (peak_score, mean_score, area_score, length_score, contiguous)[index]
                        for index, weight in enumerate(RESIDUAL_BRIDGE_SCORE_WEIGHTS)
                    ),
                0.0,
                1.5,
            )
        )

    return {
        "residual_bridge_peak_snr": float(peak),
        "residual_bridge_mean_snr": float(mean),
        "residual_bridge_integrated_snr": float(integrated),
        "residual_bridge_area_beams": float(area_beams),
        "residual_bridge_length_fraction": float(length_fraction),
        "residual_bridge_width_beams": float(width_beams),
        "residual_bridge_contiguous_fraction": float(contiguous),
        "multi_threshold_bridge_persistence": float(persistence),
        "residual_bridge_score": float(score),
    }


def compute_ridge_continuity(
    component_i: pd.Series,
    component_j: pd.Series,
    snr_map: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """Estimate ridge continuity along the center-to-center path."""

    assoc = _association_config(config)
    weak = float(assoc["threshold_weak"])
    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    xs, ys = _line_samples(x1, y1, x2, y2)
    values = _sample_image_nearest(snr_map, xs, ys)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "ridge_mean_snr": 0.0,
            "ridge_gap_fraction": 1.0,
            "ridge_continuity_score": 0.0,
            "ridge_gradient_smoothness": 0.0,
        }
    ridge_mean = float(np.nanmean(finite))
    gap_fraction = float(np.mean(finite < weak))
    weak_support = float(np.mean(finite >= weak))
    mid_support = float(np.mean(finite >= FORMAL_SNR_LEVELS["intermediate"]))
    if finite.size >= 3:
        diffs = np.abs(np.diff(finite))
        smoothness = float(1.0 / (1.0 + np.nanmedian(diffs) / (abs(ridge_mean) + 1.0)))
    else:
        smoothness = 0.5
    ridge_values = (weak_support, mid_support, smoothness)
    continuity = sum(
        weight * ridge_values[index] for index, weight in enumerate(RIDGE_SCORE_WEIGHTS)
    ) * (1.0 - 0.5 * gap_fraction)
    return {
        "ridge_mean_snr": ridge_mean,
        "ridge_gap_fraction": gap_fraction,
        "ridge_continuity_score": float(np.clip(continuity, 0.0, 1.5)),
        "ridge_gradient_smoothness": float(np.clip(smoothness, 0.0, 1.0)),
    }


def compute_artifact_penalties(
    component_i: pd.Series,
    component_j: pd.Series,
    image: np.ndarray,
    snr_map: np.ndarray,
    labels_by_threshold: Any,
    config: dict[str, Any],
    features: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    """Compute penalties for weak-only bridges and likely radio artifacts."""

    del image
    features = dict(features or {})
    labels, thresholds = _thresholds_from_labels(labels_by_threshold, config)
    assoc = _association_config(config)
    pixel_scale = resolve_pixel_scale_arcsec(component_i.get("pixel_scale_arcsec"), config, context="association pair pixel scale")
    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    distance_beam, _projected_beam, _direction = projected_beam_distance_from_delta(x2 - x1, y2 - y1, pixel_scale, config)
    weak = float(assoc["threshold_weak"])
    max_pair_distance_beam = float(assoc["max_pair_distance_beam"])
    xs, ys = _line_samples(x1, y1, x2, y2)
    line_values = _sample_image_nearest(snr_map, xs, ys)
    finite_line = line_values[np.isfinite(line_values)]
    min_line = float(np.nanmin(finite_line)) if finite_line.size else 0.0
    negative_fraction = float(np.mean(finite_line < ARTIFACT_NEGATIVE_SNR_FLOOR)) if finite_line.size else 0.0
    shared_label_2 = int(features.get("same_label_2sigma", 0) or _shared_label(component_i, component_j, labels, thresholds, FORMAL_SNR_LEVELS["weak"]))
    label_fraction = _label_area_fraction(labels, thresholds, FORMAL_SNR_LEVELS["weak"], shared_label_2, config)
    label_area_beam = _label_area_beam(labels, thresholds, FORMAL_SNR_LEVELS["weak"], shared_label_2, pixel_scale, config)
    flux_ratio = float(features.get("flux_ratio", _flux_ratio(component_i, component_j)))
    only_2sigma_connected = bool(features.get("only_2sigma_connected", False))
    bridge_score = float(features.get("bridge_score", 0.0))
    ridge_score = float(features.get("ridge_continuity_score", 0.0))
    pa_score = float(features.get("pa_alignment_score", 0.0))
    line_pa_score = float(features.get("line_to_pa_alignment_score", 0.0))

    deep_valley = float(np.clip((weak - min_line) / max(weak + ARTIFACT_DEEP_VALLEY_OFFSET, 1e-6), 0.0, 1.5))
    only_2sigma = 1.0 if only_2sigma_connected else 0.0
    negative_bowl = float(np.clip((-min_line - ARTIFACT_DEEP_VALLEY_OFFSET) / ARTIFACT_DEEP_VALLEY_SPAN, 0.0, 2.0) + np.clip(negative_fraction, 0.0, 1.0))
    too_far = float(
        np.clip(
            (distance_beam / max(max_pair_distance_beam, 1e-6) - ASSOCIATION_TOO_FAR_SUPPORT_FRACTION)
            / ASSOCIATION_TOO_FAR_PENALTY_SPAN,
            0.0,
            2.0,
        )
    )
    if distance_beam > ASSOCIATION_TOO_FAR_DISTANCE_FRACTION * max_pair_distance_beam:
        support_count = int(bridge_score >= ASSOCIATION_SUPPORT_SCORE_MIN) + int(ridge_score >= ASSOCIATION_SUPPORT_SCORE_MIN) + int(pa_score >= ASSOCIATION_PA_SUPPORT_SCORE_MIN) + int(line_pa_score >= ASSOCIATION_PA_SUPPORT_SCORE_MIN)
        if support_count < 2:
            too_far = max(too_far, 1.0)
    large_mask = 0.0
    if label_fraction > ARTIFACT_LARGE_LABEL_FRACTION:
        large_mask = max(large_mask, float(np.clip((label_fraction - ARTIFACT_LARGE_LABEL_FRACTION) / ARTIFACT_LARGE_LABEL_SPAN, 0.0, 1.5)))
    if label_area_beam > max(ARTIFACT_LARGE_LABEL_AREA_BEAMS, ARTIFACT_LARGE_LABEL_DISTANCE_SCALE * max(distance_beam, 1.0) ** 2) and only_2sigma_connected:
        large_mask = max(large_mask, 1.0)
    sidelobe = 0.0
    if flux_ratio < ARTIFACT_SIDEBAND_FLUX_RATIO and (
        negative_bowl > ARTIFACT_SIDEBAND_NEGATIVE_BOWL_MIN or deep_valley > ARTIFACT_SIDEBAND_DEEP_VALLEY_MIN
    ):
        sidelobe = max(sidelobe, ARTIFACT_SIDEBAND_SCORE_NEAR)
    if (
        distance_beam < ARTIFACT_SIDEBAND_DISTANCE_BEAM
        and bridge_score < ARTIFACT_SIDEBAND_BRIDGE_SCORE_MAX
        and ridge_score < ARTIFACT_SIDEBAND_RIDGE_SCORE_MAX
        and max(pa_score, line_pa_score) < ARTIFACT_SIDEBAND_PA_SCORE_MAX
    ):
        sidelobe = max(sidelobe, ARTIFACT_SIDEBAND_SCORE_CLOSE)

    flags = []
    if only_2sigma_connected:
        flags.append("only_2sigma")
    if negative_bowl >= ARTIFACT_FLAG_SCORE_MIN:
        flags.append("negative_bowl")
    if sidelobe >= ARTIFACT_FLAG_SCORE_MIN:
        flags.append("sidelobe_risk")
    if large_mask >= ARTIFACT_FLAG_SCORE_MIN:
        flags.append("large_mask_swallow")
    if too_far >= ARTIFACT_TOO_FAR_FLAG_MIN:
        flags.append("too_far")

    return {
        "deep_valley_penalty": deep_valley,
        "only_2sigma_penalty": only_2sigma,
        "negative_bowl_penalty": float(np.clip(negative_bowl, 0.0, 2.0)),
        "sidelobe_risk_penalty": float(np.clip(sidelobe, 0.0, 1.5)),
        "too_far_penalty": too_far,
        "large_mask_swallow_penalty": float(np.clip(large_mask, 0.0, 1.5)),
        "artifact_risk_flags": ",".join(flags),
    }


def has_independent_radio_evidence(features: dict[str, Any], config: dict[str, Any]) -> bool:
    """Return True when evidence is not just beam-induced similarity."""

    assoc = _association_config(config)
    use_contour = feature_enabled(config, "use_multithreshold_contour")
    if use_contour and bool(features.get("connected_at_3sigma", False)):
        return True
    common_area = safe_float(features.get("common_envelope_area_beam"), 0.0)
    if use_contour and bool(features.get("connected_at_2p5sigma", False)) and common_area >= float(
        assoc["unresolved_veto_min_common_envelope_area_beam"]
    ):
        return True
    if safe_float(features.get("bridge_score"), 0.0) >= float(assoc["unresolved_veto_min_bridge_score"]):
        width_ok = safe_float(features.get("bridge_width_beam"), 0.0) >= float(assoc["min_bridge_width_beam"]) * ASSOCIATION_INDEPENDENT_SUPPORT_WIDTH_FACTOR
        length_ok = safe_float(features.get("bridge_length_beam"), 0.0) >= float(assoc["min_bridge_length_beam"]) * ASSOCIATION_INDEPENDENT_SUPPORT_LENGTH_FACTOR
        if width_ok and length_ok:
            return True
    if safe_float(features.get("residual_bridge_score"), 0.0) >= float(
        assoc["unresolved_veto_min_residual_bridge_score"]
    ):
        return True
    return False


def unresolved_pair_veto(features: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    """Apply hard veto for two unresolved or strongly beam-like components."""

    assoc = _association_config(config)
    if not bool(assoc["enable_unresolved_pair_veto"]):
        return False, ""
    class_i = str(features.get("morphology_class_1", ""))
    class_j = str(features.get("morphology_class_2", ""))
    beam_like_threshold = float(assoc["unresolved_veto_beam_like_score"])
    beam_like_i = safe_float(features.get("beam_like_score_1"), 0.0) >= beam_like_threshold
    beam_like_j = safe_float(features.get("beam_like_score_2"), 0.0) >= beam_like_threshold
    both_unresolved = (class_i == "unresolved" or beam_like_i) and (class_j == "unresolved" or beam_like_j)
    if both_unresolved and not has_independent_radio_evidence(features, config):
        return True, "veto_unresolved_pair_no_independent_radio_evidence"
    return False, ""


def compute_pair_association_features(
    component_i: pd.Series,
    component_j: pd.Series,
    image: np.ndarray,
    snr_map: np.ndarray,
    labels_by_threshold: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute beam-aware pair features for two PyBDSF Gaussian components."""

    # Retain each evidence term so pair decisions remain inspectable.
    labels, thresholds = _thresholds_from_labels(labels_by_threshold, config)
    pixel_scale = resolve_pixel_scale_arcsec(component_i.get("pixel_scale_arcsec"), config, context="association pair pixel scale")
    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    dx = x2 - x1
    dy = y2 - y1
    distance_pix = float(np.hypot(x2 - x1, y2 - y1))
    distance_arcsec = distance_pix * pixel_scale
    distance_beam, projected_beam, line_angle = projected_beam_distance_from_delta(dx, dy, pixel_scale, config)
    parallel_arcsec, perpendicular_arcsec, angle_to_bpa = beam_axis_components(dx, dy, pixel_scale, config)
    assoc = _association_config(config)
    max_pair_distance_beam = float(assoc["max_pair_distance_beam"])
    max_pair_distance_arcsec = _max_pair_distance_arcsec(config)
    if distance_beam <= max_pair_distance_beam and (
        max_pair_distance_arcsec is None or distance_arcsec <= max_pair_distance_arcsec
    ):
        candidate_generation_reason = (
            "within_elliptical_beam_and_absolute_radius"
            if max_pair_distance_arcsec is not None
            else "within_elliptical_beam_radius_no_absolute_cap"
        )
    elif max_pair_distance_arcsec is not None and distance_arcsec > max_pair_distance_arcsec:
        candidate_generation_reason = "rejected_by_absolute_radius"
    else:
        candidate_generation_reason = "rejected_by_elliptical_beam_distance"
    closeness_score = float(np.clip(1.0 - distance_beam / max(max_pair_distance_beam, 1e-6), 0.0, 1.0))

    obs_major_i = safe_float(component_i.get("_maj"), safe_float(component_i.get("observed_major_arcsec"), compute_beam_size_arcsec(config)))
    obs_major_j = safe_float(component_j.get("_maj"), safe_float(component_j.get("observed_major_arcsec"), compute_beam_size_arcsec(config)))
    observed_overlap, observed_gap_beam = _ellipse_overlap_from_axes(distance_arcsec, obs_major_i, obs_major_j, projected_beam)
    ellipse_gap_beam = observed_gap_beam

    major_i, _minor_i = _component_major_minor(component_i, config)
    major_j, _minor_j = _component_major_minor(component_j, config)
    size_similarity_score = float(np.clip(min(major_i, major_j) / max(major_i, major_j, 1e-6), 0.0, 1.0))

    beam_aware_morphology = bool(assoc["enable_beam_aware_morphology"])
    if beam_aware_morphology:
        class_i = _component_classification(component_i, config)
        class_j = _component_classification(component_j, config)
        intrinsic_overlap = _intrinsic_overlap_score(component_i, component_j, distance_arcsec, projected_beam, config)
        ellipse_overlap_score = intrinsic_overlap
        pa_weight_i = effective_pa_weight({**component_i.to_dict(), **class_i}, config)
        pa_weight_j = effective_pa_weight({**component_j.to_dict(), **class_j}, config)
    else:
        class_i = {
            "morphology_class": str(component_i.get("morphology_class", "unknown") or "unknown"),
            "resolved_probability": safe_float(component_i.get("resolved_probability"), 0.0),
            "beam_like_score": safe_float(component_i.get("beam_like_score"), 0.0),
            "classification_reason": "beam_aware_morphology_disabled",
        }
        class_j = {
            "morphology_class": str(component_j.get("morphology_class", "unknown") or "unknown"),
            "resolved_probability": safe_float(component_j.get("resolved_probability"), 0.0),
            "beam_like_score": safe_float(component_j.get("beam_like_score"), 0.0),
            "classification_reason": "beam_aware_morphology_disabled",
        }
        intrinsic_overlap = observed_overlap
        ellipse_overlap_score = observed_overlap
        pa_weight_i = 1.0
        pa_weight_j = 1.0
    pa_i = effective_component_pa_pixel({**component_i.to_dict(), **class_i}, config)
    pa_j = effective_component_pa_pixel({**component_j.to_dict(), **class_j}, config)
    if np.isfinite(pa_i) and np.isfinite(pa_j):
        raw_pa_alignment_score = _alignment_score(angle_delta_180(pa_i, pa_j))
    else:
        raw_pa_alignment_score = 0.0
    pa_alignment_score = float(pa_weight_i * pa_weight_j * raw_pa_alignment_score)
    line_scores = []
    weighted_line_scores = []
    if np.isfinite(pa_i):
        score_i = _alignment_score(angle_delta_180(pa_i, line_angle))
        line_scores.append(score_i)
        weighted_line_scores.append(pa_weight_i * score_i)
    if np.isfinite(pa_j):
        score_j = _alignment_score(angle_delta_180(pa_j, line_angle))
        line_scores.append(score_j)
        weighted_line_scores.append(pa_weight_j * score_j)
    raw_line_to_pa_alignment_score = float(np.mean(line_scores)) if line_scores else 0.0
    line_to_pa_alignment_score = float(np.mean(weighted_line_scores)) if weighted_line_scores else 0.0
    flow_alignment_score = float(np.mean([pa_alignment_score, line_to_pa_alignment_score]))
    flux_ratio = _flux_ratio(component_i, component_j)
    flux_continuity_score = float(np.sqrt(flux_ratio)) if np.isfinite(flux_ratio) else 0.0

    same_3 = _shared_label(component_i, component_j, labels, thresholds, FORMAL_SNR_LEVELS["strong"])
    same_25 = _shared_label(component_i, component_j, labels, thresholds, FORMAL_SNR_LEVELS["intermediate"])
    same_2 = _shared_label(component_i, component_j, labels, thresholds, FORMAL_SNR_LEVELS["weak"])
    conn_3 = same_3 > 0
    conn_25 = same_25 > 0
    conn_2 = same_2 > 0
    only_2 = bool(conn_2 and not conn_25 and not conn_3)
    common_envelope_area_beam = 0.0
    for threshold, label_id in [
        (FORMAL_SNR_LEVELS["strong"], same_3),
        (FORMAL_SNR_LEVELS["intermediate"], same_25),
        (FORMAL_SNR_LEVELS["weak"], same_2),
    ]:
        if int(label_id) > 0:
            common_envelope_area_beam = max(
                common_envelope_area_beam,
                _label_area_beam(labels, thresholds, threshold, int(label_id), pixel_scale, config),
            )

    if not feature_enabled(config, "use_multithreshold_contour"):
        conn_3 = conn_25 = conn_2 = only_2 = False
        same_3 = same_25 = same_2 = 0
        common_envelope_area_beam = 0.0

    features: dict[str, Any] = {
        "cutout_id": component_i.get("cutout_id"),
        "gaussian_id_1": component_i.get("_gaussian_id"),
        "gaussian_id_2": component_j.get("_gaussian_id"),
        "component_index_1": int(component_i.get("component_index")),
        "component_index_2": int(component_j.get("component_index")),
        "pair_separation_arcsec": distance_arcsec,
        "distance_pix": distance_pix,
        "distance_arcsec": distance_arcsec,
        "pair_angle_deg": line_angle,
        "distance_beam": distance_beam,
        "elliptical_beam_distance": distance_beam,
        "directional_beam_arcsec": projected_beam,
        "beam_parallel_separation_arcsec": parallel_arcsec,
        "beam_perpendicular_separation_arcsec": perpendicular_arcsec,
        "beam_parallel_arcsec": parallel_arcsec,
        "beam_perpendicular_arcsec": perpendicular_arcsec,
        "pair_angle_relative_to_bpa_deg": angle_to_bpa,
        "pair_angle_relative_bpa_deg": angle_to_bpa,
        "candidate_generation_reason": candidate_generation_reason,
        "morphology_class_1": class_i.get("morphology_class", "unknown"),
        "morphology_class_2": class_j.get("morphology_class", "unknown"),
        "resolved_probability_1": class_i.get("resolved_probability", 0.0),
        "resolved_probability_2": class_j.get("resolved_probability", 0.0),
        "beam_like_score_1": class_i.get("beam_like_score", 0.0),
        "beam_like_score_2": class_j.get("beam_like_score", 0.0),
        "classification_reason_1": class_i.get("classification_reason", ""),
        "classification_reason_2": class_j.get("classification_reason", ""),
        "observed_ellipse_overlap_score": observed_overlap,
        "intrinsic_ellipse_overlap_score": intrinsic_overlap,
        "ellipse_overlap_score": ellipse_overlap_score,
        "ellipse_gap_beam": float(ellipse_gap_beam),
        "pa_weight_1": pa_weight_i,
        "pa_weight_2": pa_weight_j,
        "raw_pa_alignment_score": raw_pa_alignment_score,
        "effective_pa_alignment_score": pa_alignment_score,
        "raw_line_to_pa_alignment_score": raw_line_to_pa_alignment_score,
        "effective_line_to_pa_alignment_score": line_to_pa_alignment_score,
        "pa_alignment_score": pa_alignment_score,
        "line_to_pa_alignment_score": line_to_pa_alignment_score,
        "size_similarity_score": size_similarity_score,
        "flux_ratio": flux_ratio,
        "flux_continuity_score": flux_continuity_score,
        "closeness_score": closeness_score,
        "flow_alignment_score": flow_alignment_score,
        "connected_at_3sigma": bool(conn_3),
        "connected_at_2p5sigma": bool(conn_25),
        "connected_at_2sigma": bool(conn_2),
        "only_2sigma_connected": only_2,
        "same_label_3sigma": int(same_3),
        "same_label_2p5sigma": int(same_25),
        "same_label_2sigma": int(same_2),
        "common_envelope_area_beam": common_envelope_area_beam,
    }
    features.update(compute_bridge_features(component_i, component_j, snr_map, config))
    features.update(compute_residual_bridge_features(component_i, component_j, snr_map, config))
    features.update(compute_ridge_continuity(component_i, component_j, snr_map, config))
    features.update(compute_artifact_penalties(component_i, component_j, image, snr_map, labels_by_threshold, config, features))
    veto, veto_reason = unresolved_pair_veto(features, config)
    features["unresolved_pair_veto"] = bool(veto)
    features["unresolved_pair_veto_reason"] = veto_reason
    features["association_score"] = compute_association_score(features, config)
    features["penalties"] = json_dumps_safe(
        {
            "deep_valley_penalty": features["deep_valley_penalty"],
            "only_2sigma_penalty": features["only_2sigma_penalty"],
            "negative_bowl_penalty": features["negative_bowl_penalty"],
            "sidelobe_risk_penalty": features["sidelobe_risk_penalty"],
            "too_far_penalty": features["too_far_penalty"],
            "large_mask_swallow_penalty": features["large_mask_swallow_penalty"],
        }
    )
    return features


def compute_association_score(features: dict[str, Any], config: dict[str, Any]) -> float:
    """Compute the interpretable association score from feature evidence."""

    # Feature switches support controlled evidence ablations.
    weights = _association_weights(config)
    assoc = _association_config(config)
    use_contour = feature_enabled(config, "use_multithreshold_contour")
    use_ridge = feature_enabled(config, "use_ridge_continuity")
    use_overlap = feature_enabled(config, "use_ellipse_overlap")
    use_pa = feature_enabled(config, "use_pa_alignment")
    use_artifacts = feature_enabled(config, "use_artifact_penalties_layer1")
    score = 0.0
    score += weights["closeness"] * float(features.get("closeness_score", 0.0))
    if use_overlap:
        score += weights["overlap"] * float(features.get("ellipse_overlap_score", 0.0))
    if use_pa:
        score += weights["pa_alignment"] * float(features.get("pa_alignment_score", 0.0))
    if use_contour:
        score += weights["conn_3sigma"] * float(bool(features.get("connected_at_3sigma", False)))
        score += weights["conn_2p5sigma"] * float(bool(features.get("connected_at_2p5sigma", False)))
        score += weights["conn_2sigma"] * float(bool(features.get("connected_at_2sigma", False)))
    score += weights["bridge"] * float(features.get("bridge_score", 0.0))
    score += ASSOCIATION_RESIDUAL_BRIDGE_SCORE_FACTOR * weights["bridge"] * float(
        features.get("residual_bridge_score", 0.0)
    )
    if use_ridge:
        score += weights["ridge"] * float(features.get("ridge_continuity_score", 0.0))
    score += weights["flux_continuity"] * float(features.get("flux_continuity_score", 0.0))
    if use_pa:
        score += weights["flow_alignment"] * float(features.get("flow_alignment_score", 0.0))
    if use_artifacts:
        score -= weights["valley"] * float(features.get("deep_valley_penalty", 0.0))
        score -= weights["only_2sigma"] * float(features.get("only_2sigma_penalty", 0.0))
        score -= weights["negative_bowl"] * float(features.get("negative_bowl_penalty", 0.0))
        score -= weights["sidelobe"] * float(features.get("sidelobe_risk_penalty", 0.0))
        score -= weights["too_far"] * float(features.get("too_far_penalty", 0.0))
        score -= weights["large_mask_swallow"] * float(features.get("large_mask_swallow_penalty", 0.0))

    only_2 = bool(features.get("only_2sigma_connected", False))
    no_higher_conn = not bool(features.get("connected_at_2p5sigma", False)) and not bool(features.get("connected_at_3sigma", False))
    has_independent_support = (
        float(features.get("bridge_score", 0.0)) >= float(assoc["independent_support_bridge_min_score"])
        or float(features.get("residual_bridge_score", 0.0)) >= float(assoc["independent_support_bridge_min_score"])
        or (use_ridge and float(features.get("ridge_continuity_score", 0.0)) >= float(assoc["independent_support_ridge_min_score"]))
        or (use_overlap and float(features.get("ellipse_overlap_score", 0.0)) >= float(assoc["independent_support_overlap_min_score"]))
    )
    if use_contour and use_artifacts and only_2 and no_higher_conn and not has_independent_support:
        score = min(score, float(assoc["max_only_2sigma_score"]))
    if bool(features.get("unresolved_pair_veto", False)):
        score = min(score, float(assoc["veto_score_cap"]))
    return float(score)


def _candidate_pairs(components: pd.DataFrame, config: dict[str, Any]) -> list[tuple[int, int]]:
    """Return nearby pairs as zero-based DataFrame row positions."""

    if "component_index" in components:
        validate_integer_indices(components["component_index"], context="component_index")
    if len(components) < 2:
        return []
    # Use a KD-tree radius query before the elliptical-beam distance check.
    coords = components[["x", "y"]].to_numpy(float)
    finite = np.isfinite(coords).all(axis=1)
    if finite.sum() < 2:
        return []
    valid_positions = np.where(finite)[0]
    valid_coords = coords[finite]
    scale_value = components["pixel_scale_arcsec"].iloc[0] if "pixel_scale_arcsec" in components else None
    pixel_scale = resolve_pixel_scale_arcsec(scale_value, config, context="association group pixel scale")
    beam_major, _beam_minor, _beam_pa = beam_axes_from_config(config)
    assoc = _association_config(config)
    max_pair_distance_beam = float(assoc["max_pair_distance_beam"])
    radius_arcsec = max_pair_distance_beam * beam_major
    max_pair_distance_arcsec = _max_pair_distance_arcsec(config)
    if max_pair_distance_arcsec is not None:
        radius_arcsec = min(radius_arcsec, max_pair_distance_arcsec)
    radius_pix = radius_arcsec / max(pixel_scale, 1e-6)
    tree = cKDTree(valid_coords)
    pairs_local = tree.query_pairs(radius_pix)
    beam_cov = beam_covariance_from_config(config)
    pairs = []
    for i, j in pairs_local:
        dx_pix = float(valid_coords[j, 0] - valid_coords[i, 0])
        dy_pix = float(valid_coords[j, 1] - valid_coords[i, 1])
        dx_arcsec = dx_pix * pixel_scale
        dy_arcsec = dy_pix * pixel_scale
        distance_arcsec = float(np.hypot(dx_arcsec, dy_arcsec))
        if max_pair_distance_arcsec is not None and distance_arcsec > max_pair_distance_arcsec:
            continue
        distance_beam = elliptical_beam_distance((dx_arcsec, dy_arcsec), beam_cov)
        if not np.isfinite(distance_beam) or distance_beam > max_pair_distance_beam:
            continue
        pairs.append((int(valid_positions[i]), int(valid_positions[j])))
    pairs.sort()
    return pairs


# Public alias used by release consumers that need to inspect the Stage-1
# spatial preselection independently of graph construction.
candidate_pairs = _candidate_pairs


def build_association_graph(
    components: pd.DataFrame,
    features: pd.DataFrame | list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[nx.Graph, pd.DataFrame]:
    """Create a graph whose core edges are strong association decisions."""

    # Strong edges form graph cores; weak edges are handled during clustering.
    assoc = _association_config(config)
    threshold_strong = float(assoc["threshold_strong"])
    threshold_weak = float(assoc["threshold_weak"])
    if "component_index" not in components:
        raise ValueError("association components must provide component_index")
    components = components.copy()
    components["component_index"] = validate_integer_indices(components["component_index"], context="component_index")
    if components["component_index"].duplicated(keep=False).any():
        raise ValueError("association components contain duplicate component_index values")
    graph = nx.Graph()
    for _, row in components.iterrows():
        graph.add_node(int(row["component_index"]), **row.to_dict())

    edges = pd.DataFrame(features)
    edges = _with_columns(edges, EDGE_COLUMNS)
    if edges.empty:
        graph.graph["components"] = components.copy()
        graph.graph["association_edges"] = edges
        return graph, edges

    # Candidate generation emits one unordered pair per edge.  Enforce that
    # contract here as well because this public graph builder is also used by
    # downstream callers with precomputed feature tables.
    try:
        left = validate_integer_indices(edges["component_index_1"], context="component_index_1")
        right = validate_integer_indices(edges["component_index_2"], context="component_index_2")
    except ValueError as exc:
        raise ValueError("association edge table contains invalid component indices") from exc
    self_loop = left == right
    if self_loop.any():
        nodes = ", ".join(str(value) for value in sorted(set(left[self_loop].tolist())))
        raise ValueError(f"association edge table contains self-loop(s): {nodes}")
    pair_frame = pd.DataFrame({"left": np.minimum(left, right), "right": np.maximum(left, right)})
    if pair_frame.duplicated(keep=False).any():
        duplicates = pair_frame.loc[pair_frame.duplicated(keep=False)].drop_duplicates().head(10)
        preview = ", ".join(f"({row.left}, {row.right})" for row in duplicates.itertuples(index=False))
        raise ValueError(f"association edge table contains duplicate pair(s): {preview}")
    known_nodes = set(int(value) for value in components["component_index"].tolist())
    unknown = sorted((set(left.tolist()) | set(right.tolist())) - known_nodes)
    if unknown:
        preview = ", ".join(str(value) for value in unknown[:10])
        raise ValueError(f"association edge table references unknown component_index value(s): {preview}")
    edges = edges.assign(component_index_1=left, component_index_2=right)
    edges = edges.sort_values(["component_index_1", "component_index_2"], kind="mergesort").reset_index(drop=True)

    edge_types = []
    decisions = []
    reasons = []
    for _, row in edges.iterrows():
        score = safe_float(row.get("association_score"), -np.inf)
        distance_beam = safe_float(row.get("distance_beam"), np.inf)
        distance_arcsec = safe_float(row.get("distance_arcsec"), np.inf)
        max_pair_distance_arcsec = _max_pair_distance_arcsec(config)
        if max_pair_distance_arcsec is not None and distance_arcsec > max_pair_distance_arcsec:
            edge_types.append("rejected")
            decisions.append(False)
            reasons.append("too_far_absolute_radius")
        elif distance_beam > float(assoc["max_pair_distance_beam"]):
            edge_types.append("rejected")
            decisions.append(False)
            reasons.append("too_far")
        elif bool(row.get("unresolved_pair_veto", False)):
            edge_types.append("rejected")
            decisions.append(False)
            reasons.append(str(row.get("unresolved_pair_veto_reason") or "veto_unresolved_pair_no_independent_radio_evidence"))
        elif score >= threshold_strong:
            edge_types.append("strong")
            decisions.append(True)
            reasons.append("")
        elif score >= threshold_weak:
            edge_types.append("weak")
            decisions.append(False)
            reasons.append("pending_weak_attachment")
        else:
            edge_types.append("rejected")
            decisions.append(False)
            reasons.append("score_below_threshold")
    edges["edge_type"] = edge_types
    edges["association_decision"] = decisions
    edges["rejection_reason"] = reasons

    for _, row in edges[edges["edge_type"] == "strong"].iterrows():
        graph.add_edge(
            int(row["component_index_1"]),
            int(row["component_index_2"]),
            association_score=float(row["association_score"]),
            edge_type="strong",
        )
    graph.graph["components"] = components.copy()
    graph.graph["association_edges"] = edges
    return graph, edges


def cluster_association_groups(graph: nx.Graph, config: dict[str, Any]) -> tuple[list[list[int]], pd.DataFrame, nx.Graph]:
    """Cluster by strong edges, then attach weak edges without chain merging."""

    edges = graph.graph.get("association_edges", pd.DataFrame()).copy()
    nodes = sorted(int(node) for node in graph.nodes)
    strong_graph = nx.Graph()
    strong_graph.add_nodes_from(nodes)
    if not edges.empty:
        for _, row in edges[edges["edge_type"] == "strong"].iterrows():
            strong_graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]))

    strong_components = [sorted(list(values)) for values in nx.connected_components(strong_graph)]
    strong_components.sort(key=lambda values: (values[0] if values else -1))
    group_by_node: dict[int, int] = {}
    original_group_by_node: dict[int, int] = {}
    original_group_size: dict[int, int] = {}
    for group_idx, members in enumerate(strong_components):
        original_group_size[group_idx] = len(members)
        for node in members:
            group_by_node[int(node)] = group_idx
            original_group_by_node[int(node)] = group_idx

    final_graph = nx.Graph()
    final_graph.add_nodes_from(graph.nodes(data=True))
    for _, row in edges[edges["edge_type"] == "strong"].iterrows():
        final_graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]), edge_type="strong")

    if not edges.empty:
        weak_order = edges[edges["edge_type"] == "weak"].sort_values("association_score", ascending=False).index.tolist()
        attached_singletons: set[int] = set()
        for idx in weak_order:
            row = edges.loc[idx]
            left = int(row["component_index_1"])
            right = int(row["component_index_2"])
            group_left = group_by_node[left]
            group_right = group_by_node[right]
            if group_left == group_right:
                edges.at[idx, "rejection_reason"] = "weak_edge_already_same_group"
                continue

            left_orig = original_group_by_node[left]
            right_orig = original_group_by_node[right]
            left_core = original_group_size[left_orig] >= 2
            right_core = original_group_size[right_orig] >= 2
            left_single = original_group_size[left_orig] == 1 and left not in attached_singletons
            right_single = original_group_size[right_orig] == 1 and right not in attached_singletons

            attach_node: int | None = None
            target_group: int | None = None
            if left_core and right_single:
                attach_node = right
                target_group = group_left
            elif right_core and left_single:
                attach_node = left
                target_group = group_right

            if attach_node is not None and target_group is not None:
                group_by_node[attach_node] = target_group
                attached_singletons.add(attach_node)
                edges.at[idx, "association_decision"] = True
                edges.at[idx, "rejection_reason"] = ""
                final_graph.add_edge(left, right, edge_type="weak")
            elif left_core and right_core:
                edges.at[idx, "rejection_reason"] = "weak_edge_would_merge_core_groups"
            elif (not left_core) and (not right_core):
                edges.at[idx, "rejection_reason"] = "weak_edge_no_core_group"
            else:
                edges.at[idx, "rejection_reason"] = "weak_edge_would_form_chain"

    if feature_enabled(config, "use_weak_edge_anti_chaining"):
        grouped: dict[int, list[int]] = {}
        for node, group_idx in group_by_node.items():
            grouped.setdefault(int(group_idx), []).append(int(node))
        clusters = [sorted(values) for values in grouped.values()]
        clusters.sort(key=lambda values: (len(values), -values[0] if values else 0), reverse=True)
    else:
        final_graph = nx.Graph()
        final_graph.add_nodes_from(graph.nodes(data=True))
        if not edges.empty:
            # Keep the diagnostic edge table identical to the constrained run,
            # but cluster over all classified strong/weak edges.
            for _, row in edges[edges["edge_type"].astype(str).isin(["strong", "weak"])].iterrows():
                final_graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]), edge_type=str(row.get("edge_type", "")))
        clusters = [sorted(int(node) for node in values) for values in nx.connected_components(final_graph)]
        clusters.sort(key=lambda values: (values[0] if values else -1))
    final_graph.graph["association_edges"] = edges
    final_graph.graph["components"] = graph.graph.get("components", pd.DataFrame()).copy()
    return clusters, edges, final_graph


def classify_association_group(
    group: pd.DataFrame,
    features: pd.DataFrame,
    measurements: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Assign a radio association type without FRII-specific labels."""

    types = {
        name: {**defaults, **(config.get("association_types", {}) or {}).get(name, {})}
        for name, defaults in ASSOCIATION_TYPE_DEFAULTS.items()
    }
    n_gaussians = int(measurements.get("n_gaussians", len(group)))
    las_beam = float(measurements.get("LAS_beam", 0.0) or 0.0)
    axis_ratio = float(measurements.get("axis_ratio", 1.0) or 1.0)
    quality = str(measurements.get("association_quality", "low"))
    flags = str(measurements.get("artifact_risk_flags", ""))
    artifact_type_enabled = bool(types["artifact_risk"]["enabled"])
    if artifact_type_enabled and feature_enabled(config, "use_artifact_penalties_layer1") and (
        "negative_bowl" in flags or "sidelobe" in flags or "large_mask_swallow" in flags
    ):
        return "artifact_risk"
    if n_gaussians >= int(types["complex_association"]["min_components"]):
        return "complex_association"
    if n_gaussians <= 1:
        return "weak_association"
    if las_beam <= float(types["compact_multi_gaussian"]["max_las_beam"]):
        return "compact_multi_gaussian"
    if axis_ratio >= float(types["linear_or_tail_like"]["min_axis_ratio"]):
        return "linear_or_tail_like"
    has_continuity = False
    if not features.empty:
        has_continuity = bool(
            (
                feature_enabled(config, "use_multithreshold_contour")
                and (
                    features["connected_at_3sigma"].astype(bool).any()
                    or features["connected_at_2p5sigma"].astype(bool).any()
                )
            )
            or (pd.to_numeric(features["bridge_score"], errors="coerce").fillna(0) >= ASSOCIATION_SUPPORT_SCORE_MIN).any()
            or (
                feature_enabled(config, "use_ridge_continuity")
                and (pd.to_numeric(features["ridge_continuity_score"], errors="coerce").fillna(0) >= ASSOCIATION_SUPPORT_SCORE_MIN).any()
            )
        )
    if las_beam >= float(types["diffuse_extended"]["min_las_beam"]) and quality in {"low", "suspicious"}:
        return "diffuse_extended"
    if has_continuity and las_beam >= float(types["continuous_extended"]["min_las_beam"]):
        return "continuous_extended"
    quality_order = {"artifact_risk": 0, "suspicious": 1, "low": 1, "medium": 2, "high": 3}
    weak_limit = str(types["weak_association"]["max_quality"]).strip().lower()
    if quality_order.get(quality, 1) <= quality_order.get(weak_limit, quality_order["low"]):
        return "weak_association"
    return "continuous_extended"


def assign_association_quality(
    group: pd.DataFrame,
    edge_scores: pd.DataFrame,
    flags: list[str],
    config: dict[str, Any],
) -> str:
    """Assign high/medium/low/suspicious/artifact_risk quality labels."""

    del group
    thresholds = _association_config(config)["quality_thresholds"]
    high = float(thresholds["high"])
    medium = float(thresholds["medium"])
    low = float(thresholds["low"])
    severe_flags = {"negative_bowl", "sidelobe_risk", "large_mask_swallow"}
    if feature_enabled(config, "use_artifact_penalties_layer1") and any(flag in severe_flags for flag in flags):
        return "artifact_risk"
    if edge_scores.empty:
        return "low"
    scores = pd.to_numeric(edge_scores["association_score"], errors="coerce").dropna()
    mean_score = float(scores.mean()) if len(scores) else 0.0
    score_spread = float(scores.max() - scores.min()) if len(scores) else 0.0
    n_edges = int(len(edge_scores))
    n_strong = int((edge_scores["edge_type"].astype(str) == "strong").sum())
    n_only_2 = int(edge_scores["only_2sigma_connected"].astype(bool).sum()) if "only_2sigma_connected" in edge_scores else 0
    has_strong_support = bool(
        (
            feature_enabled(config, "use_multithreshold_contour")
            and (
                edge_scores["connected_at_3sigma"].astype(bool).any()
                or edge_scores["connected_at_2p5sigma"].astype(bool).any()
            )
        )
        or (pd.to_numeric(edge_scores["bridge_score"], errors="coerce").fillna(0) >= ASSOCIATION_STRONG_SUPPORT_SCORE_MIN).any()
        or (
            feature_enabled(config, "use_ridge_continuity")
            and (pd.to_numeric(edge_scores["ridge_continuity_score"], errors="coerce").fillna(0) >= ASSOCIATION_STRONG_SUPPORT_SCORE_MIN).any()
        )
    )
    if (
        feature_enabled(config, "use_artifact_penalties_layer1")
        and ("only_2sigma" in flags or n_only_2 > max(1, int(ASSOCIATION_SEVERE_SUPPORT_FRACTION * n_edges)) or score_spread > ASSOCIATION_SEVERE_SCORE_SPREAD)
    ):
        return "suspicious"
    if (
        mean_score >= high
        and n_strong >= max(1, int(ASSOCIATION_HIGH_QUALITY_STRONG_EDGE_FRACTION * n_edges))
        and has_strong_support
    ):
        return "high"
    if mean_score >= medium and has_strong_support:
        return "medium"
    if mean_score >= low:
        return "low"
    return "low"


def _artifact_flags_from_edges(edges: pd.DataFrame) -> list[str]:
    flags: set[str] = set()
    if edges.empty:
        return []
    for text in edges.get("artifact_risk_flags", pd.Series(dtype=str)).astype(str):
        for item in text.split(","):
            item = item.strip()
            if item:
                flags.add(item)
    return sorted(flags)


def _accepted_internal_edges(edges: pd.DataFrame, nodes: set[int]) -> pd.DataFrame:
    if edges.empty or len(nodes) < 2:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    mask = (
        edges["association_decision"].astype(bool)
        & edges["component_index_1"].astype(int).isin(nodes)
        & edges["component_index_2"].astype(int).isin(nodes)
    )
    return edges.loc[mask].copy()


def _measure_groups(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    clusters: list[list[int]],
    edges: pd.DataFrame,
    config: dict[str, Any],
    group_metadata: dict[frozenset[int], dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure catalogue fields from the supplied final clusters and edges."""

    image = np.asarray(cutout.image, dtype=float)
    beam_arcsec = compute_beam_size_arcsec(config)
    group_metadata = group_metadata or {}
    records: list[dict[str, Any]] = []
    group_id_by_node: dict[int, str] = {}
    original_group_id_by_node: dict[int, str] = {}
    group_index_by_node: dict[int, int] = {}
    group_size_by_node: dict[int, int] = {}
    quality_by_node: dict[int, str] = {}
    type_by_node: dict[int, str] = {}

    for group_idx, nodes in enumerate(clusters):
        node_set = set(int(node) for node in nodes)
        metadata = group_metadata.get(frozenset(node_set), {})
        group_rows = components[components["component_index"].astype(int).isin(node_set)].copy()
        if group_rows.empty:
            continue
        pixel_scale = resolve_pixel_scale_arcsec(group_rows["pixel_scale_arcsec"].iloc[0], config, context="association measurement pixel scale")
        padding = max(
            ASSOCIATION_GROUP_SUPPORT_MIN_PADDING_PIX,
            int(round(ASSOCIATION_GROUP_SUPPORT_PADDING_BEAM * beam_arcsec / max(pixel_scale, 1e-6))),
        )
        support_2 = component_support_mask(
            segmentation.labels_by_threshold,
            segmentation.thresholds,
            group_rows,
            FORMAL_SNR_LEVELS["weak"],
        )
        support_25 = component_support_mask(
            segmentation.labels_by_threshold,
            segmentation.thresholds,
            group_rows,
            FORMAL_SNR_LEVELS["intermediate"],
        )
        support_mask = support_25 if support_25.any() else support_2
        x = group_rows["x"].to_numpy(float)
        y = group_rows["y"].to_numpy(float)
        bbox = _bbox_from_mask(support_mask) if support_mask.any() else _bbox_from_points(x, y, padding, image.shape)
        if bbox is None:
            bbox = _bbox_from_points(x, y, padding, image.shape)
        x0, y0, x1, y1 = bbox
        weights = pd.to_numeric(group_rows["_peak_flux"], errors="coerce").to_numpy(float, copy=True)
        weights[~np.isfinite(weights) | (weights <= 0)] = 1.0
        centroid_x = float(np.average(x, weights=weights))
        centroid_y = float(np.average(y, weights=weights))
        ra = float("nan")
        dec = float("nan")
        if getattr(cutout, "wcs", None) is not None:
            try:
                ra, dec = cutout.wcs.celestial.pixel_to_world_values(centroid_x, centroid_y)
                ra = float(ra)
                dec = float(dec)
            except Exception as exc:
                raise ValueError(
                    f"failed to convert association-group centroid to sky coordinates for {cutout.cutout_id}"
                ) from exc
        las_pix, las_arcsec = _las_from_points(x, y, pixel_scale)
        if support_mask.any():
            las_pix, las_arcsec = _support_las(support_mask, pixel_scale, las_pix)
        _major, _minor, group_pa, axis_ratio = _second_moments(image, support_mask, x, y)
        total_flux = float(np.nansum(pd.to_numeric(group_rows["_total_flux"], errors="coerce")))
        peak_flux = float(np.nanmax(pd.to_numeric(group_rows["_peak_flux"], errors="coerce")))
        internal_edges = _accepted_internal_edges(edges, node_set)
        scores = pd.to_numeric(internal_edges.get("association_score", pd.Series(dtype=float)), errors="coerce").dropna()
        score_mean = float(scores.mean()) if len(scores) else 0.0
        score_min = float(scores.min()) if len(scores) else 0.0
        score_max = float(scores.max()) if len(scores) else 0.0
        n_strong = int((internal_edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong").sum()) if not internal_edges.empty else 0
        n_weak = int((internal_edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak").sum()) if not internal_edges.empty else 0
        n_only_2 = int(internal_edges.get("only_2sigma_connected", pd.Series(dtype=bool)).astype(bool).sum()) if not internal_edges.empty else 0
        flags = _artifact_flags_from_edges(internal_edges)
        if n_only_2 > max(1, int(ASSOCIATION_ONLY_2SIGMA_FLAG_FRACTION * len(internal_edges))):
            flags.append("only_2sigma")
        flags = sorted(set(flags))
        quality = assign_association_quality(group_rows, internal_edges, flags, config)
        if bool(metadata.get("force_suspicious", False)) and quality != "artifact_risk":
            quality = "suspicious"
        measurement_for_type = {
            "n_gaussians": int(len(group_rows)),
            "LAS_beam": float(las_arcsec / max(beam_arcsec, 1e-6)),
            "axis_ratio": axis_ratio,
            "association_quality": quality,
            "artifact_risk_flags": ",".join(flags),
        }
        association_type = classify_association_group(group_rows, internal_edges, measurement_for_type, config)
        component_classes = group_rows.get("morphology_class", pd.Series(dtype=str)).astype(str).tolist()
        component_probs = pd.to_numeric(group_rows.get("resolved_probability", pd.Series(dtype=float)), errors="coerce")
        component_beam_like = pd.to_numeric(group_rows.get("beam_like_score", pd.Series(dtype=float)), errors="coerce")
        if len(group_rows) == 1 and component_classes:
            group_morphology = component_classes[0] or "unknown"
            group_reason = str(group_rows.iloc[0].get("classification_reason", ""))
        elif association_type == "artifact_risk" or quality == "artifact_risk":
            group_morphology = "artifact_like"
            group_reason = "artifact_risk_group"
        elif len(group_rows) >= 2 and (
            association_type in {"continuous_extended", "diffuse_extended", "linear_or_tail_like", "complex_association"}
            or n_strong > 0
            or las_arcsec / max(beam_arcsec, 1e-6) >= ASSOCIATION_MULTI_GAUSSIAN_EXTENDED_MIN_LAS_BEAM
        ):
            group_morphology = "multi_gaussian_extended"
            group_reason = "multi_gaussian_extended"
        elif component_classes and all(value == "unresolved" for value in component_classes):
            group_morphology = "unresolved"
            group_reason = "all_components_unresolved"
        else:
            group_morphology = "unknown"
            group_reason = "mixed_or_low_support_group"
        group_resolved_probability = float(component_probs.max()) if len(component_probs.dropna()) else 0.0
        group_beam_like_score = float(component_beam_like.max()) if len(component_beam_like.dropna()) else 0.0
        group_id = str(metadata.get("association_group_id") or f"{cutout.cutout_id}_a{group_idx:03d}")
        original_group_id = str(metadata.get("original_association_group_id") or group_id)
        record = {
            "cutout_id": cutout.cutout_id,
            "association_group_id": group_id,
            "original_association_group_id": original_group_id,
            "association_group_index": int(group_idx),
            "component_ids": ",".join(map(str, sorted(node_set))),
            "n_gaussians": int(len(group_rows)),
            "gaussian_ids": ",".join(map(str, group_rows["_gaussian_id"].tolist())),
            "ra": ra,
            "dec": dec,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "bounding_box": f"{x0},{y0},{x1},{y1}",
            "LAS_arcsec": las_arcsec,
            "LAS_beam": float(las_arcsec / max(beam_arcsec, 1e-6)),
            "total_flux_gaussian": total_flux,
            "peak_flux": peak_flux,
            "group_PA": group_pa,
            "axis_ratio": axis_ratio,
            "association_score_mean": score_mean,
            "association_score_min": score_min,
            "association_score_max": score_max,
            "n_strong_edges": n_strong,
            "n_weak_edges": n_weak,
            "n_only_2sigma_edges": n_only_2,
            "association_quality": quality,
            "association_type": association_type,
            "morphology_class": group_morphology,
            "resolved_probability": group_resolved_probability,
            "beam_like_score": group_beam_like_score,
            "classification_reason": group_reason,
            "artifact_risk_flags": ",".join(flags),
            "debug_info": json_dumps_safe(
                {
                    "cluster_nodes": sorted(node_set),
                    "bbox": [x0, y0, x1, y1],
                    "support_pixels_2sigma": int(support_2.sum()),
                    "support_pixels_2p5sigma": int(support_25.sum()),
                }
            ),
        }
        records.append(record)
        for node in node_set:
            group_id_by_node[node] = group_id
            original_group_id_by_node[node] = original_group_id
            group_index_by_node[node] = group_idx
            group_size_by_node[node] = int(len(group_rows))
            quality_by_node[node] = quality
            type_by_node[node] = association_type

    groups = _with_columns(pd.DataFrame(records), GROUP_COLUMNS)
    components = components.copy()
    components["association_group_id"] = components["component_index"].astype(int).map(group_id_by_node).fillna("")
    components["original_association_group_id"] = components["component_index"].astype(int).map(original_group_id_by_node).fillna("")
    components["association_group_index"] = components["component_index"].astype(int).map(group_index_by_node).fillna(-1).astype(int)
    components["association_group_size"] = components["component_index"].astype(int).map(group_size_by_node).fillna(1).astype(int)
    components["association_quality"] = components["component_index"].astype(int).map(quality_by_node).fillna("low")
    components["association_type"] = components["component_index"].astype(int).map(type_by_node).fillna("weak_association")
    return groups, components


def run_component_association(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    config: dict[str, Any],
) -> AssociationResult:
    """Run the full beam-aware association pipeline for one cutout."""

    if "component_index" not in components:
        raise ValueError("association components must provide component_index")
    components = components.copy()
    components["component_index"] = validate_integer_indices(components["component_index"], context="component_index")
    if components["component_index"].duplicated(keep=False).any():
        raise ValueError("association components contain duplicate component_index values")
    if "_gaussian_id" in components and components["_gaussian_id"].duplicated(keep=False).any():
        raise ValueError("association components contain duplicate Gaussian IDs")
    components = add_morphology_columns(components, config) if bool(_association_config(config)["enable_beam_aware_morphology"]) else components.copy()
    if components.empty:
        graph = nx.Graph()
        return AssociationResult(
            graph=graph,
            edges=pd.DataFrame(columns=EDGE_COLUMNS),
            components=components.copy(),
            groups=pd.DataFrame(columns=GROUP_COLUMNS),
            clusters=[],
        )

    feature_records: list[dict[str, Any]] = []
    for idx_i, idx_j in _candidate_pairs(components, config):
        row_i = components.iloc[idx_i]
        row_j = components.iloc[idx_j]
        feature_records.append(
            compute_pair_association_features(
                row_i,
                row_j,
                np.asarray(cutout.image, dtype=float),
                segmentation.snr_map,
                segmentation,
                config,
            )
        )

    graph, edges = build_association_graph(components, feature_records, config)
    clusters, edges, graph = cluster_association_groups(graph, config)
    if not clusters:
        clusters = [[int(value)] for value in components["component_index"].tolist()]
    groups, assoc_components = _measure_groups(cutout, segmentation, components, clusters, edges, config)
    local_diagnostics = pd.DataFrame()
    local_needs_visual_check = pd.DataFrame()
    local_cfg = (config.get("local_association", {}) or {}).get("local_sanity", {}) or {}
    if bool(local_cfg.get("enabled", True)):
        from .local_sanity import run_local_sanity

        local_result = run_local_sanity(
            str(cutout.cutout_id),
            groups,
            edges,
            assoc_components,
            segmentation,
            np.asarray(cutout.image, dtype=float),
            config,
            wcs=getattr(cutout, "wcs", None),
        )
        local_diagnostics = local_result.diagnostics
        local_needs_visual_check = local_result.needs_visual_check
        local_groups = local_result.groups.copy()
        if not local_groups.empty:
            edges = local_result.edges.copy()
            ordered_local_groups = local_groups.sort_values("local_group_index", kind="mergesort")
            clusters = []
            group_metadata: dict[frozenset[int], dict[str, Any]] = {}
            assigned_nodes: set[int] = set()
            for _, row in ordered_local_groups.iterrows():
                local_group_id = str(row.get("local_group_id", ""))
                cluster = sorted(
                    local_result.components.loc[
                        local_result.components["local_group_id"].astype(str) == local_group_id,
                        "component_index",
                    ].astype(int)
                )
                cluster_key = frozenset(cluster)
                if not cluster or cluster_key in group_metadata or assigned_nodes.intersection(cluster):
                    raise RuntimeError("local sanity produced invalid or duplicate group membership")
                assigned_nodes.update(cluster)
                clusters.append(cluster)
                group_metadata[cluster_key] = {
                    "association_group_id": local_group_id,
                    "original_association_group_id": str(row.get("original_association_group_id", "")),
                    "force_suspicious": bool(
                        not bool(row.get("split_from_original", False))
                        and str(row.get("local_quality", "")) == "suspicious"
                        and bool(str(row.get("split_reason", "")).strip())
                    ),
                }

            node_set = set(int(value) for value in components["component_index"].tolist())
            if assigned_nodes != node_set:
                raise RuntimeError("local sanity group membership does not cover the association components")
            final_graph = nx.Graph()
            final_graph.add_nodes_from(graph.nodes(data=True))
            if not edges.empty:
                decision = edges.get("association_decision", pd.Series(False, index=edges.index))
                for _, row in edges[decision.astype(bool)].iterrows():
                    final_graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]), edge_type=str(row.get("edge_type", "")))
            graph_clusters = {frozenset(int(node) for node in values) for values in nx.connected_components(final_graph)}
            if graph_clusters != set(group_metadata):
                raise RuntimeError("local sanity groups disagree with the final association graph")
            groups, assoc_components = _measure_groups(
                cutout,
                segmentation,
                components,
                clusters,
                edges,
                config,
                group_metadata=group_metadata,
            )
            final_graph.graph["association_edges"] = edges.copy()
            final_graph.graph["components"] = assoc_components.copy()
            graph = final_graph
    # A component is represented by exactly one final local group.  Keep this
    # invariant explicit so downstream catalogues cannot silently duplicate or
    # orphan Gaussian membership.
    if assoc_components["component_index"].duplicated(keep=False).any():
        raise RuntimeError("association output contains duplicate component_index values")
    if assoc_components["association_group_id"].astype(str).eq("").any():
        raise RuntimeError("association output contains components without a group assignment")
    if not edges.empty:
        edges = _with_columns(edges, EDGE_COLUMNS)
        edges["debug_info"] = edges.apply(lambda row: json_dumps_safe(row.to_dict()), axis=1)
    return AssociationResult(graph=graph, edges=edges, components=assoc_components, groups=groups, clusters=clusters, local_sanity_diagnostics=local_diagnostics, local_needs_visual_check=local_needs_visual_check)
