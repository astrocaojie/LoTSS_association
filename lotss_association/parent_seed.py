"""parent-seed refined conservative parent-link candidates.

The refined pass is a post-processing layer on top of local association
groups. It first decides which local groups are reliable parent-link
endpoints, then proposes a small number of nearby parent-link candidates using
robust 3 sigma or 5 sigma bounding boxes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ._shared import (
    alignment_score as _alignment_score,
    angle_delta_deg as _angle_delta_deg,
    balance_ratio_score as balance_ratio_score,
    beam_area_pix as _shared_beam_area_pix,
    select_columns as _with_columns,
)
from .association import compute_beam_size_arcsec
from .config import (
    PARENT_LINK_DEFAULTS,
    PARENT_SEED_DEFAULTS,
    POSITION_ANGLE_ALIGNMENT_SCALE_DEG,
)
from .utils import json_dumps_safe, resolve_pixel_scale_arcsec, safe_float

PARENT_SEED_CANDIDATE_COLUMNS = [
    "cutout_id",
    "parent_candidate_id",
    "local_group_id_1",
    "local_group_id_2",
    "box_gap_pix",
    "box_gap_arcsec",
    "box_gap_beam",
    "box_gap_beam_raw",
    "box_gap_beam_robust",
    "box_gap_source",
    "center_distance_arcsec",
    "center_distance_beam",
    "n_gaussians_1",
    "n_gaussians_2",
    "LAS_beam_1",
    "LAS_beam_2",
    "mask_area_beam_1",
    "mask_area_beam_2",
    "area_3sigma_beam_1",
    "area_3sigma_beam_2",
    "peak_snr_1",
    "peak_snr_2",
    "axis_alignment_score",
    "facing_score",
    "flux_ratio",
    "size_ratio",
    "core_candidate_near_midpoint",
    "midpoint_core_x",
    "midpoint_core_y",
    "midpoint_symmetry_score",
    "midpoint_symmetry_source",
    "parent_score",
    "parent_candidate_quality",
    "rejection_reason",
    "needs_visual_check",
]

PARENT_AXIS_ALIGNMENT_SCALE_DEG = POSITION_ANGLE_ALIGNMENT_SCALE_DEG
PARENT_RATIO_FLOOR = 0.5
FORMAL_BBOX_THRESHOLDS_SIGMA = (3.0, 5.0)

PARENT_SEED_EDGE_DEBUG_COLUMNS = [
    *PARENT_SEED_CANDIDATE_COLUMNS,
    "parent_axis_angle",
    "group1_PA",
    "group2_PA",
    "multi_evidence_pass",
    "support_evidence_count",
    "endpoint1_pass",
    "endpoint2_pass",
    "both_parent_seed",
    "both_groups_extended_or_lobe_like",
    "compact_singleton_pair",
    "is_default_candidate",
    "debug_info",
]

PARENT_SEED_DIAGNOSTIC_COLUMNS = [
    "cutout_id",
    "n_local_groups",
    "n_parent_seed_groups",
    "n_parent_pairs_considered",
    "n_parent_candidates",
    "n_parent_high",
    "n_parent_medium",
    "n_parent_low_debug",
    "n_rejected",
    "n_rejected_compact_singleton_pair",
    "n_rejected_endpoint_not_parent_seed",
    "n_rejected_robust_box_gap_too_large",
    "n_rejected_insufficient_evidence_for_parent_link",
    "n_missing_peak_snr",
    "n_missing_area_3sigma_beam",
    "n_missing_robust_bbox",
    "parent_candidate_overflow_flag",
    "n_candidates_before_cutout_limit",
    "max_parent_candidates_per_cutout",
    "max_parent_candidates_per_group",
    "max_candidate_box_gap_beam_robust",
]

PARENT_SEED_COLUMNS = [
    "cutout_id",
    "association_group_id",
    "n_gaussians",
    "LAS_beam",
    "mask_area_beam",
    "area_3sigma_beam",
    "peak_snr",
    "association_quality",
    "association_type",
    "is_compact_singleton",
    "is_parent_seed",
    "parent_seed_reason",
    "parent_seed_reject_reason",
    "robust_bbox",
    "robust_bbox_source",
    "missing_fields",
]


@dataclass
class ParentSeedResult:
    candidates: pd.DataFrame
    edges_debug: pd.DataFrame
    diagnostics: pd.DataFrame
    needs_visual_check: pd.DataFrame
    parent_seed_table: pd.DataFrame


def parent_seed_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = copy.deepcopy(PARENT_SEED_DEFAULTS)
    raw = (config.get("parent_seed_selection", {}) or {}).copy()
    out = dict(defaults)
    out.update({key: value for key, value in raw.items() if key not in {"thresholds", "evidence"}})
    out["thresholds"] = dict(defaults["thresholds"])
    out["thresholds"].update(raw.get("thresholds", {}) or {})
    out["evidence"] = dict(defaults["evidence"])
    out["evidence"].update(raw.get("evidence", {}) or {})
    for nested in ("compact_singleton", "mask_area_fallback", "midpoint_core", "score_weights"):
        out[nested] = dict(defaults[nested])
        out[nested].update(raw.get(nested, {}) or {})
    out["max_parent_candidates_per_group"] = int(out["max_parent_candidates_per_group"])
    out["max_parent_candidates_per_cutout"] = int(out["max_parent_candidates_per_cutout"])
    return out


def parent_endpoint_thresholds(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return the shared Stage 2 endpoint thresholds."""

    defaults = copy.deepcopy(PARENT_LINK_DEFAULTS["endpoint_thresholds"])
    raw = (config or {}).get("parent_linking", {}) or {}
    defaults.update(raw.get("endpoint_thresholds", {}) or {})
    return defaults


def _bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    try:
        values = [float(item) for item in str(value).split(",")]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 4 or not np.all(np.isfinite(values)):
        return None
    x0, y0, x1, y1 = values
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _bbox_string(box: tuple[float, float, float, float] | None) -> str:
    if box is None:
        return ""
    return ",".join(str(int(round(value))) for value in box)


def _box_gap_pix(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(np.hypot(dx, dy))


def _ratio_value(a: float, b: float) -> float:
    a = safe_float(a, 0.0)
    b = safe_float(b, 0.0)
    if a <= 0 or b <= 0:
        return float("nan")
    return float(max(a, b) / max(min(a, b), 1e-6))


def _ratio_score(ratio: float, max_ratio: float, missing_score: float = float(PARENT_SEED_DEFAULTS["evidence"]["missing_ratio_score"])) -> float:
    return balance_ratio_score(ratio, max_ratio, missing_score)


def _association_type(row: pd.Series) -> str:
    return str(row.get("association_type", row.get("local_association_type", ""))).strip().lower()


def _association_quality(row: pd.Series) -> str:
    return str(row.get("association_quality", row.get("local_quality", ""))).strip().lower()


def _parse_debug_support(row: pd.Series, token: str) -> float:
    debug = str(row.get("debug_info", ""))
    marker = f'"{token}":'
    if marker not in debug:
        return float("nan")
    try:
        return float(debug.split(marker, 1)[1].split(",", 1)[0].split("}", 1)[0].strip())
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _beam_area_pix(pixel_scale: float, config: dict[str, Any]) -> float:
    return _shared_beam_area_pix(pixel_scale, compute_beam_size_arcsec(config))


def _mask_area_beam(row: pd.Series, config: dict[str, Any]) -> float:
    value = safe_float(row.get("mask_area_beam"), float("nan"))
    if np.isfinite(value):
        return value
    support_pix = _parse_debug_support(row, "support_pixels_2p5sigma")
    if not np.isfinite(support_pix):
        support_pix = _parse_debug_support(row, "support_pixels_2sigma")
    pixel_scale = resolve_pixel_scale_arcsec(row.get("pixel_scale_arcsec"), config, context="parent-seed pixel scale")
    if np.isfinite(support_pix) and support_pix > 0:
        return float(support_pix / max(_beam_area_pix(pixel_scale, config), 1e-6))
    las_beam = safe_float(row.get("LAS_beam"), 0.0)
    n_gauss = safe_float(row.get("n_gaussians"), 1.0)
    fallback_cfg = parent_seed_config(config).get("mask_area_fallback", {}) or {}
    return float(
        max(
            0.0,
            float(fallback_cfg["las_weight"]) * las_beam
            + float(fallback_cfg["gaussian_count_weight"]) * n_gauss,
        )
    )


def _component_ids(row: pd.Series) -> list[int]:
    ids: list[int] = []
    for item in str(row.get("component_ids", "")).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.append(int(float(item)))
        except (TypeError, ValueError, OverflowError):
            continue
    return ids


def _threshold_match_tolerance(config: dict[str, Any]) -> float:
    parent_cfg = config.get("parent_linking") or PARENT_LINK_DEFAULTS
    support_cfg = parent_cfg.get("support_thresholds") or PARENT_LINK_DEFAULTS["support_thresholds"]
    return float(support_cfg["threshold_match_tolerance"])


def _threshold_index(segmentation: Any, threshold: float, tolerance: float) -> int | None:
    if segmentation is None or not hasattr(segmentation, "thresholds"):
        return None
    thresholds = np.asarray(segmentation.thresholds, dtype=float)
    if thresholds.size == 0:
        return None
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    if abs(float(thresholds[idx]) - threshold) <= float(tolerance):
        return idx
    return None


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _group_threshold_bbox(
    row: pd.Series,
    components: pd.DataFrame,
    segmentation: Any,
    threshold: float,
    config: dict[str, Any],
) -> tuple[tuple[float, float, float, float] | None, int]:
    idx = _threshold_index(segmentation, threshold, _threshold_match_tolerance(config))
    if idx is None or segmentation is None:
        return None, 0
    component_ids = set(_component_ids(row))
    if not component_ids or components.empty:
        return None, 0
    group_components = components[components["component_index"].astype(int).isin(component_ids)]
    if group_components.empty:
        return None, 0
    labels = segmentation.labels_by_threshold[idx]
    chosen_labels: set[int] = set()
    for _, comp in group_components.iterrows():
        x = int(round(safe_float(comp.get("x"), -1)))
        y = int(round(safe_float(comp.get("y"), -1)))
        if y < 0 or y >= labels.shape[0] or x < 0 or x >= labels.shape[1]:
            continue
        label = int(labels[y, x])
        if label > 0:
            chosen_labels.add(label)
    if not chosen_labels:
        return None, 0
    mask = np.isin(labels, list(chosen_labels))
    bbox = _bbox_from_mask(mask)
    return bbox, int(np.count_nonzero(mask))


def _peak_snr_for_group(row: pd.Series, components: pd.DataFrame, segmentation: Any) -> tuple[float, bool]:
    value = safe_float(row.get("peak_snr"), float("nan"))
    if np.isfinite(value):
        return value, False
    if segmentation is None or not hasattr(segmentation, "snr_map"):
        return float("nan"), True
    ids = set(_component_ids(row))
    if not ids or components.empty:
        return float("nan"), True
    group_components = components[components["component_index"].astype(int).isin(ids)]
    values: list[float] = []
    for _, comp in group_components.iterrows():
        x = int(round(safe_float(comp.get("x"), -1)))
        y = int(round(safe_float(comp.get("y"), -1)))
        if y < 0 or y >= segmentation.snr_map.shape[0] or x < 0 or x >= segmentation.snr_map.shape[1]:
            continue
        values.append(float(segmentation.snr_map[y, x]))
    if not values:
        return float("nan"), True
    return float(np.nanmax(values)), False


def _robust_bbox_for_group(
    row: pd.Series,
    components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    pixel_scale = resolve_pixel_scale_arcsec(row.get("pixel_scale_arcsec"), config, context="parent-seed bounding-box pixel scale")
    bbox3, area3_pix = _group_threshold_bbox(row, components, segmentation, FORMAL_BBOX_THRESHOLDS_SIGMA[0], config)
    if bbox3 is not None:
        return {
            "robust_bbox": bbox3,
            "robust_bbox_source": "3sigma_bbox",
            "area_3sigma_beam": float(area3_pix / max(_beam_area_pix(pixel_scale, config), 1e-6)),
            "missing_area_3sigma_beam": False,
            "missing_robust_bbox": False,
        }
    bbox5, area5_pix = _group_threshold_bbox(row, components, segmentation, FORMAL_BBOX_THRESHOLDS_SIGMA[1], config)
    if bbox5 is not None:
        return {
            "robust_bbox": bbox5,
            "robust_bbox_source": "5sigma_core",
            "area_3sigma_beam": float("nan"),
            "missing_area_3sigma_beam": True,
            "missing_robust_bbox": False,
        }
    fallback = _bbox_tuple(row.get("bounding_box", ""))
    return {
        "robust_bbox": fallback,
        "robust_bbox_source": "fallback_bbox",
        "area_3sigma_beam": float("nan"),
        "missing_area_3sigma_beam": True,
        "missing_robust_bbox": fallback is None,
    }


def _compact_singleton(row: pd.Series, area_3sigma_beam: float, mask_area_beam: float, cfg: dict[str, Any]) -> bool:
    area3 = area_3sigma_beam if np.isfinite(area_3sigma_beam) else 0.0
    compact_cfg = cfg["compact_singleton"]
    return bool(
        int(safe_float(row.get("n_gaussians"), 1.0)) == 1
        and safe_float(row.get("LAS_beam"), 0.0) < float(compact_cfg["max_las_beam"])
        and area3 < float(compact_cfg["max_area_3sigma_beam"])
        and mask_area_beam < float(compact_cfg["max_mask_area_beam"])
    )


def _extended_or_lobe_like(row: pd.Series, mask_area_beam: float, cfg: dict[str, Any]) -> bool:
    allowed_types = {
        "continuous_extended",
        "diffuse_extended",
        "linear_or_tail_like",
        "complex_association",
    }
    return bool(
        safe_float(row.get("n_gaussians"), 1.0) >= float(cfg["min_endpoint_n_gaussians"])
        or safe_float(row.get("LAS_beam"), 0.0) >= float(cfg["min_endpoint_las_beam"])
        or mask_area_beam >= float(cfg["min_endpoint_mask_area_beam"])
        or _association_type(row) in allowed_types
    )


def build_parent_seed_table(
    cutout_id: str,
    groups: pd.DataFrame,
    components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    cfg = parent_seed_config(config)
    if not bool(cfg["enabled"]):
        return pd.DataFrame(columns=PARENT_SEED_COLUMNS)
    records: list[dict[str, Any]] = []
    bad_quality = {"low", "suspicious", "artifact_risk"}
    bad_type = {"weak_association", "artifact_risk"}
    endpoint_thresholds = parent_endpoint_thresholds(config)
    for _, row in groups.iterrows():
        # Keep rejection details for groups that cannot serve as parent endpoints.
        work = row.copy()
        if "pixel_scale_arcsec" not in work:
            work["pixel_scale_arcsec"] = resolve_pixel_scale_arcsec(
                groups.get("pixel_scale_arcsec", pd.Series([None])).iloc[0] if not groups.empty else None,
                config,
                context="parent-seed table pixel scale",
            )
        robust = _robust_bbox_for_group(work, components, segmentation, config)
        peak_snr, missing_peak = _peak_snr_for_group(work, components, segmentation)
        mask_area = _mask_area_beam(work, config)
        area3 = safe_float(robust["area_3sigma_beam"], float("nan"))
        compact = _compact_singleton(work, area3, mask_area, cfg)
        extended = _extended_or_lobe_like(work, mask_area, cfg)
        quality = _association_quality(work)
        atype = _association_type(work)
        singleton_extended = bool(
            int(safe_float(work.get("n_gaussians"), 1.0)) == 1
            and np.isfinite(peak_snr)
            and peak_snr >= float(endpoint_thresholds["noise_peak_snr_min"])
            and np.isfinite(area3)
            and area3 >= float(endpoint_thresholds["noise_area_min_beam"])
            and sum(
                bool(value)
                for value in (
                    safe_float(work.get("LAS_beam"), 0.0) >= float(endpoint_thresholds["extended_las_min_beam"]),
                    area3 >= float(endpoint_thresholds["extended_area_min_beam"]),
                    safe_float(work.get("axis_ratio"), 1.0) >= float(endpoint_thresholds["extended_axis_ratio_min"]),
                )
            ) >= 2
        )
        missing_fields: list[str] = []
        if missing_peak:
            missing_fields.append("peak_snr")
        if bool(robust["missing_area_3sigma_beam"]):
            missing_fields.append("area_3sigma_beam")
        if bool(robust["missing_robust_bbox"]):
            missing_fields.append("robust_bbox")

        reject_reasons: list[str] = []
        if compact:
            reject_reasons.append("compact_singleton")
        if not np.isfinite(peak_snr) or peak_snr < float(cfg["min_parent_seed_peak_snr"]):
            reject_reasons.append(f"peak_snr_below_{cfg['min_parent_seed_peak_snr']:g}")
        if not np.isfinite(area3) or area3 < float(cfg["min_parent_seed_area_3sigma_beam"]):
            reject_reasons.append(f"area_3sigma_beam_below_{cfg['min_parent_seed_area_3sigma_beam']:g}")
        if not extended:
            reject_reasons.append("not_extended_or_lobe_like")
        if quality in bad_quality and not singleton_extended:
            reject_reasons.append(f"bad_quality:{quality}")
        if atype in bad_type and not singleton_extended:
            reject_reasons.append(f"bad_type:{atype}")
        is_seed = not reject_reasons
        reason_parts = [
            f"peak_snr={peak_snr:.2f}" if np.isfinite(peak_snr) else "peak_snr=missing",
            f"area3_beam={area3:.2f}" if np.isfinite(area3) else "area3_beam=missing",
            f"extended={extended}",
            f"quality={quality}",
            f"type={atype}",
            f"bbox={robust['robust_bbox_source']}",
        ]
        records.append(
            {
                "cutout_id": cutout_id,
                "association_group_id": work.get("association_group_id", work.get("local_group_id", "")),
                "n_gaussians": int(safe_float(work.get("n_gaussians"), 1.0)),
                "LAS_beam": safe_float(work.get("LAS_beam"), 0.0),
                "mask_area_beam": mask_area,
                "area_3sigma_beam": area3,
                "peak_snr": peak_snr,
                "association_quality": quality,
                "association_type": atype,
                "is_compact_singleton": bool(compact),
                "is_parent_seed": bool(is_seed),
                "parent_seed_reason": ";".join(reason_parts) if is_seed else "",
                "parent_seed_reject_reason": ";".join(reject_reasons),
                "robust_bbox": _bbox_string(robust["robust_bbox"]),
                "robust_bbox_source": robust["robust_bbox_source"],
                "missing_fields": ",".join(missing_fields),
            }
        )
    return _with_columns(pd.DataFrame(records), PARENT_SEED_COLUMNS)


def _candidate_search_pairs(
    groups: pd.DataFrame,
    max_gap_pix: float,
    *,
    padding_factor: float = float(PARENT_SEED_DEFAULTS["candidate_search_padding_factor"]),
) -> list[tuple[int, int]]:
    if len(groups) < 2:
        return []
    boxes = [_bbox_tuple(row.get("robust_bbox", "")) for _, row in groups.iterrows()]
    centers: list[list[float]] = []
    valid_indices: list[int] = []
    half_diags: list[float] = []
    for idx, box in enumerate(boxes):
        if box is None:
            continue
        x0, y0, x1, y1 = box
        centers.append([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        valid_indices.append(idx)
        half_diags.append(0.5 * float(np.hypot(x1 - x0, y1 - y0)))
    if len(centers) < 2:
        return []
    centers_arr = np.asarray(centers, dtype=float)
    half_diags_arr = np.asarray(half_diags, dtype=float)
    max_half_diag = float(np.nanmax(half_diags_arr)) if half_diags_arr.size else 0.0
    tree = cKDTree(centers_arr)
    candidate_gap_pix = max(float(padding_factor), 1.0) * max_gap_pix
    seen: set[tuple[int, int]] = set()
    for local_i, original_i in enumerate(valid_indices):
        # Use centre distance for recall, then filter by robust bounding-box gap.
        radius = candidate_gap_pix + half_diags_arr[local_i] + max_half_diag + 1.0
        for local_j in tree.query_ball_point(centers_arr[local_i], radius):
            if local_j <= local_i:
                continue
            original_j = int(valid_indices[local_j])
            box_i = boxes[original_i]
            box_j = boxes[original_j]
            if box_i is None or box_j is None:
                continue
            if _box_gap_pix(box_i, box_j) <= candidate_gap_pix:
                seen.add((int(original_i), original_j))
    return sorted(seen)


def _midpoint_core_position(
    group_a: pd.Series,
    group_b: pd.Series,
    all_groups: pd.DataFrame,
    seed_by_id: dict[str, pd.Series],
    config: dict[str, Any],
) -> tuple[float, float] | None:
    beam_arcsec = compute_beam_size_arcsec(config)
    cfg = parent_seed_config(config)
    pixel_scale = resolve_pixel_scale_arcsec(group_a.get("pixel_scale_arcsec"), config, context="midpoint-core pixel scale")
    core_cfg = cfg["midpoint_core"]
    radius_pix = float(core_cfg["search_radius_beam"]) * beam_arcsec / max(pixel_scale, 1e-6)
    x1, y1 = safe_float(group_a.get("centroid_x")), safe_float(group_a.get("centroid_y"))
    x2, y2 = safe_float(group_b.get("centroid_x")), safe_float(group_b.get("centroid_y"))
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return None
    midpoint_x = 0.5 * (x1 + x2)
    midpoint_y = 0.5 * (y1 + y2)
    pair_ids = {str(group_a.get("association_group_id")), str(group_b.get("association_group_id"))}
    axis_dx = x2 - x1
    axis_dy = y2 - y1
    axis_len2 = axis_dx * axis_dx + axis_dy * axis_dy
    nearest: tuple[float, float] | None = None
    nearest_distance = float("inf")
    for _, row in all_groups.iterrows():
        gid = str(row.get("association_group_id", ""))
        if gid in pair_ids:
            continue
        seed_row = seed_by_id.get(gid)
        if seed_row is None or not bool(seed_row.get("is_compact_singleton", False)):
            continue
        cx = safe_float(row.get("centroid_x"))
        cy = safe_float(row.get("centroid_y"))
        if not np.all(np.isfinite([cx, cy])):
            continue
        if float(np.hypot(cx - midpoint_x, cy - midpoint_y)) > radius_pix or axis_len2 <= 0:
            continue
        t = ((cx - x1) * axis_dx + (cy - y1) * axis_dy) / axis_len2
        axis_dist = abs((cx - x1) * axis_dy - (cy - y1) * axis_dx) / max(np.sqrt(axis_len2), 1e-6)
        if (
            float(core_cfg["axial_fraction_min"]) <= t <= float(core_cfg["axial_fraction_max"])
            and axis_dist <= radius_pix
        ):
            midpoint_distance = float(np.hypot(cx - midpoint_x, cy - midpoint_y))
            if midpoint_distance < nearest_distance:
                nearest = (cx, cy)
                nearest_distance = midpoint_distance
    return nearest


def _find_core_candidate_between(
    group_a: pd.Series,
    group_b: pd.Series,
    all_groups: pd.DataFrame,
    seed_by_id: dict[str, pd.Series],
    config: dict[str, Any],
) -> bool:
    """Return whether a compact local group meets the midpoint-core criteria."""

    return _midpoint_core_position(group_a, group_b, all_groups, seed_by_id, config) is not None


def _inward_extent_fraction(
    box: tuple[float, float, float, float] | None,
    centroid_x: float,
    centroid_y: float,
    direction: np.ndarray,
) -> float:
    """Measure the fraction of a bbox extent directed toward the companion."""

    if box is None or not np.all(np.isfinite([centroid_x, centroid_y, *direction])):
        return 0.5
    x0, y0, x1, y1 = box
    corners = np.asarray(((x0, y0), (x0, y1), (x1, y0), (x1, y1)), dtype=float)
    projections = (corners - np.asarray((centroid_x, centroid_y), dtype=float)) @ direction
    inward = max(float(np.nanmax(projections)), 0.0)
    outward = max(float(-np.nanmin(projections)), 0.0)
    extent = inward + outward
    return float(inward / extent) if extent > 1e-6 else 0.5


def _facing_score(
    box_a: tuple[float, float, float, float] | None,
    box_b: tuple[float, float, float, float] | None,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Return inward endpoint extension along the group-to-group axis."""

    axis = np.asarray((x2 - x1, y2 - y1), dtype=float)
    length = float(np.hypot(*axis))
    if not np.isfinite(length) or length <= 1e-6:
        return 0.5
    direction = axis / length
    return float(
        np.mean(
            (
                _inward_extent_fraction(box_a, x1, y1, direction),
                _inward_extent_fraction(box_b, x2, y2, -direction),
            )
        )
    )


def _score_pair(features: dict[str, Any], cfg: dict[str, Any]) -> float:
    evidence = cfg["evidence"]
    max_gap = float(cfg["max_box_gap_beam"])
    gap_score = float(np.clip(1.0 - float(features["box_gap_beam_robust"]) / max(max_gap, 1e-6), 0.0, 1.0))
    missing_ratio_score = float(evidence["missing_ratio_score"])
    flux_score = _ratio_score(float(features["flux_ratio"]), float(evidence["max_flux_ratio"]), missing_ratio_score)
    size_score = _ratio_score(float(features["size_ratio"]), float(evidence["max_size_ratio"]), missing_ratio_score)
    core_score = 1.0 if bool(features["core_candidate_near_midpoint"]) else 0.0
    weights = cfg["score_weights"]
    return float(
        float(weights["gap"]) * gap_score
        + float(weights["axis_alignment"]) * float(features["axis_alignment_score"])
        + float(weights["facing"]) * float(features["facing_score"])
        + float(weights["flux_balance"]) * flux_score
        + float(weights["size_balance"]) * size_score
        + float(weights["midpoint_core"]) * core_score
    )


def _compute_pair(
    cutout_id: str,
    pair_index: int,
    group_a: pd.Series,
    group_b: pd.Series,
    seed_a: pd.Series,
    seed_b: pd.Series,
    all_groups: pd.DataFrame,
    seed_by_id: dict[str, pd.Series],
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = parent_seed_config(config)
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = resolve_pixel_scale_arcsec(
        group_a.get("pixel_scale_arcsec"), config, context="parent-seed pair pixel scale"
    )
    raw_box_a = _bbox_tuple(group_a.get("bounding_box", ""))
    raw_box_b = _bbox_tuple(group_b.get("bounding_box", ""))
    robust_box_a = _bbox_tuple(seed_a.get("robust_bbox", ""))
    robust_box_b = _bbox_tuple(seed_b.get("robust_bbox", ""))
    raw_gap_pix = _box_gap_pix(raw_box_a, raw_box_b) if raw_box_a is not None and raw_box_b is not None else float("inf")
    robust_gap_pix = _box_gap_pix(robust_box_a, robust_box_b) if robust_box_a is not None and robust_box_b is not None else float("inf")
    raw_gap_beam = raw_gap_pix * pixel_scale / max(beam_arcsec, 1e-6)
    robust_gap_arcsec = robust_gap_pix * pixel_scale
    robust_gap_beam = robust_gap_arcsec / max(beam_arcsec, 1e-6)
    x1, y1 = safe_float(group_a.get("centroid_x")), safe_float(group_a.get("centroid_y"))
    x2, y2 = safe_float(group_b.get("centroid_x")), safe_float(group_b.get("centroid_y"))
    center_pix = float(np.hypot(x2 - x1, y2 - y1))
    center_arcsec = center_pix * pixel_scale
    center_beam = center_arcsec / max(beam_arcsec, 1e-6)
    axis_angle = float((np.rad2deg(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0) if np.all(np.isfinite([x1, y1, x2, y2])) else float("nan")
    pa1 = safe_float(group_a.get("group_PA"), float("nan"))
    pa2 = safe_float(group_b.get("group_PA"), float("nan"))
    axis_scores = [_alignment_score(_angle_delta_deg(pa, axis_angle)) for pa in [pa1, pa2] if np.isfinite(pa)]
    axis_alignment = float(np.mean(axis_scores)) if axis_scores else 0.0
    facing = _facing_score(robust_box_a, robust_box_b, x1, y1, x2, y2)
    flux_ratio = _ratio_value(
        safe_float(group_a.get("total_flux_gaussian"), safe_float(group_a.get("peak_flux"), 0.0)),
        safe_float(group_b.get("total_flux_gaussian"), safe_float(group_b.get("peak_flux"), 0.0)),
    )
    size_ratio = _ratio_value(
        max(safe_float(group_a.get("LAS_beam"), 0.0), PARENT_RATIO_FLOOR),
        max(safe_float(group_b.get("LAS_beam"), 0.0), PARENT_RATIO_FLOOR),
    )
    core_position = _midpoint_core_position(group_a, group_b, all_groups, seed_by_id, config)
    core_between = core_position is not None
    if core_position is None:
        midpoint_core_x = float("nan")
        midpoint_core_y = float("nan")
        midpoint_symmetry = 0.5
        midpoint_symmetry_source = "no_midpoint_core"
    else:
        midpoint_core_x, midpoint_core_y = core_position
        distance_a = float(np.hypot(x1 - midpoint_core_x, y1 - midpoint_core_y))
        distance_b = float(np.hypot(x2 - midpoint_core_x, y2 - midpoint_core_y))
        midpoint_symmetry = min(distance_a, distance_b) / max(distance_a, distance_b, 1e-6)
        midpoint_symmetry_source = "compact_midpoint_core"
    both_seed = bool(seed_a.get("is_parent_seed", False)) and bool(seed_b.get("is_parent_seed", False))
    compact_pair = bool(seed_a.get("is_compact_singleton", False)) and bool(seed_b.get("is_compact_singleton", False))
    both_extended = _extended_or_lobe_like(group_a, safe_float(seed_a.get("mask_area_beam"), 0.0), cfg) and _extended_or_lobe_like(group_b, safe_float(seed_b.get("mask_area_beam"), 0.0), cfg)
    support_count = int(axis_alignment >= float(cfg["evidence"]["axis_alignment_min"]))
    support_count += int(facing >= float(cfg["evidence"]["facing_min"]))
    support_count += int(np.isfinite(flux_ratio) and flux_ratio <= float(cfg["evidence"]["max_flux_ratio"]))
    support_count += int(np.isfinite(size_ratio) and size_ratio <= float(cfg["evidence"]["max_size_ratio"]))
    support_count += int(core_between)
    if robust_gap_beam <= float(cfg["strong_box_gap_beam"]):
        support_count += int(both_seed)
    else:
        support_count += int(both_extended)
    source_a = str(seed_a.get("robust_bbox_source") or "fallback_bbox")
    source_b = str(seed_b.get("robust_bbox_source") or "fallback_bbox")
    box_gap_source = source_a if source_a == source_b else f"{source_a}+{source_b}"
    record = {
        "cutout_id": cutout_id,
        "parent_candidate_id": f"{cutout_id}_seed_pc{pair_index:03d}",
        "local_group_id_1": group_a.get("association_group_id"),
        "local_group_id_2": group_b.get("association_group_id"),
        "box_gap_pix": robust_gap_pix,
        "box_gap_arcsec": robust_gap_arcsec,
        "box_gap_beam": robust_gap_beam,
        "box_gap_beam_raw": raw_gap_beam,
        "box_gap_beam_robust": robust_gap_beam,
        "box_gap_source": box_gap_source,
        "center_distance_arcsec": center_arcsec,
        "center_distance_beam": center_beam,
        "n_gaussians_1": int(safe_float(group_a.get("n_gaussians"), 1.0)),
        "n_gaussians_2": int(safe_float(group_b.get("n_gaussians"), 1.0)),
        "LAS_beam_1": safe_float(group_a.get("LAS_beam"), 0.0),
        "LAS_beam_2": safe_float(group_b.get("LAS_beam"), 0.0),
        "mask_area_beam_1": safe_float(seed_a.get("mask_area_beam"), 0.0),
        "mask_area_beam_2": safe_float(seed_b.get("mask_area_beam"), 0.0),
        "area_3sigma_beam_1": safe_float(seed_a.get("area_3sigma_beam"), float("nan")),
        "area_3sigma_beam_2": safe_float(seed_b.get("area_3sigma_beam"), float("nan")),
        "peak_snr_1": safe_float(seed_a.get("peak_snr"), float("nan")),
        "peak_snr_2": safe_float(seed_b.get("peak_snr"), float("nan")),
        "axis_alignment_score": axis_alignment,
        "facing_score": facing,
        "flux_ratio": flux_ratio,
        "size_ratio": size_ratio,
        "core_candidate_near_midpoint": bool(core_between),
        "midpoint_core_x": midpoint_core_x,
        "midpoint_core_y": midpoint_core_y,
        "midpoint_symmetry_score": midpoint_symmetry,
        "midpoint_symmetry_source": midpoint_symmetry_source,
        "parent_axis_angle": axis_angle,
        "group1_PA": pa1,
        "group2_PA": pa2,
        "endpoint1_pass": bool(seed_a.get("is_parent_seed", False)),
        "endpoint2_pass": bool(seed_b.get("is_parent_seed", False)),
        "both_parent_seed": bool(both_seed),
        "both_groups_extended_or_lobe_like": bool(both_extended),
        "compact_singleton_pair": bool(compact_pair),
        "support_evidence_count": int(support_count),
    }
    evidence_cfg = cfg["evidence"]
    strong_required = int(evidence_cfg["strong_gap_min_count"])
    wide_required = int(evidence_cfg["wide_gap_min_count"])
    required_support = strong_required if robust_gap_beam <= float(cfg["strong_box_gap_beam"]) else wide_required
    record["multi_evidence_pass"] = bool(support_count >= required_support)
    record["parent_score"] = _score_pair(record, cfg)

    rejection = ""
    if compact_pair:
        rejection = "compact_singleton_pair"
    elif not both_seed:
        rejection = "endpoint_not_parent_seed"
    elif robust_gap_beam > float(cfg["max_box_gap_beam"]):
        rejection = "robust_box_gap_too_large"
    else:
        if support_count < required_support:
            rejection = "insufficient_evidence_for_parent_link"

    thresholds = cfg["thresholds"]
    score = float(record["parent_score"])
    parent_seed_quality = "rejected"
    default = False
    if not rejection:
        if robust_gap_beam <= float(cfg["strong_box_gap_beam"]) and score >= float(thresholds["high_score"]) and bool(record["multi_evidence_pass"]):
            parent_seed_quality = "high"
            default = True
        elif robust_gap_beam <= float(cfg["max_box_gap_beam"]) and score >= float(thresholds["medium_score"]) and bool(record["multi_evidence_pass"]):
            parent_seed_quality = "medium"
            default = True
        elif score >= float(thresholds["low_score"]):
            parent_seed_quality = "low"
        else:
            parent_seed_quality = "rejected"
            rejection = "score_below_parent_candidate_threshold"
    record["parent_candidate_quality"] = parent_seed_quality
    record["rejection_reason"] = rejection
    record["is_default_candidate"] = bool(default)
    record["needs_visual_check"] = bool(default)
    record["debug_info"] = json_dumps_safe(
        {
            "parent_seed_reject_reason_1": seed_a.get("parent_seed_reject_reason", ""),
            "parent_seed_reject_reason_2": seed_b.get("parent_seed_reject_reason", ""),
            "support_evidence_count": support_count,
            "required_support_count": required_support,
        }
    )
    return record


def _apply_group_limit(edges: pd.DataFrame, max_per_group: int) -> pd.Series:
    if edges.empty:
        return pd.Series(False, index=edges.index)
    selected = pd.Series(False, index=edges.index)
    counts: dict[str, int] = {}
    default_edges = edges[edges["is_default_candidate"].astype(bool)].sort_values(
        ["parent_score", "box_gap_beam_robust"], ascending=[False, True]
    )
    for idx, row in default_edges.iterrows():
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if counts.get(left, 0) >= max_per_group or counts.get(right, 0) >= max_per_group:
            continue
        selected.loc[idx] = True
        counts[left] = counts.get(left, 0) + 1
        counts[right] = counts.get(right, 0) + 1
    return selected


def _empty_result(cutout_id: str, n_groups: int, cfg: dict[str, Any]) -> ParentSeedResult:
    diagnostics = pd.DataFrame(
        [
            {
                "cutout_id": cutout_id,
                "n_local_groups": int(n_groups),
                "n_parent_seed_groups": 0,
                "n_parent_pairs_considered": 0,
                "n_parent_candidates": 0,
                "n_parent_high": 0,
                "n_parent_medium": 0,
                "n_parent_low_debug": 0,
                "n_rejected": 0,
                "n_rejected_compact_singleton_pair": 0,
                "n_rejected_endpoint_not_parent_seed": 0,
                "n_rejected_robust_box_gap_too_large": 0,
                "n_rejected_insufficient_evidence_for_parent_link": 0,
                "n_missing_peak_snr": 0,
                "n_missing_area_3sigma_beam": 0,
                "n_missing_robust_bbox": 0,
                "parent_candidate_overflow_flag": False,
                "n_candidates_before_cutout_limit": 0,
                "max_parent_candidates_per_cutout": int(cfg["max_parent_candidates_per_cutout"]),
                "max_parent_candidates_per_group": int(cfg["max_parent_candidates_per_group"]),
                "max_candidate_box_gap_beam_robust": 0.0,
            }
        ]
    )
    return ParentSeedResult(
        candidates=pd.DataFrame(columns=PARENT_SEED_CANDIDATE_COLUMNS),
        edges_debug=pd.DataFrame(columns=PARENT_SEED_EDGE_DEBUG_COLUMNS),
        diagnostics=_with_columns(diagnostics, PARENT_SEED_DIAGNOSTIC_COLUMNS),
        needs_visual_check=pd.DataFrame(columns=["cutout_id", "record_type", "object_id", "reason", "priority", "details"]),
        parent_seed_table=pd.DataFrame(columns=PARENT_SEED_COLUMNS),
    )


def run_parent_seed(
    cutout_id: str,
    local_groups: pd.DataFrame,
    local_components: pd.DataFrame,
    config: dict[str, Any],
    segmentation: Any | None = None,
) -> ParentSeedResult:
    cfg = parent_seed_config(config)
    if not bool(cfg["enabled"]):
        return _empty_result(cutout_id, len(local_groups), cfg)

    groups = local_groups.copy().reset_index(drop=True)
    if "pixel_scale_arcsec" not in groups:
        if groups.empty:
            groups["pixel_scale_arcsec"] = np.nan
        elif local_components is not None and not local_components.empty and "pixel_scale_arcsec" in local_components:
            groups["pixel_scale_arcsec"] = resolve_pixel_scale_arcsec(
                local_components["pixel_scale_arcsec"].iloc[0], config, context="parent-seed pixel scale"
            )
        else:
            groups["pixel_scale_arcsec"] = resolve_pixel_scale_arcsec(
                None, config, context="parent-seed pixel scale"
            )
    seed_table = build_parent_seed_table(cutout_id, groups, local_components, segmentation, config)
    if groups.empty:
        result = _empty_result(cutout_id, 0, cfg)
        result.parent_seed_table = seed_table
        return result
    seed_by_id = {str(row["association_group_id"]): row for _, row in seed_table.iterrows()}
    groups = groups.merge(seed_table[["association_group_id", "robust_bbox", "robust_bbox_source", "is_parent_seed"]], on="association_group_id", how="left")
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = resolve_pixel_scale_arcsec(groups["pixel_scale_arcsec"].iloc[0], config, context="parent-seed pixel scale")
    max_gap_pix = float(cfg["max_box_gap_beam"]) * beam_arcsec / max(pixel_scale, 1e-6)
    records: list[dict[str, Any]] = []
    for pair_index, (idx_i, idx_j) in enumerate(
        _candidate_search_pairs(
            groups,
            max_gap_pix,
            padding_factor=float(cfg["candidate_search_padding_factor"]),
        )
    ):
        # Compute radio geometry once before host evidence is added.
        group_i = groups.iloc[idx_i]
        group_j = groups.iloc[idx_j]
        seed_i = seed_by_id.get(str(group_i.get("association_group_id")), pd.Series(dtype=object))
        seed_j = seed_by_id.get(str(group_j.get("association_group_id")), pd.Series(dtype=object))
        records.append(_compute_pair(cutout_id, pair_index, group_i, group_j, seed_i, seed_j, groups, seed_by_id, config))
    edges = _with_columns(pd.DataFrame(records), PARENT_SEED_EDGE_DEBUG_COLUMNS)

    if not edges.empty:
        keep_group = _apply_group_limit(edges, int(cfg["max_parent_candidates_per_group"]))
        demote_mask = edges["is_default_candidate"].astype(bool) & ~keep_group
        edges.loc[demote_mask, "is_default_candidate"] = False
        edges.loc[demote_mask, "parent_candidate_quality"] = "low"

    candidates_all = edges[edges["is_default_candidate"].astype(bool)].copy() if not edges.empty else pd.DataFrame(columns=PARENT_SEED_EDGE_DEBUG_COLUMNS)
    candidates_before_cutout_limit = int(len(candidates_all))
    max_per_cutout = int(cfg["max_parent_candidates_per_cutout"])
    overflow = bool(len(candidates_all) > max_per_cutout)
    if overflow:
        kept_idx = candidates_all.sort_values(["parent_score", "box_gap_beam_robust"], ascending=[False, True]).head(max_per_cutout).index
        drop_mask = edges["is_default_candidate"].astype(bool) & ~edges.index.isin(kept_idx)
        edges.loc[drop_mask, "is_default_candidate"] = False
        edges.loc[drop_mask, "parent_candidate_quality"] = "low"
        candidates_all = edges.loc[kept_idx].copy()
    candidates = _with_columns(candidates_all, PARENT_SEED_CANDIDATE_COLUMNS)

    needs_records: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        needs_records.append(
            {
                "cutout_id": cutout_id,
                "record_type": "parent_seed_candidate",
                "object_id": row.get("parent_candidate_id"),
                "reason": "refined parent-seed endpoints with robust bbox gap",
                "priority": row.get("parent_candidate_quality", "medium"),
                "details": json_dumps_safe(
                    {
                        "local_group_id_1": row.get("local_group_id_1"),
                        "local_group_id_2": row.get("local_group_id_2"),
                        "box_gap_beam_robust": row.get("box_gap_beam_robust"),
                        "box_gap_source": row.get("box_gap_source"),
                        "parent_score": row.get("parent_score"),
                    }
                ),
            }
        )
    needs = pd.DataFrame(needs_records, columns=["cutout_id", "record_type", "object_id", "reason", "priority", "details"])
    reason = edges.get("rejection_reason", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    quality = edges.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    missing = seed_table.get("missing_fields", pd.Series(dtype=str)).astype(str) if not seed_table.empty else pd.Series(dtype=str)
    diagnostics = pd.DataFrame(
        [
            {
                "cutout_id": cutout_id,
                "n_local_groups": int(len(groups)),
                "n_parent_seed_groups": int(seed_table.get("is_parent_seed", pd.Series(dtype=bool)).astype(bool).sum()) if not seed_table.empty else 0,
                "n_parent_pairs_considered": int(len(edges)),
                "n_parent_candidates": int(len(candidates)),
                "n_parent_high": int((candidates.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) == "high").sum()) if not candidates.empty else 0,
                "n_parent_medium": int((candidates.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) == "medium").sum()) if not candidates.empty else 0,
                "n_parent_low_debug": int((quality == "low").sum()),
                "n_rejected": int((quality == "rejected").sum()),
                "n_rejected_compact_singleton_pair": int((reason == "compact_singleton_pair").sum()),
                "n_rejected_endpoint_not_parent_seed": int((reason == "endpoint_not_parent_seed").sum()),
                "n_rejected_robust_box_gap_too_large": int((reason == "robust_box_gap_too_large").sum()),
                "n_rejected_insufficient_evidence_for_parent_link": int((reason == "insufficient_evidence_for_parent_link").sum()),
                "n_missing_peak_snr": int(missing.str.contains("peak_snr", na=False).sum()),
                "n_missing_area_3sigma_beam": int(missing.str.contains("area_3sigma_beam", na=False).sum()),
                "n_missing_robust_bbox": int(missing.str.contains("robust_bbox", na=False).sum()),
                "parent_candidate_overflow_flag": overflow,
                "n_candidates_before_cutout_limit": candidates_before_cutout_limit,
                "max_parent_candidates_per_cutout": max_per_cutout,
                "max_parent_candidates_per_group": int(cfg["max_parent_candidates_per_group"]),
                "max_candidate_box_gap_beam_robust": float(pd.to_numeric(candidates.get("box_gap_beam_robust", pd.Series(dtype=float)), errors="coerce").max()) if not candidates.empty else 0.0,
            }
        ]
    )
    return ParentSeedResult(
        candidates=candidates,
        edges_debug=_with_columns(edges, PARENT_SEED_EDGE_DEBUG_COLUMNS),
        diagnostics=_with_columns(diagnostics, PARENT_SEED_DIAGNOSTIC_COLUMNS),
        needs_visual_check=needs,
        parent_seed_table=_with_columns(seed_table, PARENT_SEED_COLUMNS),
    )
