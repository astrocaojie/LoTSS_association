"""Local sanity checks for radio component association.

This module consumes the local Gaussian-level radio association output and keeps
the local-association contract intact: Stage 1/1.5 only decides whether a local
group is internally continuous enough, and optionally splits it by
removing weak or low-support internal edges. It does not make parent-source
associations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ._shared import (
    bbox_from_mask as _bbox_from_mask,
    bbox_from_points as _bbox_from_points,
    ensure_columns as _with_columns,
    las_from_points as _las_from_points,
    line_samples as _line_samples,
    sample_image_nearest as _sample_nearest,
    second_moments as _shared_second_moments,
    support_las as _support_las,
)
from .association import compute_beam_size_arcsec
from .config import (
    FORMAL_SNR_LEVELS,
    LOCAL_BEAM_AREA_GAUSSIAN_FACTOR,
    LOCAL_CHAIN_MIN_COMPONENTS,
    LOCAL_CHAIN_RISK,
    LOCAL_CHAIN_STRONG_EDGE_FRACTION,
    LOCAL_EDGE_BRIDGE_SCORE_MIN,
    LOCAL_EDGE_CUT_RISK_MIN,
    LOCAL_EDGE_DEEP_VALLEY_MIN,
    LOCAL_EDGE_DEEP_VALLEY_RISK,
    LOCAL_EDGE_LOW_SUPPORT_RISK,
    LOCAL_EDGE_ONLY_2SIGMA_RISK,
    LOCAL_EDGE_RIDGE_GAP_RISK,
    LOCAL_EDGE_RIDGE_SCORE_MIN,
    LOCAL_EDGE_SCORE_REFERENCE,
    LOCAL_EDGE_SCORE_RISK_WEIGHT,
    LOCAL_EDGE_WEAK_RISK,
    LOCAL_HIGH_RISK_MIN,
    LOCAL_LARGE_MASK_COMPONENT_COUNT,
    LOCAL_LARGE_MASK_LAS_AREA_FACTOR,
    LOCAL_LARGE_MASK_MIN_AREA_BEAMS,
    LOCAL_LARGE_MASK_ONLY_2SIGMA_FRACTION,
    LOCAL_LARGE_MASK_RISK,
    LOCAL_LARGE_MASK_WEAK_EDGE_FRACTION,
    LOCAL_MEDIUM_RISK_MIN,
    LOCAL_MIN_PAD_PIX,
    LOCAL_MULTI_PEAK_FRACTION,
    LOCAL_MULTI_PEAK_MIN_COUNT,
    LOCAL_MULTI_PEAK_MIN_SEPARATION_BEAM,
    LOCAL_MULTI_PEAK_RISK,
    LOCAL_MULTI_PEAK_SNR_FLOOR,
    LOCAL_NEEDS_CHECK_RISK,
    LOCAL_ONLY_2SIGMA_RISK_SPAN,
    LOCAL_ONLY_2SIGMA_RISK_WEIGHT,
    LOCAL_PAD_BEAM_FACTOR,
    LOCAL_POST_SPLIT_RISK_FACTOR,
    LOCAL_RIDGE_RISK_SPAN,
    LOCAL_RIDGE_RISK_WEIGHT,
    LOCAL_RISK_SCORE_MAX,
    LOCAL_SADDLE_RISK_WEIGHT,
    LOCAL_SANITY_DEFAULTS,
    LOCAL_SCORE_NORMALIZATION_SPAN,
    LOCAL_SPLIT_REQUIRED_RISK,
    LOCAL_WEAK_EDGE_RISK_SPAN,
    LOCAL_WEAK_EDGE_RISK_WEIGHT,
)
from .segmentation import component_support_mask
from .utils import json_dumps_safe, resolve_pixel_scale_arcsec, safe_float

LOCAL_GROUP_COLUMNS = [
    "cutout_id",
    "local_group_id",
    "local_group_index",
    "original_association_group_id",
    "n_gaussians",
    "component_ids",
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
    "group_axis_ratio",
    "local_quality",
    "local_association_type",
    "local_overmerge_risk_score",
    "local_overmerge_risk",
    "split_from_original",
    "split_reason",
    "ridge_gap_fraction",
    "saddle_to_peak_ratio",
    "weak_edge_fraction",
    "only_2sigma_edge_fraction",
    "edge_score_min",
    "edge_score_mean",
    "edge_score_max",
    "n_strong_edges",
    "n_weak_edges",
    "n_only_2sigma_edges",
    "multi_peak_count",
    "largest_peak_separation_beam",
    "large_mask_swallow_flag",
    "chain_merge_flag",
    "needs_visual_check",
    "debug_info",
]

LOCAL_DIAGNOSTIC_COLUMNS = [
    "cutout_id",
    "original_group_id",
    "n_gaussians_before",
    "n_groups_after_split",
    "split_applied",
    "split_reason",
    "local_overmerge_risk_score",
    "saddle_to_peak_ratio",
    "ridge_gap_fraction",
    "weak_edge_fraction",
    "only_2sigma_edge_fraction",
]


@dataclass
class LocalSanityResult:
    groups: pd.DataFrame
    edges: pd.DataFrame
    components: pd.DataFrame
    diagnostics: pd.DataFrame
    needs_visual_check: pd.DataFrame


def _local_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(LOCAL_SANITY_DEFAULTS)
    out.update((config.get("local_association", {}) or {}).get("local_sanity", {}) or {})
    return out


def parse_int_list(value: Any) -> list[int]:
    ids: list[int] = []
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ids
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.append(int(float(item)))
        except (TypeError, ValueError, OverflowError):
            continue
    return ids


def _bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.dtype == bool:
            return values.fillna(False)
        return values.astype(str).str.lower().isin(["true", "1", "yes"])
    return pd.Series(dtype=bool)


def _second_moments(image: np.ndarray, mask: np.ndarray, fallback_x: np.ndarray, fallback_y: np.ndarray) -> tuple[float, float]:
    _major, _minor, pa, axis_ratio = _shared_second_moments(image, mask, fallback_x, fallback_y)
    return pa, axis_ratio


def _accepted_internal_edges(edges: pd.DataFrame, nodes: set[int]) -> pd.DataFrame:
    if edges.empty or len(nodes) < 2:
        return pd.DataFrame()
    decision = _bool_series(edges.get("association_decision", pd.Series(False, index=edges.index)))
    return edges[
        decision
        & edges["component_index_1"].astype(int).isin(nodes)
        & edges["component_index_2"].astype(int).isin(nodes)
    ].copy()


def _all_internal_edges(edges: pd.DataFrame, nodes: set[int]) -> pd.DataFrame:
    if edges.empty or len(nodes) < 2:
        return pd.DataFrame()
    return edges[
        edges["component_index_1"].astype(int).isin(nodes)
        & edges["component_index_2"].astype(int).isin(nodes)
    ].copy()


def _peak_snr(row: pd.Series, snr_map: np.ndarray) -> float:
    x = safe_float(row.get("x"))
    y = safe_float(row.get("y"))
    if not np.isfinite(x) or not np.isfinite(y):
        return safe_float(row.get("_peak_flux"), 0.0)
    values = _sample_nearest(snr_map, np.asarray([x]), np.asarray([y]))
    value = float(values[0]) if len(values) else float("nan")
    if np.isfinite(value):
        return value
    return safe_float(row.get("_peak_flux"), 0.0)


def _peak_path_features(group_rows: pd.DataFrame, snr_map: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    local_cfg = _local_config(config)
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = resolve_pixel_scale_arcsec(
        group_rows["pixel_scale_arcsec"].iloc[0] if "pixel_scale_arcsec" in group_rows else None,
        config,
        context="local-sanity pixel scale",
    )
    if len(group_rows) <= 1:
        return {
            "saddle_to_peak_ratio": 1.0,
            "ridge_gap_fraction": 0.0,
            "multi_peak_count": int(len(group_rows)),
            "largest_peak_separation_beam": 0.0,
        }

    work = group_rows.copy()
    work["_local_peak_snr"] = work.apply(lambda row: _peak_snr(row, snr_map), axis=1)
    peak_values = pd.to_numeric(work["_local_peak_snr"], errors="coerce").fillna(0.0)
    max_peak = float(peak_values.max()) if len(peak_values) else 0.0
    if max_peak <= 0:
        multi_peak_count = int(len(work))
    else:
        multi_peak_count = int(
            (peak_values >= max(LOCAL_MULTI_PEAK_SNR_FLOOR, LOCAL_MULTI_PEAK_FRACTION * max_peak)).sum()
        )
    candidates = work.sort_values("_local_peak_snr", ascending=False).head(8)
    if len(candidates) < 2:
        return {
            "saddle_to_peak_ratio": 1.0,
            "ridge_gap_fraction": 0.0,
            "multi_peak_count": max(1, multi_peak_count),
            "largest_peak_separation_beam": 0.0,
        }

    saddle_ratios: list[float] = []
    gap_fractions: list[float] = []
    largest_sep = 0.0
    rows = list(candidates.iterrows())
    weak_threshold = min(
        FORMAL_SNR_LEVELS["strong"],
        max(
            FORMAL_SNR_LEVELS["intermediate"],
            float(local_cfg["min_path_snr"]),
        ),
    )
    for idx, (_, left) in enumerate(rows):
        for _, right in rows[idx + 1 :]:
            x1, y1 = safe_float(left.get("x")), safe_float(left.get("y"))
            x2, y2 = safe_float(right.get("x")), safe_float(right.get("y"))
            if not np.all(np.isfinite([x1, y1, x2, y2])):
                continue
            sep_beam = float(np.hypot(x2 - x1, y2 - y1) * pixel_scale / max(beam_arcsec, 1e-6))
            largest_sep = max(largest_sep, sep_beam)
            if sep_beam < 1.0:
                continue
            xs, ys = _line_samples(x1, y1, x2, y2)
            values = _sample_nearest(snr_map, xs, ys)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            peak_floor = min(max(float(left["_local_peak_snr"]), 0.0), max(float(right["_local_peak_snr"]), 0.0))
            if peak_floor <= 0:
                continue
            saddle_ratios.append(float(np.nanmin(finite) / max(peak_floor, 1e-6)))
            gap_fractions.append(float(np.mean(finite < weak_threshold)))

    return {
        "saddle_to_peak_ratio": float(np.clip(min(saddle_ratios) if saddle_ratios else 1.0, -1.0, 1.5)),
        "ridge_gap_fraction": float(np.clip(max(gap_fractions) if gap_fractions else 0.0, 0.0, 1.0)),
        "multi_peak_count": int(max(1, multi_peak_count)),
        "largest_peak_separation_beam": float(largest_sep),
    }


def compute_local_group_features(
    group_row: pd.Series,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    segmentation: Any,
    image: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute overmerge sanity features for one local association group."""

    nodes = set(parse_int_list(group_row.get("component_ids")))
    if not nodes and "association_group_id" in components:
        group_id = str(group_row.get("association_group_id", ""))
        nodes = set(components.loc[components["association_group_id"].astype(str) == group_id, "component_index"].astype(int).tolist())
    group_rows = components[components["component_index"].astype(int).isin(nodes)].copy()
    n_gaussians = int(len(group_rows))
    accepted_edges = _accepted_internal_edges(edges, nodes)
    all_internal_edges = _all_internal_edges(edges, nodes)

    scores = pd.to_numeric(accepted_edges.get("association_score", pd.Series(dtype=float)), errors="coerce").dropna()
    n_edges = int(len(accepted_edges))
    n_strong = int((accepted_edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong").sum()) if n_edges else 0
    n_weak = int((accepted_edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak").sum()) if n_edges else 0
    n_only_2 = int(_bool_series(accepted_edges.get("only_2sigma_connected", pd.Series(dtype=bool))).sum()) if n_edges else 0
    weak_fraction = float(n_weak / max(n_edges, 1))
    only_2_fraction = float(n_only_2 / max(n_edges, 1))
    edge_ridge_gaps = pd.to_numeric(accepted_edges.get("ridge_gap_fraction", pd.Series(dtype=float)), errors="coerce").dropna()

    path_features = _peak_path_features(group_rows, segmentation.snr_map, config) if n_gaussians else {
        "saddle_to_peak_ratio": 1.0,
        "ridge_gap_fraction": 0.0,
        "multi_peak_count": 0,
        "largest_peak_separation_beam": 0.0,
    }
    ridge_gap_fraction = float(max(path_features["ridge_gap_fraction"], edge_ridge_gaps.mean() if len(edge_ridge_gaps) else 0.0))

    las_beam = safe_float(group_row.get("LAS_beam"), 0.0)
    axis_ratio = safe_float(group_row.get("axis_ratio"), safe_float(group_row.get("group_axis_ratio"), 1.0))
    local_cfg = _local_config(config)

    support_area_beam = 0.0
    large_mask_swallow = False
    if n_gaussians and getattr(segmentation, "labels_by_threshold", None) is not None:
        support_2 = component_support_mask(
            segmentation.labels_by_threshold,
            segmentation.thresholds,
            group_rows,
            FORMAL_SNR_LEVELS["weak"],
        )
        pixel_scale = resolve_pixel_scale_arcsec(
            group_rows["pixel_scale_arcsec"].iloc[0] if "pixel_scale_arcsec" in group_rows else None,
            config,
            context="local-sanity pixel scale",
        )
        beam_arcsec = compute_beam_size_arcsec(config)
        beam_area_pix = np.pi * (beam_arcsec / max(pixel_scale, 1e-6)) ** 2 / (
            LOCAL_BEAM_AREA_GAUSSIAN_FACTOR * np.log(2.0)
        )
        support_area_beam = float(np.count_nonzero(support_2) / max(beam_area_pix, 1e-6))
        large_mask_swallow = bool(
            support_area_beam
            > max(LOCAL_LARGE_MASK_MIN_AREA_BEAMS, LOCAL_LARGE_MASK_LAS_AREA_FACTOR * max(las_beam, 1.0) ** 2)
            and (
                weak_fraction > LOCAL_LARGE_MASK_WEAK_EDGE_FRACTION
                or only_2_fraction > LOCAL_LARGE_MASK_ONLY_2SIGMA_FRACTION
                or n_gaussians > LOCAL_LARGE_MASK_COMPONENT_COUNT
            )
        )
    flags = str(group_row.get("artifact_risk_flags", ""))
    if "large_mask_swallow" in flags:
        large_mask_swallow = True

    chain_merge = bool(
        weak_fraction > float(local_cfg["max_weak_edge_chain_fraction"])
        or only_2_fraction > float(local_cfg["max_only_2sigma_edge_fraction"])
        or (
            n_edges > 0
            and n_strong <= max(1, int(LOCAL_CHAIN_STRONG_EDGE_FRACTION * n_edges))
            and n_gaussians >= LOCAL_CHAIN_MIN_COMPONENTS
        )
    )

    saddle_ratio = float(path_features["saddle_to_peak_ratio"])
    multi_peak_count = int(path_features["multi_peak_count"])
    largest_peak_sep = float(path_features["largest_peak_separation_beam"])

    risk = 0.0
    needs_check = bool(
        las_beam >= float(local_cfg["max_group_las_beam_before_check"])
        or n_gaussians >= int(local_cfg["max_group_n_gaussians_before_check"])
    )
    if needs_check:
        risk += LOCAL_NEEDS_CHECK_RISK
    risk += LOCAL_RIDGE_RISK_WEIGHT * max(
        0.0,
        (ridge_gap_fraction - float(local_cfg["max_ridge_gap_fraction"])) / LOCAL_RIDGE_RISK_SPAN,
    )
    risk += LOCAL_SADDLE_RISK_WEIGHT * max(
        0.0,
        (float(local_cfg["min_saddle_to_peak_ratio"]) - saddle_ratio)
        / max(float(local_cfg["min_saddle_to_peak_ratio"]), 1e-6),
    )
    risk += LOCAL_WEAK_EDGE_RISK_WEIGHT * max(
        0.0,
        (weak_fraction - float(local_cfg["max_weak_edge_chain_fraction"])) / LOCAL_WEAK_EDGE_RISK_SPAN,
    )
    risk += LOCAL_ONLY_2SIGMA_RISK_WEIGHT * max(
        0.0,
        (only_2_fraction - float(local_cfg["max_only_2sigma_edge_fraction"]))
        / LOCAL_ONLY_2SIGMA_RISK_SPAN,
    )
    risk += LOCAL_CHAIN_RISK if chain_merge else 0.0
    risk += LOCAL_LARGE_MASK_RISK if large_mask_swallow else 0.0
    if (
        multi_peak_count >= LOCAL_MULTI_PEAK_MIN_COUNT
        and largest_peak_sep >= LOCAL_MULTI_PEAK_MIN_SEPARATION_BEAM
        and saddle_ratio < float(local_cfg["min_saddle_to_peak_ratio"])
    ):
        risk += LOCAL_MULTI_PEAK_RISK
    risk = float(np.clip(risk, 0.0, LOCAL_RISK_SCORE_MAX))

    return {
        "cutout_id": group_row.get("cutout_id"),
        "original_association_group_id": group_row.get("association_group_id", group_row.get("original_association_group_id", "")),
        "n_gaussians": n_gaussians,
        "LAS_beam": las_beam,
        "group_axis_ratio": axis_ratio,
        "n_strong_edges": n_strong,
        "n_weak_edges": n_weak,
        "n_only_2sigma_edges": n_only_2,
        "weak_edge_fraction": weak_fraction,
        "only_2sigma_edge_fraction": only_2_fraction,
        "edge_score_min": float(scores.min()) if len(scores) else 0.0,
        "edge_score_mean": float(scores.mean()) if len(scores) else 0.0,
        "edge_score_max": float(scores.max()) if len(scores) else 0.0,
        "ridge_gap_fraction": ridge_gap_fraction,
        "saddle_to_peak_ratio": saddle_ratio,
        "multi_peak_count": multi_peak_count,
        "largest_peak_separation_beam": largest_peak_sep,
        "large_mask_swallow_flag": large_mask_swallow,
        "chain_merge_flag": chain_merge,
        "local_overmerge_risk_score": risk,
        "needs_check": needs_check,
        "accepted_edges": accepted_edges,
        "all_internal_edges": all_internal_edges,
        "component_nodes": sorted(nodes),
        "support_area_beam": support_area_beam,
    }


def _edge_cut_risk(edge: pd.Series, config: dict[str, Any]) -> float:
    local_cfg = _local_config(config)
    score = safe_float(edge.get("association_score"), 0.0)
    score_risk = max(
        0.0,
        (float(local_cfg["min_split_score_gain"]) + LOCAL_EDGE_SCORE_REFERENCE - score)
        / LOCAL_EDGE_SCORE_REFERENCE,
    )
    risk = 0.0
    risk += LOCAL_EDGE_WEAK_RISK if str(edge.get("edge_type", "")).lower() == "weak" else 0.0
    risk += (
        LOCAL_EDGE_ONLY_2SIGMA_RISK
        if str(edge.get("only_2sigma_connected", "")).lower() in {"true", "1", "yes"}
        else 0.0
    )
    risk += (
        LOCAL_EDGE_RIDGE_GAP_RISK
        if safe_float(edge.get("ridge_gap_fraction"), 0.0) > float(local_cfg["max_ridge_gap_fraction"])
        else 0.0
    )
    risk += (
        LOCAL_EDGE_DEEP_VALLEY_RISK
        if safe_float(edge.get("deep_valley_penalty"), 0.0) >= LOCAL_EDGE_DEEP_VALLEY_MIN
        else 0.0
    )
    risk += (
        LOCAL_EDGE_LOW_SUPPORT_RISK
        if safe_float(edge.get("bridge_score"), 0.0) < LOCAL_EDGE_BRIDGE_SCORE_MIN
        and safe_float(edge.get("ridge_continuity_score"), 0.0) < LOCAL_EDGE_RIDGE_SCORE_MIN
        else 0.0
    )
    risk += LOCAL_EDGE_SCORE_RISK_WEIGHT * score_risk
    return float(risk)


def _subgroup_gain(original_features: dict[str, Any], subgroups: list[list[int]], group_rows: pd.DataFrame, edges: pd.DataFrame) -> float:
    if len(subgroups) < 2:
        return 0.0
    original_risk = float(original_features.get("local_overmerge_risk_score", 0.0))
    if group_rows.empty:
        return 0.0
    size_penalty = max(len(values) for values in subgroups) / max(len(group_rows), 1)
    kept_scores: list[float] = []
    for subgroup in subgroups:
        internal = _accepted_internal_edges(edges, set(subgroup))
        scores = pd.to_numeric(internal.get("association_score", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(scores):
            kept_scores.append(float(scores.mean()))
    score_bonus = max(
        0.0,
        (np.mean(kept_scores) if kept_scores else 0.0)
        - float(original_features.get("edge_score_mean", 0.0)),
    ) / LOCAL_SCORE_NORMALIZATION_SPAN
    return float(original_risk * (1.0 - size_penalty) + score_bonus)


def split_overmerged_local_group(
    group_row: pd.Series,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    features: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[list[int]], str, bool]:
    """Split a suspicious local group by pruning weak or low-support edges."""

    local_cfg = _local_config(config)
    nodes = [int(value) for value in features.get("component_nodes", [])]
    if len(nodes) <= 1:
        return [nodes], "", False
    risk = float(features.get("local_overmerge_risk_score", 0.0))
    overmerge_condition = bool(
        risk >= LOCAL_SPLIT_REQUIRED_RISK
        and (
            features.get("chain_merge_flag")
            or features.get("large_mask_swallow_flag")
            or float(features.get("ridge_gap_fraction", 0.0)) > float(local_cfg["max_ridge_gap_fraction"])
            or float(features.get("saddle_to_peak_ratio", 1.0)) < float(local_cfg["min_saddle_to_peak_ratio"])
        )
    )
    if not overmerge_condition or not bool(local_cfg["split_overmerged_groups"]):
        return [nodes], "", False

    accepted = features.get("accepted_edges", pd.DataFrame()).copy()
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    cut_edges: list[tuple[int, int, float]] = []
    kept_edges = 0
    for _, edge in accepted.iterrows():
        left = int(edge["component_index_1"])
        right = int(edge["component_index_2"])
        cut_risk = _edge_cut_risk(edge, config)
        if cut_risk >= LOCAL_EDGE_CUT_RISK_MIN:
            cut_edges.append((left, right, cut_risk))
            continue
        graph.add_edge(left, right)
        kept_edges += 1

    subgroups = [sorted(list(values)) for values in nx.connected_components(graph)]
    subgroups.sort(key=lambda values: (len(values), -values[0] if values else 0), reverse=True)
    min_size = int(local_cfg["min_subgroup_size"])
    if len(subgroups) < 2 or any(len(values) < min_size for values in subgroups):
        return [nodes], "unsplittable_no_stable_components", False

    group_rows = components[components["component_index"].astype(int).isin(nodes)].copy()
    gain = _subgroup_gain(features, subgroups, group_rows, edges)
    if gain < float(local_cfg["min_split_score_gain"]) and kept_edges > 0:
        return [nodes], f"split_gain_too_low:{gain:.2f}", False

    reasons = []
    if features.get("chain_merge_flag"):
        reasons.append("weak_edge_chain")
    if features.get("large_mask_swallow_flag"):
        reasons.append("large_mask_swallow")
    if float(features.get("ridge_gap_fraction", 0.0)) > float(local_cfg["max_ridge_gap_fraction"]):
        reasons.append("ridge_gap")
    if float(features.get("saddle_to_peak_ratio", 1.0)) < float(local_cfg["min_saddle_to_peak_ratio"]):
        reasons.append("low_saddle")
    reasons.append(f"cut_edges={len(cut_edges)}")
    return subgroups, ";".join(reasons), True


def _measure_local_group(
    cutout_id: str,
    original_group: pd.Series,
    local_group_id: str,
    local_index: int,
    nodes: list[int],
    components: pd.DataFrame,
    edges: pd.DataFrame,
    segmentation: Any,
    image: np.ndarray,
    config: dict[str, Any],
    original_features: dict[str, Any],
    split_applied: bool,
    split_reason: str,
    unsplittable_reason: str = "",
    wcs: Any | None = None,
) -> dict[str, Any]:
    node_set = set(int(value) for value in nodes)
    group_rows = components[components["component_index"].astype(int).isin(node_set)].copy()
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = resolve_pixel_scale_arcsec(
        group_rows["pixel_scale_arcsec"].iloc[0] if not group_rows.empty and "pixel_scale_arcsec" in group_rows else None,
        config,
        context="local-sanity pixel scale",
    )
    padding = max(LOCAL_MIN_PAD_PIX, int(round(LOCAL_PAD_BEAM_FACTOR * beam_arcsec / max(pixel_scale, 1e-6))))
    x = pd.to_numeric(group_rows.get("x", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    y = pd.to_numeric(group_rows.get("y", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    support_2 = (
        component_support_mask(
            segmentation.labels_by_threshold,
            segmentation.thresholds,
            group_rows,
            FORMAL_SNR_LEVELS["weak"],
        )
        if not group_rows.empty
        else np.zeros_like(segmentation.snr_map, dtype=bool)
    )
    support_25 = (
        component_support_mask(
            segmentation.labels_by_threshold,
            segmentation.thresholds,
            group_rows,
            FORMAL_SNR_LEVELS["intermediate"],
        )
        if not group_rows.empty
        else np.zeros_like(segmentation.snr_map, dtype=bool)
    )
    support = support_25 if support_25.any() else support_2
    bbox = _bbox_from_mask(support) if support.any() else _bbox_from_points(x, y, padding, image.shape)
    if bbox is None:
        bbox = _bbox_from_points(x, y, padding, image.shape)
    x0, y0, x1, y1 = bbox
    weights = pd.to_numeric(group_rows.get("_peak_flux", pd.Series(dtype=float)), errors="coerce").to_numpy(float, copy=True)
    weights[~np.isfinite(weights) | (weights <= 0)] = 1.0
    centroid_x = float(np.average(x, weights=weights)) if len(x) else float("nan")
    centroid_y = float(np.average(y, weights=weights)) if len(y) else float("nan")
    las_pix, las_arcsec = _las_from_points(x, y, pixel_scale)
    if support.any():
        las_pix, las_arcsec = _support_las(support, pixel_scale, las_pix)
    group_pa, axis_ratio = _second_moments(image, support, x, y)

    local_features = compute_local_group_features(
        pd.Series(
            {
                "cutout_id": cutout_id,
                "association_group_id": original_group.get("association_group_id", ""),
                "component_ids": ",".join(map(str, sorted(node_set))),
                "LAS_beam": float(las_arcsec / max(beam_arcsec, 1e-6)),
                "axis_ratio": axis_ratio,
                "artifact_risk_flags": original_group.get("artifact_risk_flags", ""),
            }
        ),
        components,
        edges,
        segmentation,
        image,
        config,
    )
    if split_applied:
        local_features["local_overmerge_risk_score"] = min(
            float(local_features["local_overmerge_risk_score"]),
            LOCAL_POST_SPLIT_RISK_FACTOR * float(original_features.get("local_overmerge_risk_score", 0.0)),
        )

    risk = float(local_features.get("local_overmerge_risk_score", 0.0))
    high_risk = risk >= LOCAL_HIGH_RISK_MIN
    quality = str(original_group.get("association_quality", "low"))
    if "artifact" in str(original_group.get("association_type", "")) or "sidelobe" in str(original_group.get("artifact_risk_flags", "")):
        quality = "artifact_risk"
    elif high_risk and not split_applied:
        quality = "suspicious"
    elif split_applied and quality == "suspicious" and risk < LOCAL_SPLIT_REQUIRED_RISK:
        quality = "medium"
    needs_visual = bool(high_risk or quality in {"suspicious", "artifact_risk"} or split_applied)
    association_type = str(original_group.get("association_type", "weak_association"))
    if quality == "artifact_risk":
        association_type = "artifact_risk"

    total_flux = float(np.nansum(pd.to_numeric(group_rows.get("_total_flux", pd.Series(dtype=float)), errors="coerce"))) if not group_rows.empty else 0.0
    peak_flux = float(np.nanmax(pd.to_numeric(group_rows.get("_peak_flux", pd.Series(dtype=float)), errors="coerce"))) if not group_rows.empty else 0.0
    ra = safe_float(original_group.get("ra"), float("nan"))
    dec = safe_float(original_group.get("dec"), float("nan"))
    if wcs is not None and np.isfinite(centroid_x) and np.isfinite(centroid_y):
        # Split subgroups move the centroid; reproject it instead of inheriting
        # the original group's sky position.
        try:
            ra_values, dec_values = wcs.celestial.pixel_to_world_values(centroid_x, centroid_y)
            ra, dec = float(ra_values), float(dec_values)
        except Exception as exc:
            raise ValueError(
                f"failed to convert local-group centroid to sky coordinates for {cutout_id}"
            ) from exc
    return {
        "cutout_id": cutout_id,
        "local_group_id": local_group_id,
        "local_group_index": int(local_index),
        "original_association_group_id": original_group.get("association_group_id", ""),
        "n_gaussians": int(len(group_rows)),
        "component_ids": ",".join(map(str, sorted(node_set))),
        "gaussian_ids": ",".join(map(str, group_rows.get("_gaussian_id", pd.Series(dtype=int)).tolist())),
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
        "group_axis_ratio": axis_ratio,
        "local_quality": quality,
        "local_association_type": association_type,
        "local_overmerge_risk_score": risk,
        "local_overmerge_risk": (
            "high" if risk >= LOCAL_HIGH_RISK_MIN else ("medium" if risk >= LOCAL_MEDIUM_RISK_MIN else "low")
        ),
        "split_from_original": bool(split_applied),
        "split_reason": split_reason or unsplittable_reason,
        "ridge_gap_fraction": float(local_features.get("ridge_gap_fraction", 0.0)),
        "saddle_to_peak_ratio": float(local_features.get("saddle_to_peak_ratio", 1.0)),
        "weak_edge_fraction": float(local_features.get("weak_edge_fraction", 0.0)),
        "only_2sigma_edge_fraction": float(local_features.get("only_2sigma_edge_fraction", 0.0)),
        "edge_score_min": float(local_features.get("edge_score_min", 0.0)),
        "edge_score_mean": float(local_features.get("edge_score_mean", 0.0)),
        "edge_score_max": float(local_features.get("edge_score_max", 0.0)),
        "n_strong_edges": int(local_features.get("n_strong_edges", 0)),
        "n_weak_edges": int(local_features.get("n_weak_edges", 0)),
        "n_only_2sigma_edges": int(local_features.get("n_only_2sigma_edges", 0)),
        "multi_peak_count": int(local_features.get("multi_peak_count", 0)),
        "largest_peak_separation_beam": float(local_features.get("largest_peak_separation_beam", 0.0)),
        "large_mask_swallow_flag": bool(local_features.get("large_mask_swallow_flag", False)),
        "chain_merge_flag": bool(local_features.get("chain_merge_flag", False)),
        "needs_visual_check": needs_visual,
        "debug_info": json_dumps_safe(
            {
                "original_features": {
                    key: value
                    for key, value in original_features.items()
                    if key not in {"accepted_edges", "all_internal_edges"}
                },
                "support_pixels_2sigma": int(support_2.sum()),
                "support_pixels_2p5sigma": int(support_25.sum()),
            }
        ),
    }


def run_local_sanity(
    cutout_id: str,
    groups: pd.DataFrame,
    edges: pd.DataFrame,
    components: pd.DataFrame,
    segmentation: Any,
    image: np.ndarray,
    config: dict[str, Any],
    wcs: Any | None = None,
) -> LocalSanityResult:
    """Run Stage 1.5 local sanity checks for one cutout."""

    local_cfg = _local_config(config)
    groups = groups.copy()
    edges = edges.copy()
    components = components.copy()
    if groups.empty:
        return LocalSanityResult(
            groups=pd.DataFrame(columns=LOCAL_GROUP_COLUMNS),
            edges=edges,
            components=components,
            diagnostics=pd.DataFrame(columns=LOCAL_DIAGNOSTIC_COLUMNS),
            needs_visual_check=pd.DataFrame(),
        )

    records: list[dict[str, Any]] = []
    diag_records: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []
    node_to_local: dict[int, str] = {}
    local_rows_by_node: dict[int, dict[str, Any]] = {}

    local_index = 0
    for _, original_group in groups.sort_values("association_group_index" if "association_group_index" in groups else "association_group_id").iterrows():
        features = compute_local_group_features(original_group, components, edges, segmentation, image, config)
        subgroups, split_reason, split_applied = split_overmerged_local_group(original_group, components, edges, features, config)
        unsplittable_reason = "" if split_applied else split_reason
        if not split_applied and unsplittable_reason and bool(local_cfg["mark_suspicious_if_unsplittable"]):
            features["local_overmerge_risk_score"] = max(
                float(features.get("local_overmerge_risk_score", 0.0)),
                LOCAL_HIGH_RISK_MIN,
            )

        for subgroup_nodes in subgroups:
            local_group_id = (
                str(original_group.get("association_group_id"))
                if not split_applied and len(subgroups) == 1
                else f"{cutout_id}_l{local_index:03d}"
            )
            row = _measure_local_group(
                cutout_id,
                original_group,
                local_group_id,
                local_index,
                subgroup_nodes,
                components,
                edges,
                segmentation,
                image,
                config,
                features,
                split_applied=split_applied,
                split_reason=split_reason if split_applied else "",
                unsplittable_reason=unsplittable_reason,
                wcs=wcs,
            )
            if not split_applied and unsplittable_reason and bool(local_cfg["mark_suspicious_if_unsplittable"]):
                row["local_quality"] = "suspicious"
                row["local_overmerge_risk"] = "high"
                row["needs_visual_check"] = True
            records.append(row)
            for node in subgroup_nodes:
                node_to_local[int(node)] = local_group_id
                local_rows_by_node[int(node)] = row
            local_index += 1

        diag_records.append(
            {
                "cutout_id": cutout_id,
                "original_group_id": original_group.get("association_group_id", ""),
                "n_gaussians_before": int(features.get("n_gaussians", 0)),
                "n_groups_after_split": int(len(subgroups)),
                "split_applied": bool(split_applied),
                "split_reason": split_reason or unsplittable_reason,
                "local_overmerge_risk_score": float(features.get("local_overmerge_risk_score", 0.0)),
                "saddle_to_peak_ratio": float(features.get("saddle_to_peak_ratio", 1.0)),
                "ridge_gap_fraction": float(features.get("ridge_gap_fraction", 0.0)),
                "weak_edge_fraction": float(features.get("weak_edge_fraction", 0.0)),
                "only_2sigma_edge_fraction": float(features.get("only_2sigma_edge_fraction", 0.0)),
            }
        )

    local_groups = _with_columns(pd.DataFrame(records), LOCAL_GROUP_COLUMNS)

    components["original_association_group_id"] = components.get("association_group_id", "")
    components["local_group_id"] = components["component_index"].astype(int).map(node_to_local).fillna("")
    components["local_group_index"] = components["local_group_id"].map(
        local_groups.set_index("local_group_id")["local_group_index"].to_dict() if not local_groups.empty else {}
    ).fillna(-1).astype(int)
    components["local_group_size"] = components["local_group_id"].map(
        local_groups.set_index("local_group_id")["n_gaussians"].to_dict() if not local_groups.empty else {}
    ).fillna(1).astype(int)
    components["local_quality"] = components["local_group_id"].map(
        local_groups.set_index("local_group_id")["local_quality"].to_dict() if not local_groups.empty else {}
    ).fillna("low")
    components["local_association_type"] = components["local_group_id"].map(
        local_groups.set_index("local_group_id")["local_association_type"].to_dict() if not local_groups.empty else {}
    ).fillna("weak_association")
    components["local_overmerge_risk_score"] = components["local_group_id"].map(
        local_groups.set_index("local_group_id")["local_overmerge_risk_score"].to_dict() if not local_groups.empty else {}
    ).fillna(0.0)
    components["needs_visual_check"] = components["local_group_id"].map(
        local_groups.set_index("local_group_id")["needs_visual_check"].to_dict() if not local_groups.empty else {}
    ).fillna(False).astype(bool)

    if not edges.empty:
        edges["local_group_id_1"] = edges["component_index_1"].astype(int).map(node_to_local).fillna("")
        edges["local_group_id_2"] = edges["component_index_2"].astype(int).map(node_to_local).fillna("")
        same_local = (edges["local_group_id_1"] != "") & (edges["local_group_id_1"] == edges["local_group_id_2"])
        original_decision = _bool_series(edges.get("association_decision", pd.Series(False, index=edges.index)))
        edges["local_edge_decision"] = same_local & original_decision
        edges["local_edge_type"] = edges.get("edge_type", "").astype(str)
        cut_mask = original_decision & ~same_local
        edges.loc[cut_mask, "local_edge_type"] = "cut"
        edges["local_rejection_reason"] = edges.get("rejection_reason", "").fillna("").astype(str)
        edges.loc[cut_mask, "local_rejection_reason"] = "local_sanity_split"
        edges["association_decision"] = edges["local_edge_decision"].astype(bool)
        edges.loc[cut_mask, "edge_type"] = "rejected"
        edges.loc[cut_mask, "rejection_reason"] = "local_sanity_split"

    for _, row in local_groups[local_groups["needs_visual_check"].astype(bool)].iterrows():
        reasons = []
        if str(row.get("local_quality")) == "suspicious":
            reasons.append("suspicious local overmerge")
        if bool(row.get("split_from_original")):
            reasons.append("local group split applied")
        if bool(row.get("large_mask_swallow_flag")):
            reasons.append("dense/crowded region")
        if bool(row.get("chain_merge_flag")):
            reasons.append("weak-chain large group")
        visual_records.append(
            {
                "cutout_id": cutout_id,
                "record_type": "local_group",
                "object_id": row.get("local_group_id"),
                "reason": "; ".join(reasons) if reasons else "suspicious local overmerge",
                "priority": "high" if str(row.get("local_overmerge_risk")) == "high" else "medium",
                "details": json_dumps_safe(
                    {
                        "original_association_group_id": row.get("original_association_group_id"),
                        "local_overmerge_risk_score": row.get("local_overmerge_risk_score"),
                        "split_reason": row.get("split_reason"),
                    }
                ),
            }
        )

    return LocalSanityResult(
        groups=_with_columns(local_groups, LOCAL_GROUP_COLUMNS),
        edges=edges,
        components=components,
        diagnostics=_with_columns(pd.DataFrame(diag_records), LOCAL_DIAGNOSTIC_COLUMNS),
        needs_visual_check=pd.DataFrame(visual_records),
    )
