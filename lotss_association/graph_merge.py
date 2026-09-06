"""Gaussian component graph construction and rule-based merging."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ._shared import (
    line_samples as _line_samples,
    sample_image_nearest as _sample_image_nearest,
)
from .beam import beam_axes_from_config
from .config import (
    BRIDGE_BASELINE_SNR,
    BRIDGE_SCORE_SNR_SPAN,
    FORMAL_SNR_LEVELS,
    GRAPH_COMPACT_PAIR_MAX_DISTANCE_BEAM,
    GRAPH_COMPACT_PAIR_MAX_FLUX_RATIO,
    GRAPH_COMPACT_PAIR_PENALTY,
    GRAPH_DEFAULTS,
    GRAPH_EVIDENCE_DISPLAY_MIN,
    GRAPH_NEGATIVE_BOWL_WEIGHT,
    GRAPH_PA_MISALIGNMENT_WEIGHT,
    GRAPH_PENALTY_SCORE_MAX,
    GRAPH_TOO_FAR_START_FRACTION,
    GRAPH_WEIGHT_DEFAULTS,
    NEGATIVE_BOWL_SCORE_SNR_SPAN,
    POSITION_ANGLE_ALIGNMENT_SCALE_DEG,
    POSITION_ANGLE_MODULUS_DEG,
    VALLEY_REFERENCE_SNR,
    VALLEY_SCORE_SNR_SPAN,
)
from .segmentation import connected_at_threshold
from .utils import json_dumps_safe, resolve_pixel_scale_arcsec, safe_float, validate_integer_indices


def _graph_settings(config: dict[str, Any]) -> dict[str, float]:
    """Return graph-merge settings from their dedicated namespace."""

    settings = dict(GRAPH_DEFAULTS)
    section = config.get("graph_merge", {}) or {}
    if isinstance(section, dict):
        settings.update(section)
    return settings


@dataclass
class GraphMergeResult:
    graph: nx.Graph
    edges: pd.DataFrame
    components: pd.DataFrame
    clusters: list[list[int]]


def _bridge_features(snr_map: np.ndarray, row_i: pd.Series, row_j: pd.Series) -> dict[str, float]:
    xs, ys = _line_samples(row_i["x"], row_i["y"], row_j["x"], row_j["y"])
    values = _sample_image_nearest(snr_map, xs, ys)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "bridge_snr_mean": 0.0,
            "bridge_snr_min": 0.0,
            "bridge_snr_max": 0.0,
            "bridge_snr_score": 0.0,
            "valley_penalty": 1.0,
            "negative_bowl_penalty": 0.0,
            "pixel_support_score": 0.0,
        }
    mean = float(np.nanmean(finite))
    min_value = float(np.nanmin(finite))
    max_value = float(np.nanmax(finite))
    bridge_score = float(np.clip((mean - BRIDGE_BASELINE_SNR) / BRIDGE_SCORE_SNR_SPAN, 0.0, 1.5))
    support_score = float(np.mean(finite > FORMAL_SNR_LEVELS["weak"]))
    valley_penalty = float(np.clip((VALLEY_REFERENCE_SNR - min_value) / VALLEY_SCORE_SNR_SPAN, 0.0, 1.5))
    negative_bowl_penalty = float(np.clip(-min_value / NEGATIVE_BOWL_SCORE_SNR_SPAN, 0.0, 2.0))
    return {
        "bridge_snr_mean": mean,
        "bridge_snr_min": min_value,
        "bridge_snr_max": max_value,
        "bridge_snr_score": bridge_score,
        "valley_penalty": valley_penalty,
        "negative_bowl_penalty": negative_bowl_penalty,
        "pixel_support_score": support_score,
    }


def _pa_alignment(row_i: pd.Series, row_j: pd.Series) -> tuple[float, float]:
    pa_i = safe_float(row_i.get("_pa"))
    pa_j = safe_float(row_j.get("_pa"))
    if not np.isfinite(pa_i) or not np.isfinite(pa_j):
        return 0.0, 0.0
    delta = abs((pa_i - pa_j + POSITION_ANGLE_MODULUS_DEG / 2.0) % POSITION_ANGLE_MODULUS_DEG - POSITION_ANGLE_MODULUS_DEG / 2.0)
    score = float(np.clip(1.0 - delta / POSITION_ANGLE_ALIGNMENT_SCALE_DEG, 0.0, 1.0))
    penalty = float(np.clip((delta - POSITION_ANGLE_ALIGNMENT_SCALE_DEG) / POSITION_ANGLE_ALIGNMENT_SCALE_DEG, 0.0, 1.0))
    return score, penalty


def _flux_ratio_score(row_i: pd.Series, row_j: pd.Series) -> float:
    for flux_field in ("_total_flux", "_peak_flux"):
        f1 = safe_float(row_i.get(flux_field))
        f2 = safe_float(row_j.get(flux_field))
        if np.isfinite(f1) and np.isfinite(f2) and f1 > 0 and f2 > 0:
            ratio = min(f1, f2) / max(f1, f2)
            return float(np.clip(ratio, 0.0, 1.0))
    return float("nan")


def _ellipse_overlap_approx(
    row_i: pd.Series,
    row_j: pd.Series,
    distance_pix: float,
    config: dict[str, Any],
) -> float:
    maj_i = safe_float(row_i.get("_dc_maj"), safe_float(row_i.get("_maj")))
    maj_j = safe_float(row_j.get("_dc_maj"), safe_float(row_j.get("_maj")))
    scale = resolve_pixel_scale_arcsec(row_i.get("pixel_scale_arcsec"), config, context="graph-merge pixel scale")
    if not np.isfinite(maj_i) or not np.isfinite(maj_j) or scale <= 0:
        return 0.0
    radius_pix = 0.5 * (maj_i + maj_j) / scale
    if radius_pix <= 0:
        return 0.0
    return float(np.clip(1.0 - distance_pix / max(radius_pix, 1e-6), 0.0, 1.0))


def _evidence_strings(features: dict[str, Any]) -> tuple[str, str]:
    positive = []
    negative = []
    for key in [
        "same_pybdsf_island",
        "connected_at_3sigma",
        "connected_at_2p5sigma",
        "connected_at_2sigma",
    ]:
        if features.get(key):
            positive.append(key)
    for key in ["bridge_snr_score", "gaussian_ellipse_overlap_approx", "PA_alignment_score"]:
        if features.get(key, 0) > GRAPH_EVIDENCE_DISPLAY_MIN:
            positive.append(key)
    for key in ["valley_penalty", "negative_bowl_penalty", "too_far_penalty", "pa_misalignment_penalty"]:
        if features.get(key, 0) > GRAPH_EVIDENCE_DISPLAY_MIN:
            negative.append(key)
    return ",".join(positive), ",".join(negative)


def compute_pair_features(
    row_i: pd.Series,
    row_j: pd.Series,
    segmentation: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute graph edge features for one Gaussian pair."""

    dx = safe_float(row_i["x"]) - safe_float(row_j["x"])
    dy = safe_float(row_i["y"]) - safe_float(row_j["y"])
    distance_pix = float(np.hypot(dx, dy))
    pixel_scale = resolve_pixel_scale_arcsec(row_i.get("pixel_scale_arcsec"), config, context="graph-merge pixel scale")
    distance_arcsec = distance_pix * pixel_scale
    beam_major, _beam_minor, _beam_pa = beam_axes_from_config(config)
    beam_norm = distance_arcsec / max(beam_major, 1e-6)

    same_island = (
        str(row_i.get("_island_id")) == str(row_j.get("_island_id"))
        and str(row_i.get("_island_id")) not in {"-1", "nan", "None"}
    )
    conn_3 = connected_at_threshold(row_i, row_j, FORMAL_SNR_LEVELS["strong"])
    conn_25 = connected_at_threshold(row_i, row_j, FORMAL_SNR_LEVELS["intermediate"])
    conn_2 = connected_at_threshold(row_i, row_j, FORMAL_SNR_LEVELS["weak"])
    bridge = _bridge_features(segmentation.snr_map, row_i, row_j)
    pa_score, pa_penalty = _pa_alignment(row_i, row_j)
    flux_ratio = _flux_ratio_score(row_i, row_j)
    overlap = _ellipse_overlap_approx(row_i, row_j, distance_pix, config)

    graph_settings = _graph_settings(config)
    max_pair_distance_arcsec = float(graph_settings["max_pair_distance_arcsec"])
    max_pair_distance_beam = float(graph_settings["max_pair_distance_beam"])
    too_far_penalty = float(
        np.clip(
            max(
                distance_arcsec / max(max_pair_distance_arcsec, 1e-6),
                beam_norm / max(max_pair_distance_beam, 1e-6),
            )
            - GRAPH_TOO_FAR_START_FRACTION,
            0.0,
            GRAPH_PENALTY_SCORE_MAX,
        )
    )
    closeness_score = float(np.clip(1.0 - beam_norm / max(max_pair_distance_beam, 1e-6), 0.0, 1.0))
    compact_pair_penalty = 0.0
    if beam_norm < GRAPH_COMPACT_PAIR_MAX_DISTANCE_BEAM and flux_ratio < GRAPH_COMPACT_PAIR_MAX_FLUX_RATIO:
        compact_pair_penalty = GRAPH_COMPACT_PAIR_PENALTY

    features: dict[str, Any] = {
        "cutout_id": row_i.get("cutout_id"),
        "gaussian_id_1": row_i.get("_gaussian_id"),
        "gaussian_id_2": row_j.get("_gaussian_id"),
        "component_index_1": int(row_i.get("component_index")),
        "component_index_2": int(row_j.get("component_index")),
        "distance_pix": distance_pix,
        "distance_arcsec": distance_arcsec,
        "beam_normalized_distance": beam_norm,
        "same_pybdsf_island": bool(same_island),
        "connected_at_3sigma": bool(conn_3),
        "connected_at_2p5sigma": bool(conn_25),
        "connected_at_2sigma": bool(conn_2),
        "gaussian_ellipse_overlap_approx": overlap,
        "PA_alignment_score": pa_score,
        "flux_ratio_score": flux_ratio,
        "closeness_score": closeness_score,
        "compact_pair_penalty": compact_pair_penalty,
        "too_far_penalty": too_far_penalty,
        "pa_misalignment_penalty": pa_penalty,
    }
    features.update(bridge)
    return features


def score_pair(features: dict[str, Any], config: dict[str, Any]) -> float:
    """Compute the rule-based merge score."""

    # Keep graph-merge weights separate from Stage-1 association weights.
    weights = dict(GRAPH_WEIGHT_DEFAULTS)
    weights.update(config.get("weights_graph", {}) or {})
    score = 0.0
    score += float(weights["same_island"]) * float(features["same_pybdsf_island"])
    score += float(weights["conn_3sigma"]) * float(features["connected_at_3sigma"])
    score += float(weights["conn_2p5sigma"]) * float(features["connected_at_2p5sigma"])
    score += float(weights["conn_2sigma"]) * float(features["connected_at_2sigma"])
    score += float(weights["bridge"]) * float(features["bridge_snr_score"])
    score += float(weights["overlap"]) * float(features["gaussian_ellipse_overlap_approx"])
    score += float(weights["closeness"]) * float(features["closeness_score"])
    score += float(weights["pa_alignment"]) * float(features["PA_alignment_score"])
    score += float(weights["pixel_support"]) * float(features["pixel_support_score"])
    score -= float(weights["valley"]) * float(features["valley_penalty"])
    score -= float(weights["compact_pair"]) * float(features["compact_pair_penalty"])
    score -= float(weights["too_far"]) * float(features["too_far_penalty"])
    score -= GRAPH_NEGATIVE_BOWL_WEIGHT * float(features.get("negative_bowl_penalty", 0.0))
    score -= GRAPH_PA_MISALIGNMENT_WEIGHT * float(features.get("pa_misalignment_penalty", 0.0))
    return float(score)


def candidate_pairs(components: pd.DataFrame, config: dict[str, Any]) -> list[tuple[int, int]]:
    """Generate nearby pairs as zero-based DataFrame row positions."""

    if "component_index" not in components:
        raise ValueError("graph components must provide component_index")
    validate_integer_indices(components["component_index"], context="component_index")
    if len(components) < 2:
        return []
    coords = components[["x", "y"]].to_numpy(float)
    finite = np.isfinite(coords).all(axis=1)
    if finite.sum() < 2:
        return []
    valid_positions = np.where(finite)[0]
    valid_coords = coords[finite]
    scale_value = components["pixel_scale_arcsec"].iloc[0] if "pixel_scale_arcsec" in components else None
    pixel_scale = resolve_pixel_scale_arcsec(scale_value, config, context="graph-merge pixel scale")
    beam_major, _beam_minor, _beam_pa = beam_axes_from_config(config)
    graph_settings = _graph_settings(config)
    max_arcsec = float(graph_settings["max_pair_distance_arcsec"])
    max_beam = float(graph_settings["max_pair_distance_beam"]) * beam_major
    max_distance_arcsec = min(max_arcsec, max_beam)
    radius_pix = max_distance_arcsec / max(pixel_scale, 1e-6)
    tree = cKDTree(valid_coords)
    pairs_local = tree.query_pairs(radius_pix)
    pairs = [(int(valid_positions[i]), int(valid_positions[j])) for i, j in pairs_local]
    pairs.sort()
    return pairs


def build_component_graph(
    components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
) -> GraphMergeResult:
    """Build a graph and merge components by connected components."""

    if "component_index" not in components:
        raise ValueError("graph components must provide component_index")
    components = components.copy()
    components["component_index"] = validate_integer_indices(components["component_index"], context="component_index")
    if components["component_index"].duplicated(keep=False).any():
        raise ValueError("graph components contain duplicate component_index values")
    graph = nx.Graph()
    for _, row in components.iterrows():
        node_id = int(row["component_index"])
        graph.add_node(node_id, **row.to_dict())

    edge_records = []
    threshold = float(_graph_settings(config)["merge_threshold"])
    for idx_i, idx_j in candidate_pairs(components, config):
        row_i = components.iloc[idx_i]
        row_j = components.iloc[idx_j]
        features = compute_pair_features(row_i, row_j, segmentation, config)
        score = score_pair(features, config)
        decision = score > threshold
        positive, negative = _evidence_strings(features)
        features["merge_score"] = score
        features["merge_decision"] = bool(decision)
        features["positive_evidence"] = positive
        features["negative_evidence"] = negative
        edge_records.append(features)
        if decision:
            graph.add_edge(
                int(row_i["component_index"]),
                int(row_j["component_index"]),
                merge_score=score,
                features=features,
            )

    edges = pd.DataFrame(edge_records)
    clusters = [sorted(list(cluster)) for cluster in nx.connected_components(graph)]
    clusters.sort(key=lambda values: (len(values), values[0] if values else -1), reverse=True)
    components = components.copy()
    cluster_id_by_node = {}
    for cluster_idx, cluster in enumerate(clusters):
        for node in cluster:
            cluster_id_by_node[node] = cluster_idx
    components["merged_component_group"] = components["component_index"].map(cluster_id_by_node)
    components["debug_graph_degree"] = components["component_index"].map(dict(graph.degree())).fillna(0).astype(int)
    if not edges.empty:
        edges["debug_info"] = edges.apply(lambda row: json_dumps_safe(row.to_dict()), axis=1)
    return GraphMergeResult(graph=graph, edges=edges, components=components, clusters=clusters)


def strongest_evidence_for_cluster(edges: pd.DataFrame, nodes: list[int]) -> tuple[float, float, str]:
    """Return mean score, max score, and evidence string for edges inside a cluster."""

    if edges.empty or len(nodes) < 2:
        return 0.0, 0.0, ""
    node_set = set(nodes)
    decision_column = "merge_decision" if "merge_decision" in edges else "association_decision"
    score_column = "merge_score" if "merge_score" in edges else "association_score"
    mask = edges["component_index_1"].isin(node_set) & edges["component_index_2"].isin(node_set) & edges[
        decision_column
    ].astype(bool)
    subset = edges.loc[mask]
    if subset.empty:
        return 0.0, 0.0, ""
    mean_score = float(subset[score_column].mean())
    max_idx = subset[score_column].idxmax()
    max_score = float(subset.loc[max_idx, score_column])
    if "positive_evidence" in subset:
        evidence = str(subset.loc[max_idx, "positive_evidence"])
    else:
        # Stage-1 association edges use the richer association schema rather
        # than the legacy graph-merge evidence columns.  Preserve a concise
        # human-readable explanation for merged-source measurements without
        # requiring callers to convert between the two edge representations.
        strongest = subset.loc[max_idx]
        evidence_keys = (
            "connected_at_3sigma",
            "connected_at_2p5sigma",
            "connected_at_2sigma",
            "ellipse_overlap_score",
            "bridge_score",
            "ridge_continuity_score",
            "effective_pa_alignment_score",
        )
        def is_displayable(value: Any) -> bool:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            try:
                number = float(value)
            except (TypeError, ValueError):
                return bool(value)
            return bool(np.isfinite(number) and number > GRAPH_EVIDENCE_DISPLAY_MIN)

        evidence = ",".join(key for key in evidence_keys if is_displayable(strongest.get(key, False)))
    return mean_score, max_score, evidence


def complete_graph_edges_for_singletons(components: pd.DataFrame) -> pd.DataFrame:
    """Build an empty-decision edge table for diagnostics when needed."""

    records = []
    for left, right in combinations(range(len(components)), 2):
        row_i = components.iloc[left]
        row_j = components.iloc[right]
        records.append(
            {
                "cutout_id": row_i.get("cutout_id"),
                "gaussian_id_1": row_i.get("_gaussian_id"),
                "gaussian_id_2": row_j.get("_gaussian_id"),
                "merge_score": np.nan,
                "merge_decision": False,
            }
        )
    return pd.DataFrame(records)
