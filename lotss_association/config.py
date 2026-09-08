"""Validated configuration schema and scientific defaults for the package.

Scientific defaults live here once; production entry points pass through the
schema in this module so implementation fallbacks cannot drift from the YAML
configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

SCIENTIFIC_DEFAULTS: dict[str, Any] = {
    "snr_thresholds": (5.0, 4.0, 3.0, 2.5, 2.0),
    "min_mask_area_pix": 30,
    "connectivity": 2,
    "mean_mode": "median",
    "rms_mode": "mad",
    "smooth_before_segmentation": True,
    "gaussian_smooth_sigma_pix": 1.0,
    "binary_opening": False,
    "binary_closing": True,
}

# Formal contour levels used by the published association method.  The
# validator requires these values in ``snr_thresholds``; production code uses
# the names below instead of scattering equivalent numeric literals.
FORMAL_SNR_LEVELS: dict[str, float] = {
    "strong": 3.0,
    "intermediate": 2.5,
    "weak": 2.0,
}

# Fixed mathematical scales used by the rule-based scores. These formula
# definitions are not user-tunable configuration keys.
POSITION_ANGLE_ALIGNMENT_SCALE_DEG = 45.0
POSITION_ANGLE_MODULUS_DEG = 180.0
BRIDGE_BASELINE_SNR = 1.0
BRIDGE_SCORE_SNR_SPAN = 2.0
VALLEY_REFERENCE_SNR = 2.0
VALLEY_SCORE_SNR_SPAN = 4.0
NEGATIVE_BOWL_SCORE_SNR_SPAN = 3.0

# Fixed score-formula coefficients for the published rule-based method; these
# are not user-tunable configuration keys.
BRIDGE_SCORE_WEIGHTS = (0.35, 0.25, 0.25, 0.15)
RESIDUAL_BRIDGE_SCORE_WEIGHTS = (0.30, 0.20, 0.20, 0.20, 0.10)
RIDGE_SCORE_WEIGHTS = (0.55, 0.30, 0.15)
BRIDGE_LENGTH_SHORT_SCORE = 0.7
RESIDUAL_BRIDGE_MIN_WIDTH_BEAM = 0.6
RESIDUAL_BRIDGE_PEAK_SCORE_SPAN = 2.0
RESIDUAL_BRIDGE_NEGATIVE_MEAN_FRACTION = 0.5
# Frozen method decision gates (not runtime/environment defaults); changing
# them is a method revision.
ASSOCIATION_SUPPORT_SCORE_MIN = 0.45
ASSOCIATION_STRONG_SUPPORT_SCORE_MIN = 0.55
ASSOCIATION_PA_SUPPORT_SCORE_MIN = 0.55
ASSOCIATION_TOO_FAR_DISTANCE_FRACTION = 0.55
ASSOCIATION_TOO_FAR_SUPPORT_FRACTION = 0.7
ASSOCIATION_TOO_FAR_PENALTY_SPAN = 0.3
ASSOCIATION_INDEPENDENT_SUPPORT_WIDTH_FACTOR = 0.7
ASSOCIATION_INDEPENDENT_SUPPORT_LENGTH_FACTOR = 0.7
ASSOCIATION_SEVERE_SUPPORT_FRACTION = 0.4
ASSOCIATION_SEVERE_SCORE_SPREAD = 2.5
ASSOCIATION_HIGH_QUALITY_STRONG_EDGE_FRACTION = 0.6
ASSOCIATION_RESIDUAL_BRIDGE_SCORE_FACTOR = 0.8
ASSOCIATION_GROUP_SUPPORT_PADDING_BEAM = 2.5
ASSOCIATION_GROUP_SUPPORT_MIN_PADDING_PIX = 5
ASSOCIATION_MULTI_GAUSSIAN_EXTENDED_MIN_LAS_BEAM = 3.0
RESOLVED_SIGNIFICANCE_NORMALIZATION = 3.0
RESOLVED_PROBABILITY_BOUNDARY = 0.75
MARGINAL_PROBABILITY_BOUNDARY = 0.25
MORPHOLOGY_SCORE_WEIGHTS = (0.40, 0.40, 0.20)
GRAPH_EVIDENCE_DISPLAY_MIN = 0.5
GRAPH_NEGATIVE_BOWL_WEIGHT = 0.5
GRAPH_PA_MISALIGNMENT_WEIGHT = 0.5
GRAPH_TOO_FAR_START_FRACTION = 0.75
GRAPH_PENALTY_SCORE_MAX = 1.5
GRAPH_COMPACT_PAIR_MAX_DISTANCE_BEAM = 1.5
GRAPH_COMPACT_PAIR_MAX_FLUX_RATIO = 0.2
GRAPH_COMPACT_PAIR_PENALTY = 0.5
ARTIFACT_NEGATIVE_SNR_FLOOR = -1.0
ARTIFACT_DEEP_VALLEY_OFFSET = 0.3
ARTIFACT_DEEP_VALLEY_SPAN = 2.2
# Frozen artifact-rejection gates from the rule-based method (see the
# threshold-governance note in docs/software_package.md).
ARTIFACT_LARGE_LABEL_FRACTION = 0.05
ARTIFACT_LARGE_LABEL_SPAN = 0.10
ARTIFACT_LARGE_LABEL_AREA_BEAMS = 30.0
ARTIFACT_LARGE_LABEL_DISTANCE_SCALE = 4.0
ARTIFACT_SIDEBAND_FLUX_RATIO = 0.08
ARTIFACT_SIDEBAND_DISTANCE_BEAM = 6.0

# Merged-source measurement constants for the legacy
# `lotss_association_merged_sources` catalogue.
MEASUREMENT_PADDING_ARCSEC = 10.0
MEASUREMENT_MIN_FLUX_RATIO_DOUBLE_LOBE = 0.1
MEASUREMENT_PA_ALIGNMENT_TOLERANCE_DEG = 35.0
MEASUREMENT_DOUBLE_LOBE_MIN_COMPONENTS = 2
MEASUREMENT_DOUBLE_LOBE_MIN_LAS_PIX = 5.0
MEASUREMENT_CONFIDENCE_BASE = 0.2
MEASUREMENT_CONFIDENCE_PER_COMPONENT = 0.15
MEASUREMENT_CONFIDENCE_PER_MEAN_SCORE = 0.15
MEASUREMENT_CONFIDENCE_DOUBLE_LOBE_BONUS = 0.1

# Artifact-rejection score gates used by the pair penalty layer.
ARTIFACT_SIDEBAND_NEGATIVE_BOWL_MIN = 0.2
ARTIFACT_SIDEBAND_DEEP_VALLEY_MIN = 0.8
ARTIFACT_SIDEBAND_SCORE_NEAR = 0.7
ARTIFACT_SIDEBAND_SCORE_CLOSE = 0.4
ARTIFACT_SIDEBAND_BRIDGE_SCORE_MAX = 0.2
ARTIFACT_SIDEBAND_RIDGE_SCORE_MAX = 0.2
ARTIFACT_SIDEBAND_PA_SCORE_MAX = 0.25
ARTIFACT_FLAG_SCORE_MIN = 0.5
ARTIFACT_TOO_FAR_FLAG_MIN = 1.0
ASSOCIATION_ONLY_2SIGMA_FLAG_FRACTION = 0.5

MATCHING_DEFAULTS: dict[str, float] = {"preselect_margin_arcsec": 120.0}

ASSOCIATION_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_pair_distance_beam": 15.0,
    "max_pair_distance_arcsec": 120.0,
    "threshold_strong": 3.0,
    "threshold_weak": 2.0,
    "min_bridge_width_beam": 0.8,
    "min_bridge_length_beam": 1.0,
    "max_only_2sigma_score": 0.5,
    "enable_beam_aware_morphology": True,
    "enable_unresolved_pair_veto": True,
    "enable_residual_bridge": True,
    "residual_bridge_threshold_snr": 2.5,
    "residual_bridge_min_length_fraction": 0.45,
    "residual_bridge_min_area_beams": 0.25,
    "residual_bridge_endpoint_exclusion_beam": 0.6,
    "unresolved_veto_beam_like_score": 0.70,
    "unresolved_veto_min_bridge_score": 0.70,
    "unresolved_veto_min_residual_bridge_score": 0.55,
    "unresolved_veto_min_common_envelope_area_beam": 1.0,
    "veto_score_cap": -1.0,
    "independent_support_bridge_min_score": 0.55,
    "independent_support_ridge_min_score": 0.55,
    "independent_support_overlap_min_score": 0.75,
    "quality_thresholds": {"high": 3.5, "medium": 2.5, "low": 1.5},
}

ASSOCIATION_WEIGHT_DEFAULTS: dict[str, float] = {
    "closeness": 1.0,
    "overlap": 1.0,
    "pa_alignment": 0.8,
    "conn_3sigma": 1.5,
    "conn_2p5sigma": 0.6,
    "conn_2sigma": 0.0,
    "bridge": 1.2,
    "ridge": 1.2,
    "flux_continuity": 0.5,
    "flow_alignment": 0.6,
    "valley": 1.5,
    "only_2sigma": 1.0,
    "negative_bowl": 1.2,
    "sidelobe": 1.5,
    "too_far": 2.0,
    "large_mask_swallow": 1.5,
}

# Association-type thresholds for callers constructing a config
# programmatically; the YAML materializes the same values.
ASSOCIATION_TYPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "compact_multi_gaussian": {"max_las_beam": 4.0},
    "continuous_extended": {"min_las_beam": 4.0},
    "diffuse_extended": {"min_las_beam": 6.0},
    "linear_or_tail_like": {"min_axis_ratio": 2.0},
    "complex_association": {"min_components": 5},
    "weak_association": {"max_quality": "low"},
    "artifact_risk": {"enabled": True},
}

CLASSIFICATION_DEFAULTS: dict[str, float] = {
    "beam_like_axis_ratio_tolerance": 0.35,
    "beam_like_size_tolerance": 0.35,
    "beam_like_pa_tolerance_deg": 20.0,
    "beam_like_score_unresolved": 0.70,
    "resolved_deconv_major_fraction": 0.45,
    "resolved_deconv_minor_fraction": 0.35,
    "marginal_deconv_major_fraction": 0.20,
    "intrinsic_resolved_fraction": 0.35,
    "intrinsic_marginal_fraction": 0.15,
    "low_snr_threshold": 5.0,
    "pa_weight_unresolved": 0.0,
    "pa_weight_beam_like": 0.0,
    "pa_weight_marginal": 0.35,
    "pa_weight_resolved": 1.0,
    "pa_weight_unknown": 0.15,
}

GRAPH_DEFAULTS: dict[str, float] = {
    "max_pair_distance_arcsec": 120.0,
    "max_pair_distance_beam": 15.0,
    "merge_threshold": 3.0,
}

GRAPH_WEIGHT_DEFAULTS: dict[str, float] = {
    "same_island": 0.0,
    "conn_3sigma": 2.0,
    "conn_2p5sigma": 1.0,
    "conn_2sigma": 0.3,
    "bridge": 1.0,
    "overlap": 0.8,
    "closeness": 0.8,
    "pa_alignment": 0.7,
    "pixel_support": 0.8,
    "valley": 1.5,
    "compact_pair": 1.0,
    "too_far": 2.5,
}

BEAM_DEFAULTS: dict[str, Any] = {
    "major_arcsec": None,
    "minor_arcsec": None,
    "pa_deg": 0.0,
}

# Fixed Stage 1.5 method constants used by local risk, split, and quality
# decisions. Runtime-configurable local-sanity values remain in the mapping
# below.
LOCAL_MULTI_PEAK_SNR_FLOOR = 5.0
LOCAL_MULTI_PEAK_FRACTION = 0.35
LOCAL_BEAM_AREA_GAUSSIAN_FACTOR = 4.0
LOCAL_EDGE_BRIDGE_SCORE_MIN = 0.25
LOCAL_EDGE_RIDGE_SCORE_MIN = 0.35
LOCAL_SCORE_NORMALIZATION_SPAN = 3.0
LOCAL_MIN_PAD_PIX = 5
LOCAL_PAD_BEAM_FACTOR = 2.5
LOCAL_LARGE_MASK_MIN_AREA_BEAMS = 30.0
LOCAL_LARGE_MASK_LAS_AREA_FACTOR = 1.8
LOCAL_LARGE_MASK_WEAK_EDGE_FRACTION = 0.25
LOCAL_LARGE_MASK_ONLY_2SIGMA_FRACTION = 0.10
LOCAL_LARGE_MASK_COMPONENT_COUNT = 8
LOCAL_CHAIN_STRONG_EDGE_FRACTION = 0.30
LOCAL_CHAIN_MIN_COMPONENTS = 4
LOCAL_NEEDS_CHECK_RISK = 0.35
LOCAL_RIDGE_RISK_WEIGHT = 1.0
LOCAL_RIDGE_RISK_SPAN = 0.65
LOCAL_SADDLE_RISK_WEIGHT = 1.1
LOCAL_WEAK_EDGE_RISK_WEIGHT = 0.8
LOCAL_WEAK_EDGE_RISK_SPAN = 0.50
LOCAL_ONLY_2SIGMA_RISK_WEIGHT = 0.7
LOCAL_ONLY_2SIGMA_RISK_SPAN = 0.75
LOCAL_CHAIN_RISK = 0.55
LOCAL_LARGE_MASK_RISK = 0.60
LOCAL_MULTI_PEAK_MIN_COUNT = 3
LOCAL_MULTI_PEAK_MIN_SEPARATION_BEAM = 4.0
LOCAL_MULTI_PEAK_RISK = 0.55
LOCAL_RISK_SCORE_MAX = 5.0
LOCAL_EDGE_SCORE_REFERENCE = 2.2
LOCAL_EDGE_WEAK_RISK = 1.0
LOCAL_EDGE_ONLY_2SIGMA_RISK = 0.9
LOCAL_EDGE_RIDGE_GAP_RISK = 0.7
LOCAL_EDGE_DEEP_VALLEY_MIN = 0.8
LOCAL_EDGE_DEEP_VALLEY_RISK = 0.8
LOCAL_EDGE_LOW_SUPPORT_RISK = 0.6
LOCAL_EDGE_SCORE_RISK_WEIGHT = 0.5
LOCAL_SPLIT_REQUIRED_RISK = 1.0
LOCAL_EDGE_CUT_RISK_MIN = 1.15
LOCAL_POST_SPLIT_RISK_FACTOR = 0.75
LOCAL_HIGH_RISK_MIN = 1.5
LOCAL_MEDIUM_RISK_MIN = 0.9

# Stage-2 and local-sanity defaults; release YAML materializes all values,
# and the dictionaries support programmatic configs before validation.
LOCAL_SANITY_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_group_las_beam_before_check": 8.0,
    "max_group_n_gaussians_before_check": 6,
    "min_saddle_to_peak_ratio": 0.35,
    "max_ridge_gap_fraction": 0.35,
    "max_weak_edge_chain_fraction": 0.50,
    "max_only_2sigma_edge_fraction": 0.25,
    "split_overmerged_groups": True,
    "mark_suspicious_if_unsplittable": True,
    "min_subgroup_size": 1,
    "min_split_score_gain": 0.5,
    "min_path_snr": 2.5,
}

PARENT_SEED_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_box_gap_beam": 10.0,
    "strong_box_gap_beam": 5.0,
    "min_endpoint_las_beam": 3.0,
    "min_endpoint_mask_area_beam": 3.0,
    "min_endpoint_n_gaussians": 2,
    "min_parent_seed_peak_snr": 8.0,
    "min_parent_seed_area_3sigma_beam": 2.0,
    "max_parent_candidates_per_group": 1,
    "max_parent_candidates_per_cutout": 10,
    "candidate_search_padding_factor": 1.5,
    "compact_singleton": {"max_las_beam": 3.0, "max_area_3sigma_beam": 2.0, "max_mask_area_beam": 2.0},
    "mask_area_fallback": {"las_weight": 0.45, "gaussian_count_weight": 0.35},
    "midpoint_core": {"search_radius_beam": 3.0, "axial_fraction_min": 0.15, "axial_fraction_max": 0.85},
    "score_weights": {"gap": 1.5, "axis_alignment": 1.4, "facing": 1.0, "flux_balance": 0.7, "size_balance": 0.6, "midpoint_core": 0.8},
    "thresholds": {"high_score": 4.0, "medium_score": 3.2, "low_score": 2.5},
    "evidence": {"axis_alignment_min": 0.7, "facing_min": 0.6, "max_flux_ratio": 10.0, "max_size_ratio": 5.0, "missing_ratio_score": 0.35, "strong_gap_min_count": 2, "wide_gap_min_count": 3},
}

HOST_SUPPORT_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "require_host_for_parent_link": True,
    "catalog_priority": ["catwise2020", "allwise"],
    "min_search_radius_arcsec": 10.0,
    "max_search_radius_arcsec": 30.0,
    "search_radius_fraction_of_sep": 0.12,
    "max_host_results_per_query": 20,
    "host_score_weight": 1.0,
    "min_host_quality_for_default_candidate": "medium",
    "host_quality_thresholds": {"high": 3.0, "medium": 2.0},
    "geometry": {"high_max_perp_offset_beam": 1.5, "medium_max_perp_offset_beam": 2.0, "high_fractional_position_min": 0.35, "high_fractional_position_max": 0.65, "medium_fractional_position_min": 0.25, "medium_fractional_position_max": 0.75},
    "wise_color": {"agn_bonus_w1_w2_min": 0.8, "agn_bonus": 0.5, "require_agn_color": False},
    "score_weights": {"midpoint_closeness": 2.0, "axis_consistency": 1.5, "wise_detection": 0.8, "artifact_penalty": 1.0},
    "detection_scoring": {"snr_scale": 10.0, "catalogued_w1_fallback": 0.5},
    "parent_quality": {"high_final_score_min": 6.0},
}

PARENT_LINK_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_box_gap_beam": 12.0,
    "max_center_distance_beam": 40.0,
    "min_axis_alignment": 0.7,
    "min_facing_score": 0.6,
    "max_flux_ratio": 20.0,
    "max_size_ratio": 8.0,
    "min_symmetry_score": 0.6,
    "artifact_veto_score": 1.2,
    "artifact_suspicious_score": 0.8,
    "min_parent_ridge_support": 0.65,
    "bright_source_near_beam": 8.0,
    "bright_source_radial_min": 0.72,
    "bright_source_flux_ratio_min": 5.0,
    "fragment_density_radius_beam": 8.0,
    "fragment_count_artifact": 12,
    "lobe_peak_host_radius_arcsec_min": 5.0,
    "lobe_peak_host_radius_arcsec_max": 10.0,
    "max_parent_candidates_per_group": 2,
    "max_parent_candidates_per_cutout": 20,
    "candidate_search_padding_factor": 1.5,
    "endpoint_thresholds": {
        "point_las_max_beam": 3.0, "point_major_max_beam": 1.8, "point_area_max_beam": 3.0, "point_axis_ratio_max": 1.7,
        "compact_las_max_beam": 3.0, "compact_area_max_beam": 2.0, "compact_major_max_beam": 1.5, "compact_axis_ratio_max": 1.5,
        "singleton_las_max_beam": 3.5, "singleton_area_max_beam": 2.5, "singleton_axis_ratio_max": 1.6,
        "observed_major_max_beam": 1.4, "observed_minor_max_beam": 1.4, "noise_peak_snr_min": 6.0, "noise_area_min_beam": 1.5,
        "extended_las_min_beam": 3.0, "extended_area_min_beam": 2.5, "extended_major_min_beam": 2.0, "extended_axis_ratio_min": 1.5,
        "extended_peak_snr_min": 6.0, "extended_n_gaussians_min": 2, "strong_multi_min_gaussians": 3, "strong_multi_min_area_beam": 2.0,
        "strong_multi_las_margin_beam": 0.5, "self_extended_area_margin_beam": 0.5, "self_extended_major_min_beam": 2.2,
        "self_extended_axis_ratio_margin": 0.1,
    },
    "rescue_thresholds": {
        "near_gap_beam": 6.0, "gap_to_mean_box_max": 1.0, "gap_to_min_box_max": 1.5, "support_ridge_min": 0.45,
        "axis_alignment_min": 0.50, "facing_min": 0.40, "max_flux_ratio": 30.0, "max_size_ratio": 10.0,
        "min_gaussians_each": 2, "extended_gaussians_min": 3, "min_area_each_beam": 2.5,
    },
    "artifact_environment_thresholds": {
        "very_close_distance_beam": 4.0, "very_bright_flux_ratio": 10.0, "very_close_score": 0.7, "artifact_flag_score": 1.4,
        "radial_base_score": 1.1, "radial_alignment_bonus": 0.5, "crowded_fragment_score": 0.8, "bright_reference_count": 5,
    },
    "support_thresholds": {
        "sigma_3": 3.0, "sigma_2p5": 2.5, "bridge_low_snr": 2.0, "bridge_high_snr": 2.5,
        "bridge_low_fraction": 0.45, "bridge_high_fraction": 0.25, "ridge_low_weight": 0.6, "ridge_high_weight": 0.4,
        "neighbor_search_radius_beam": 8.0, "neighbor_ridge_min": 0.45, "threshold_match_tolerance": 0.26,
    },
    "symmetry_weights": {"axis_alignment": 0.30, "facing": 0.25, "midpoint_symmetry": 0.15, "flux_balance": 0.15, "size_balance": 0.15},
    "lobe_peak_host_scoring": {"snr_scale": 10.0, "w1_detection_weight": 1.0, "w2_detection_weight": 0.5, "catalogued_w1_fallback": 0.5, "closeness_weight": 1.8, "detection_weight": 0.8, "high_score_min": 2.0, "medium_score_min": 1.2, "high_radius_fraction": 0.6},
    "candidate_quality_thresholds": {"high_symmetry_min": 0.70, "needs_host_check_symmetry_min": 0.68},
    "score_weights": {"symmetry": 1.4, "gap": 0.7, "midpoint_core": 0.3},
}

_ROOT_KEYS = {
    "h5",
    "matching",
    "runtime",
    "features",
    "snr_thresholds",
    "mean_mode",
    "rms_mode",
    "smooth_before_segmentation",
    "gaussian_smooth_sigma_pix",
    "min_mask_area_pix",
    "connectivity",
    "binary_opening",
    "binary_closing",
    "pixel_scale_arcsec",
    "beam",
    "association",
    "weights_association",
    "association_types",
    "beam_aware_classification",
    "weights_graph",
    "graph_merge",
    "visualization",
    "local_association",
    "parent_seed_selection",
    "host_support",
    "parent_linking",
}

_SECTION_KEYS = {
    "h5": {"image_key", "rms_key", "mean_key", "id_key", "ra_key", "dec_key", "wcs_key"},
    "matching": set(MATCHING_DEFAULTS),
    "runtime": {"strict_metadata"},
    "features": {
        "use_multithreshold_contour",
        "use_ridge_continuity",
        "use_ellipse_overlap",
        "use_pa_alignment",
        "use_weak_edge_anti_chaining",
        "use_artifact_penalties_layer1",
        "use_artifact_penalties_layer2",
        "use_midpoint_host_support",
        "use_lobe_peak_host_contradiction",
        "use_stage2_relative_scale_constraints",
        "use_stage2_endpoint_filtering",
    },
    "beam": {
        "major_arcsec",
        "minor_arcsec",
        "pa_deg",
        "pixel_pa_deg",
        "ra_axis_sign",
        "dec_axis_sign",
    },
    "graph_merge": set(GRAPH_DEFAULTS),
    "association": set(ASSOCIATION_DEFAULTS) | {"quality_thresholds"},
    "quality_thresholds": {"high", "medium", "low"},
    "weights_graph": {
        *GRAPH_WEIGHT_DEFAULTS,
    },
    "weights_association": {
        "closeness",
        "overlap",
        "pa_alignment",
        "conn_3sigma",
        "conn_2p5sigma",
        "conn_2sigma",
        "bridge",
        "ridge",
        "flux_continuity",
        "flow_alignment",
        "valley",
        "only_2sigma",
        "negative_bowl",
        "sidelobe",
        "too_far",
        "large_mask_swallow",
    },
    "beam_aware_classification": {
        "beam_like_axis_ratio_tolerance",
        "beam_like_size_tolerance",
        "beam_like_pa_tolerance_deg",
        "beam_like_score_unresolved",
        "resolved_deconv_major_fraction",
        "resolved_deconv_minor_fraction",
        "marginal_deconv_major_fraction",
        "intrinsic_resolved_fraction",
        "intrinsic_marginal_fraction",
        "low_snr_threshold",
        "pa_weight_unresolved",
        "pa_weight_beam_like",
        "pa_weight_marginal",
        "pa_weight_resolved",
        "pa_weight_unknown",
    },
    "visualization": {
        "stretch",
        "percent_clip",
        "draw_gaussian_ellipses",
        "draw_graph_edges",
        "draw_segmentation_contours",
        "overview",
        "zoom",
    },
    "visualization_overview": {
        "draw_all_labels", "draw_singletons", "label_min_components", "max_labels", "label_top_by",
        "draw_gaussian_ids", "draw_component_ids", "max_gaussians_drawn", "gaussian_marker_size", "max_edges_drawn",
        "draw_nonmerged_edges", "edge_min_score", "contour_thresholds",
    },
    "visualization_zoom": {
        "enabled", "max_zoom_per_cutout", "select_min_components", "select_top_las", "select_top_confidence",
        "padding_pix", "min_size_pix", "max_size_pix", "contour_thresholds", "draw_gaussian_ids", "draw_edge_scores",
    },
    "local_association": {"enabled", "local_sanity"},
    "local_sanity": {
        "enabled", "max_group_las_beam_before_check", "max_group_n_gaussians_before_check",
        "min_saddle_to_peak_ratio", "max_ridge_gap_fraction", "max_weak_edge_chain_fraction",
        "max_only_2sigma_edge_fraction", "split_overmerged_groups", "mark_suspicious_if_unsplittable",
        "min_subgroup_size", "min_split_score_gain", "min_path_snr",
    },
    "parent_seed_selection": {
        "enabled", "max_box_gap_beam", "strong_box_gap_beam", "min_endpoint_las_beam",
        "min_endpoint_mask_area_beam", "min_endpoint_n_gaussians", "min_parent_seed_peak_snr",
        "min_parent_seed_area_3sigma_beam", "max_parent_candidates_per_group", "max_parent_candidates_per_cutout",
        "candidate_search_padding_factor", "compact_singleton", "mask_area_fallback", "midpoint_core",
        "score_weights", "thresholds", "evidence",
    },
    "compact_singleton": {"max_las_beam", "max_area_3sigma_beam", "max_mask_area_beam"},
    "mask_area_fallback": {"las_weight", "gaussian_count_weight"},
    "midpoint_core": {"search_radius_beam", "axial_fraction_min", "axial_fraction_max"},
    "parent_seed_score_weights": {"gap", "axis_alignment", "facing", "flux_balance", "size_balance", "midpoint_core"},
    "parent_seed_thresholds": {"high_score", "medium_score", "low_score"},
    "parent_seed_evidence": {
        "axis_alignment_min", "facing_min", "max_flux_ratio", "max_size_ratio", "missing_ratio_score",
        "strong_gap_min_count", "wide_gap_min_count",
    },
    "host_support": {
        "enabled", "require_host_for_parent_link", "catalog_priority", "min_search_radius_arcsec",
        "max_search_radius_arcsec", "search_radius_fraction_of_sep", "max_host_results_per_query", "host_score_weight",
        "min_host_quality_for_default_candidate", "host_quality_thresholds", "geometry", "wise_color", "score_weights",
        "detection_scoring", "parent_quality",
    },
    "host_geometry": {
        "high_max_perp_offset_beam", "medium_max_perp_offset_beam", "high_fractional_position_min",
        "high_fractional_position_max", "medium_fractional_position_min", "medium_fractional_position_max",
    },
    "host_wise_color": {"agn_bonus_w1_w2_min", "agn_bonus", "require_agn_color"},
    "host_score_weights": {"midpoint_closeness", "axis_consistency", "wise_detection", "artifact_penalty"},
    "host_detection_scoring": {"snr_scale", "catalogued_w1_fallback"},
    "host_parent_quality": {"high_final_score_min"},
    "parent_linking": {
        "enabled", "max_box_gap_beam", "max_center_distance_beam", "min_axis_alignment", "min_facing_score",
        "max_flux_ratio", "max_size_ratio", "min_symmetry_score", "artifact_veto_score", "artifact_suspicious_score",
        "min_parent_ridge_support", "bright_source_near_beam", "bright_source_radial_min", "bright_source_flux_ratio_min",
        "fragment_density_radius_beam", "fragment_count_artifact", "lobe_peak_host_radius_arcsec_min",
        "lobe_peak_host_radius_arcsec_max", "max_parent_candidates_per_group", "max_parent_candidates_per_cutout",
        "candidate_search_padding_factor", "endpoint_thresholds", "rescue_thresholds", "artifact_environment_thresholds",
        "support_thresholds", "symmetry_weights", "lobe_peak_host_scoring", "candidate_quality_thresholds", "score_weights",
    },
    "parent_endpoint_thresholds": {
        "point_las_max_beam", "point_major_max_beam", "point_area_max_beam", "point_axis_ratio_max",
        "compact_las_max_beam", "compact_area_max_beam", "compact_major_max_beam", "compact_axis_ratio_max",
        "singleton_las_max_beam", "singleton_area_max_beam", "singleton_axis_ratio_max", "observed_major_max_beam",
        "observed_minor_max_beam", "noise_peak_snr_min", "noise_area_min_beam", "extended_las_min_beam",
        "extended_area_min_beam", "extended_major_min_beam", "extended_axis_ratio_min", "extended_peak_snr_min",
        "extended_n_gaussians_min", "strong_multi_min_gaussians", "strong_multi_min_area_beam", "strong_multi_las_margin_beam",
        "self_extended_area_margin_beam", "self_extended_major_min_beam", "self_extended_axis_ratio_margin",
    },
    "parent_rescue_thresholds": {
        "near_gap_beam", "gap_to_mean_box_max", "gap_to_min_box_max", "support_ridge_min", "axis_alignment_min",
        "facing_min", "max_flux_ratio", "max_size_ratio", "min_gaussians_each", "extended_gaussians_min", "min_area_each_beam",
    },
    "parent_artifact_environment_thresholds": {
        "very_close_distance_beam", "very_bright_flux_ratio", "very_close_score", "artifact_flag_score", "radial_base_score",
        "radial_alignment_bonus", "crowded_fragment_score", "bright_reference_count",
    },
    "parent_support_thresholds": {
        "sigma_3", "sigma_2p5", "bridge_low_snr", "bridge_high_snr", "bridge_low_fraction", "bridge_high_fraction",
        "ridge_low_weight", "ridge_high_weight", "neighbor_search_radius_beam", "neighbor_ridge_min", "threshold_match_tolerance",
    },
    "parent_symmetry_weights": {"axis_alignment", "facing", "midpoint_symmetry", "flux_balance", "size_balance"},
    "parent_lobe_peak_host_scoring": {
        "snr_scale", "w1_detection_weight", "w2_detection_weight", "catalogued_w1_fallback", "closeness_weight",
        "detection_weight", "high_score_min", "medium_score_min", "high_radius_fraction",
    },
    "parent_candidate_quality_thresholds": {"high_symmetry_min", "needs_host_check_symmetry_min"},
    "parent_link_score_weights": {"symmetry", "gap", "midpoint_core"},
    "association_types": {
        "compact_multi_gaussian",
        "continuous_extended",
        "diffuse_extended",
        "linear_or_tail_like",
        "complex_association",
        "weak_association",
        "artifact_risk",
    },
}

_TYPE_SECTION_KEYS = {
    "compact_multi_gaussian": {"max_las_beam"},
    "continuous_extended": {"min_las_beam"},
    "diffuse_extended": {"min_las_beam"},
    "linear_or_tail_like": {"min_axis_ratio"},
    "complex_association": {"min_components"},
    "weak_association": {"max_quality"},
    "artifact_risk": {"enabled"},
}

_ASSOCIATION_BOOL_KEYS = {
    "enabled",
    "enable_beam_aware_morphology",
    "enable_unresolved_pair_veto",
    "enable_residual_bridge",
}

_ROOT_BOOL_KEYS = {"smooth_before_segmentation", "binary_opening", "binary_closing"}

_PARENT_BOOL_KEYS = {
    "enabled",
    "split_overmerged_groups",
    "mark_suspicious_if_unsplittable",
    "require_host_for_parent_link",
    "require_agn_color",
    "draw_all_labels",
    "draw_singletons",
    "draw_gaussian_ids",
    "draw_component_ids",
    "draw_nonmerged_edges",
    "draw_edge_scores",
    "draw_gaussian_ellipses",
    "draw_graph_edges",
    "draw_segmentation_contours",
}
_PARENT_STRING_KEYS = {"min_host_quality_for_default_candidate", "max_quality", "stretch", "label_top_by"}
_PARENT_LIST_KEYS = {"catalog_priority", "percent_clip", "contour_thresholds"}
_PARENT_NESTED_KEYS = {
    "overview": "visualization_overview",
    "zoom": "visualization_zoom",
    "local_sanity": "local_sanity",
    "compact_singleton": "compact_singleton",
    "mask_area_fallback": "mask_area_fallback",
    "midpoint_core": "midpoint_core",
    "score_weights": "parent_score_weights",
    "thresholds": "parent_seed_thresholds",
    "evidence": "parent_seed_evidence",
    "host_quality_thresholds": "quality_thresholds",
    "geometry": "host_geometry",
    "wise_color": "host_wise_color",
    "detection_scoring": "host_detection_scoring",
    "parent_quality": "host_parent_quality",
    "endpoint_thresholds": "parent_endpoint_thresholds",
    "rescue_thresholds": "parent_rescue_thresholds",
    "artifact_environment_thresholds": "parent_artifact_environment_thresholds",
    "support_thresholds": "parent_support_thresholds",
    "symmetry_weights": "parent_symmetry_weights",
    "lobe_peak_host_scoring": "parent_lobe_peak_host_scoring",
    "candidate_quality_thresholds": "parent_candidate_quality_thresholds",
}


def _validate_nested_values(mapping: Mapping[str, Any], path: str, schema_name: str) -> None:
    """Validate known Stage 2/visualization values without silently coercing them."""

    allowed = _SECTION_KEYS[schema_name]
    _check_unknown(mapping, allowed, path)
    for key, value in mapping.items():
        if key in _PARENT_NESTED_KEYS and isinstance(value, Mapping):
            nested_name = _PARENT_NESTED_KEYS[key]
            if key == "score_weights" and schema_name == "host_support":
                nested_name = "host_score_weights"
            elif key == "score_weights" and schema_name == "parent_seed_selection":
                nested_name = "parent_seed_score_weights"
            elif key == "score_weights" and schema_name == "parent_linking":
                nested_name = "parent_link_score_weights"
            _validate_nested_values(_mapping(value, f"{path}.{key}"), f"{path}.{key}", nested_name)
        elif key in _PARENT_BOOL_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"{path}.{key} must be boolean")
        elif key in _PARENT_STRING_KEYS:
            if not isinstance(value, str):
                raise ValueError(f"{path}.{key} must be a string")
        elif key in _PARENT_LIST_KEYS:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{path}.{key} must be a sequence")
            if key == "catalog_priority":
                if not all(isinstance(item, str) and item for item in value):
                    raise ValueError(f"{path}.{key} must contain non-empty strings")
            else:
                for index, item in enumerate(value):
                    _finite_number(item, f"{path}.{key}[{index}]")
                if key == "percent_clip":
                    if len(value) != 2 or not 0 <= float(value[0]) < float(value[1]) <= 100:
                        raise ValueError(f"{path}.{key} must be two ordered percentages in [0, 100]")
                if key == "contour_thresholds" and any(
                    value[index] <= value[index + 1] for index in range(len(value) - 1)
                ):
                    raise ValueError(f"{path}.{key} must be strictly descending")
        elif key == "edge_min_score" and value is None:
            continue
        else:
            _finite_number(value, f"{path}.{key}")


@dataclass(frozen=True)
class AssociationConfig:
    """Canonical pair-association thresholds used by the production path."""

    max_pair_distance_beam: float
    max_pair_distance_arcsec: float | None
    threshold_strong: float
    threshold_weak: float


@dataclass(frozen=True)
class BeamConfig:
    """Validated restoring-beam metadata."""

    major_arcsec: float
    minor_arcsec: float
    pa_deg: float


@dataclass(frozen=True)
class PipelineConfig:
    """Small typed view over the mapping accepted by public entry points."""

    snr_thresholds: tuple[float, ...]
    min_mask_area_pix: int
    connectivity: int
    pixel_scale_arcsec: float | None
    beam: BeamConfig
    association: AssociationConfig


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _check_unknown(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(str(key) for key in mapping if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unknown configuration key(s) at {path}: {', '.join(unknown)}")


def _check_required_tree(mapping: Mapping[str, Any], expected: Mapping[str, Any], path: str) -> None:
    """Require every key in a release section, including nested mappings."""

    missing = sorted(set(expected) - set(mapping))
    if missing:
        raise ValueError(f"strict release configuration missing key(s) at {path}: {', '.join(missing)}")
    for key, value in expected.items():
        if isinstance(value, Mapping):
            actual = mapping.get(key)
            if not isinstance(actual, Mapping):
                raise ValueError(f"{path}.{key} must be a mapping in strict release configuration")
            _check_required_tree(actual, value, f"{path}.{key}")


def _finite_number(value: Any, path: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite numeric value") from exc
    if not isfinite(number) or (positive and number <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{path} must be a finite {qualifier}numeric value")
    return number


def parse_pipeline_config(config: Mapping[str, Any], *, require_core: bool = True) -> PipelineConfig:
    """Validate and return the typed core configuration view."""

    _check_unknown(config, _ROOT_KEYS, "root")
    for section_name in (
        "h5", "runtime", "features", "beam", "association", "weights_association", "weights_graph",
        "beam_aware_classification", "visualization", "association_types", "graph_merge", "local_association",
        "parent_seed_selection", "host_support", "parent_linking",
    ):
        section = config.get(section_name)
        if section is not None:
            _check_unknown(_mapping(section, section_name), _SECTION_KEYS[section_name], section_name)
    association_types = config.get("association_types")
    if association_types is not None:
        for name, values in _mapping(association_types, "association_types").items():
            if name not in _TYPE_SECTION_KEYS:
                raise ValueError(f"Unknown configuration key(s) at association_types: {name}")
            _check_unknown(
                _mapping(values, f"association_types.{name}"),
                _TYPE_SECTION_KEYS[name],
                f"association_types.{name}",
            )
    runtime = _mapping(config.get("runtime", {}), "runtime")
    if "strict_metadata" in runtime and not isinstance(runtime["strict_metadata"], bool):
        raise ValueError("runtime.strict_metadata must be boolean")
    strict_release = bool(runtime.get("strict_metadata", False))
    if require_core and strict_release:
        required_sections = {
            "association", "weights_association", "weights_graph", "beam_aware_classification", "matching",
            "graph_merge", "local_association", "parent_seed_selection", "host_support", "parent_linking",
        }
        missing_sections = sorted(section for section in required_sections if section not in config)
        if missing_sections:
            raise ValueError(
                "strict release configuration requires sections: " + ", ".join(missing_sections)
            )
        required_association = set(ASSOCIATION_DEFAULTS)
        missing_association = sorted(required_association - set(_mapping(config["association"], "association")))
        if missing_association:
            raise ValueError("strict release configuration missing association key(s): " + ", ".join(missing_association))
        for section_name, required_keys in (
            ("matching", set(MATCHING_DEFAULTS)),
            ("weights_association", set(ASSOCIATION_WEIGHT_DEFAULTS)),
            ("weights_graph", set(GRAPH_WEIGHT_DEFAULTS)),
            ("beam_aware_classification", set(CLASSIFICATION_DEFAULTS)),
            ("graph_merge", set(GRAPH_DEFAULTS)),
        ):
            missing = sorted(required_keys - set(_mapping(config[section_name], section_name)))
            if missing:
                raise ValueError(f"strict release configuration missing {section_name} key(s): {', '.join(missing)}")
        from .feature_flags import FEATURE_DEFAULTS

        required_trees: dict[str, Mapping[str, Any]] = {
            "features": FEATURE_DEFAULTS,
            "association": ASSOCIATION_DEFAULTS,
            "weights_association": ASSOCIATION_WEIGHT_DEFAULTS,
            "weights_graph": GRAPH_WEIGHT_DEFAULTS,
            "beam_aware_classification": CLASSIFICATION_DEFAULTS,
            "graph_merge": GRAPH_DEFAULTS,
            "local_association": {"enabled": True, "local_sanity": LOCAL_SANITY_DEFAULTS},
            "parent_seed_selection": PARENT_SEED_DEFAULTS,
            "host_support": HOST_SUPPORT_DEFAULTS,
            "parent_linking": PARENT_LINK_DEFAULTS,
            "association_types": ASSOCIATION_TYPE_DEFAULTS,
        }
        for section_name, expected in required_trees.items():
            section = _mapping(config.get(section_name), section_name)
            _check_required_tree(section, expected, section_name)
    for key in _ROOT_BOOL_KEYS:
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    features = _mapping(config.get("features", {}), "features")
    for key, value in features.items():
        if not isinstance(value, bool):
            raise ValueError(f"features.{key} must be boolean")

    h5 = _mapping(config.get("h5", {}), "h5")
    for key, value in h5.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(f"h5.{key} must be a string or null")

    matching = _mapping(config.get("matching", MATCHING_DEFAULTS), "matching")
    matching_values = {**MATCHING_DEFAULTS, **matching}
    _finite_number(matching_values["preselect_margin_arcsec"], "matching.preselect_margin_arcsec")
    if float(matching_values["preselect_margin_arcsec"]) < 0:
        raise ValueError("matching.preselect_margin_arcsec must be non-negative")

    for section_name in ("local_association", "parent_seed_selection", "host_support", "parent_linking", "visualization"):
        section = config.get(section_name)
        if section is not None:
            _validate_nested_values(_mapping(section, section_name), section_name, section_name)

    association_types = config.get("association_types")
    if association_types is not None:
        for name, values in _mapping(association_types, "association_types").items():
            for key, value in _mapping(values, f"association_types.{name}").items():
                if key == "enabled":
                    if not isinstance(value, bool):
                        raise ValueError(f"association_types.{name}.enabled must be boolean")
                elif key == "max_quality":
                    if not isinstance(value, str):
                        raise ValueError(f"association_types.{name}.max_quality must be a string")
                else:
                    _finite_number(value, f"association_types.{name}.{key}")

    required = ("snr_thresholds", "min_mask_area_pix", "beam")
    if require_core:
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"missing required configuration key(s): {', '.join(missing)}")

    raw_thresholds = config.get("snr_thresholds", SCIENTIFIC_DEFAULTS["snr_thresholds"])
    if isinstance(raw_thresholds, (str, bytes)) or isinstance(raw_thresholds, Mapping):
        raise ValueError("snr_thresholds must be a finite numeric sequence")
    try:
        thresholds = tuple(_finite_number(value, "snr_thresholds") for value in raw_thresholds)
    except TypeError as exc:
        raise ValueError("snr_thresholds must be a finite numeric sequence") from exc
    if not thresholds:
        raise ValueError("snr_thresholds must be non-empty")
    if any(value <= 0 for value in thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("snr_thresholds must contain unique positive values")
    if any(thresholds[index] <= thresholds[index + 1] for index in range(len(thresholds) - 1)):
        raise ValueError("snr_thresholds must be strictly descending")
    required_levels = {3.0, 2.5, 2.0}
    if not required_levels.issubset(set(thresholds)):
        missing_levels = ", ".join(f"{level:g}" for level in sorted(required_levels - set(thresholds), reverse=True))
        raise ValueError(f"snr_thresholds must include the formal levels 3, 2.5, and 2 sigma; missing: {missing_levels}")

    min_area = _finite_number(config.get("min_mask_area_pix", SCIENTIFIC_DEFAULTS["min_mask_area_pix"]), "min_mask_area_pix")
    if min_area < 1 or int(min_area) != min_area:
        raise ValueError("min_mask_area_pix must be a positive integer")
    raw_connectivity = config.get("connectivity", SCIENTIFIC_DEFAULTS["connectivity"])
    if isinstance(raw_connectivity, bool) or not isinstance(raw_connectivity, int):
        raise ValueError("connectivity must be an integer")
    connectivity = int(raw_connectivity)
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")
    if str(config.get("mean_mode", "median")).lower() not in {"median", "mean", "zero"}:
        raise ValueError("mean_mode must be one of: median, mean, zero")
    if str(config.get("rms_mode", "mad")).lower() not in {"mad", "std"}:
        raise ValueError("rms_mode must be one of: mad, std")
    sigma = _finite_number(config.get("gaussian_smooth_sigma_pix", 1.0), "gaussian_smooth_sigma_pix")
    if sigma < 0:
        raise ValueError("gaussian_smooth_sigma_pix must be >= 0")

    beam = _mapping(config.get("beam", {}), "beam")
    major = _finite_number(beam.get("major_arcsec"), "beam.major_arcsec", positive=True) if beam.get("major_arcsec") is not None else float("nan")
    minor = _finite_number(beam.get("minor_arcsec"), "beam.minor_arcsec", positive=True) if beam.get("minor_arcsec") is not None else float("nan")
    if not isfinite(major) or not isfinite(minor):
        raise ValueError("beam.major_arcsec and beam.minor_arcsec must be provided and positive")
    for key in ("pixel_pa_deg", "ra_axis_sign", "dec_axis_sign"):
        if key in beam and beam[key] is not None:
            value = _finite_number(beam[key], f"beam.{key}")
            if key in {"ra_axis_sign", "dec_axis_sign"} and value == 0:
                raise ValueError(f"beam.{key} must be non-zero when provided")
    if "pixel_scale_arcsec" in config and config["pixel_scale_arcsec"] is not None:
        pixel_scale = _finite_number(config["pixel_scale_arcsec"], "pixel_scale_arcsec", positive=True)
    else:
        pixel_scale = None

    association_raw = _mapping(config.get("association", {}), "association")
    association_values = {**ASSOCIATION_DEFAULTS, **association_raw}
    for key in _ASSOCIATION_BOOL_KEYS:
        if not isinstance(association_values[key], bool):
            raise ValueError(f"association.{key} must be boolean")
    quality = _mapping(association_values.get("quality_thresholds", {}), "association.quality_thresholds")
    _check_unknown(quality, _SECTION_KEYS["quality_thresholds"], "association.quality_thresholds")
    for key in ("max_pair_distance_beam", "threshold_strong", "threshold_weak"):
        _finite_number(association_values[key], f"association.{key}", positive=True)
    for key, value in association_values.items():
        if key in _ASSOCIATION_BOOL_KEYS or key == "quality_thresholds":
            continue
        if key == "max_pair_distance_arcsec" and value is None:
            continue
        number = _finite_number(value, f"association.{key}")
        if "fraction" in key and not 0.0 <= number <= 1.0:
            raise ValueError(f"association.{key} must be within [0, 1]")
    max_arcsec = association_values.get("max_pair_distance_arcsec")
    if max_arcsec is not None:
        _finite_number(max_arcsec, "association.max_pair_distance_arcsec", positive=True)
    if float(association_values["threshold_strong"]) < float(association_values["threshold_weak"]):
        raise ValueError("association.threshold_strong must be >= threshold_weak")
    quality_values = {**ASSOCIATION_DEFAULTS["quality_thresholds"], **quality}
    for key, value in quality_values.items():
        _finite_number(value, f"association.quality_thresholds.{key}")
    if not quality_values["high"] >= quality_values["medium"] >= quality_values["low"]:
        raise ValueError("association.quality_thresholds must be ordered high >= medium >= low")

    if "graph_merge" in config:
        graph = _mapping(config["graph_merge"], "graph_merge")
        for key in GRAPH_DEFAULTS:
            _finite_number(graph.get(key), f"graph_merge.{key}", positive=True)

    local_association = _mapping(config.get("local_association", {}), "local_association")
    local_sanity = _mapping(local_association.get("local_sanity", {}), "local_association.local_sanity")
    for key in ("max_ridge_gap_fraction", "max_weak_edge_chain_fraction", "max_only_2sigma_edge_fraction"):
        if key in local_sanity:
            value = _finite_number(local_sanity[key], f"local_association.local_sanity.{key}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"local_association.local_sanity.{key} must be within [0, 1]")
    if "min_saddle_to_peak_ratio" in local_sanity:
        value = _finite_number(local_sanity["min_saddle_to_peak_ratio"], "local_association.local_sanity.min_saddle_to_peak_ratio")
        if not 0.0 <= value <= 1.0:
            raise ValueError("local_association.local_sanity.min_saddle_to_peak_ratio must be within [0, 1]")

    parent_seed = _mapping(config.get("parent_seed_selection", {}), "parent_seed_selection")
    midpoint = _mapping(parent_seed.get("midpoint_core", {}), "parent_seed_selection.midpoint_core")
    if "axial_fraction_min" in midpoint and "axial_fraction_max" in midpoint:
        fraction_min = _finite_number(midpoint["axial_fraction_min"], "parent_seed_selection.midpoint_core.axial_fraction_min")
        fraction_max = _finite_number(midpoint["axial_fraction_max"], "parent_seed_selection.midpoint_core.axial_fraction_max")
        if not 0.0 <= fraction_min < fraction_max <= 1.0:
            raise ValueError("parent_seed_selection.midpoint_core axial fractions must satisfy 0 <= min < max <= 1")
    for key in ("max_parent_candidates_per_group", "max_parent_candidates_per_cutout", "min_endpoint_n_gaussians"):
        if key in parent_seed:
            value = _finite_number(parent_seed[key], f"parent_seed_selection.{key}")
            if value < 1 or int(value) != value:
                raise ValueError(f"parent_seed_selection.{key} must be a positive integer")

    host_support = _mapping(config.get("host_support", {}), "host_support")
    host_enabled = bool(host_support.get("enabled", HOST_SUPPORT_DEFAULTS["enabled"]))
    host_required = bool(
        host_support.get(
            "require_host_for_parent_link",
            HOST_SUPPORT_DEFAULTS["require_host_for_parent_link"],
        )
    )
    if host_required and not host_enabled:
        raise ValueError(
            "host_support.require_host_for_parent_link cannot be true when host_support.enabled is false"
        )
    if "min_search_radius_arcsec" in host_support and "max_search_radius_arcsec" in host_support:
        min_radius = _finite_number(host_support["min_search_radius_arcsec"], "host_support.min_search_radius_arcsec", positive=True)
        max_radius = _finite_number(host_support["max_search_radius_arcsec"], "host_support.max_search_radius_arcsec", positive=True)
        if min_radius > max_radius:
            raise ValueError("host_support.min_search_radius_arcsec must not exceed max_search_radius_arcsec")
    if "search_radius_fraction_of_sep" in host_support:
        fraction = _finite_number(host_support["search_radius_fraction_of_sep"], "host_support.search_radius_fraction_of_sep")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("host_support.search_radius_fraction_of_sep must be within (0, 1]")

    for section_name in ("weights_association", "weights_graph"):
        for key, value in _mapping(config.get(section_name, {}), section_name).items():
            _finite_number(value, f"{section_name}.{key}")
    classification = _mapping(config.get("beam_aware_classification", {}), "beam_aware_classification")
    for key, value in classification.items():
        number = _finite_number(value, f"beam_aware_classification.{key}")
        if "fraction" in key or "tolerance" in key or key.startswith("pa_weight"):
            if number < 0 or ("fraction" in key and number > 1):
                raise ValueError(f"beam_aware_classification.{key} is outside its valid range")
    return PipelineConfig(
        snr_thresholds=thresholds,
        min_mask_area_pix=int(min_area),
        connectivity=connectivity,
        pixel_scale_arcsec=pixel_scale,
        beam=BeamConfig(major, minor, _finite_number(beam.get("pa_deg", 0.0), "beam.pa_deg")),
        association=AssociationConfig(
            max_pair_distance_beam=float(association_values["max_pair_distance_beam"]),
            max_pair_distance_arcsec=None if max_arcsec is None else float(max_arcsec),
            threshold_strong=float(association_values["threshold_strong"]),
            threshold_weak=float(association_values["threshold_weak"]),
        ),
    )


def validate_mapping(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate a user mapping and return its canonical effective mapping.

    Production entry points consume the resolved mapping returned here, so
    validation and provenance always describe the same configuration.
    """

    if config is None or not isinstance(config, Mapping):
        raise ValueError("configuration must be a mapping")
    parse_pipeline_config(config)
    # Imported lazily to avoid a config <-> utils import cycle.
    from .utils import resolve_effective_config

    resolved = resolve_effective_config(config)
    # Validate after merging so malformed optional sections fail before any
    # output is written while the strict release contract still applies to
    # user-supplied YAML.
    parse_pipeline_config(resolved)
    return resolved
