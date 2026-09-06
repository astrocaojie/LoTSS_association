from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lotss_association.parent_links import (
    _classify_near_boundary_rescue,
    _near_boundary_strong_support,
    _query_position_hosts,
    _source_morph,
    _symmetry_scores,
    parent_link_config,
    run_parent_links,
)
from lotss_association.parent_seed import (
    _candidate_search_pairs,
    _compute_pair,
    _ratio_score,
    _robust_bbox_for_group,
    run_parent_seed,
)
from lotss_association.utils import load_yaml, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def release_config() -> dict:
    return validate_config(load_yaml(PROJECT_ROOT / "configs/default.yaml"))


def synthetic_parent_inputs() -> tuple[pd.DataFrame, pd.DataFrame, SimpleNamespace]:
    """Return two deterministic extended endpoints and their segmentation."""

    snr = np.full((80, 130), 12.0, dtype=float)
    labels = []
    for _threshold in (3.0, 5.0):
        label_map = np.zeros_like(snr, dtype=np.int32)
        label_map[30:50, 10:30] = 1
        label_map[30:50, 80:100] = 2
        labels.append(label_map)
    segmentation = SimpleNamespace(
        snr_map=snr,
        thresholds=np.asarray([3.0, 5.0], dtype=float),
        labels_by_threshold=np.stack(labels),
    )
    groups = pd.DataFrame(
        [
            {
                "association_group_id": "g1",
                "n_gaussians": 2,
                "LAS_beam": 4.0,
                "total_flux_gaussian": 10.0,
                "peak_flux": 8.0,
                "group_PA": 0.0,
                "component_ids": "0",
                "bounding_box": "10,30,29,49",
                "centroid_x": 19.5,
                "centroid_y": 39.5,
                "pixel_scale_arcsec": 1.0,
            },
            {
                "association_group_id": "g2",
                "n_gaussians": 2,
                "LAS_beam": 4.0,
                "total_flux_gaussian": 10.0,
                "peak_flux": 8.0,
                "group_PA": 0.0,
                "component_ids": "1",
                "bounding_box": "80,30,99,49",
                "centroid_x": 89.5,
                "centroid_y": 39.5,
                "pixel_scale_arcsec": 1.0,
            },
        ]
    )
    components = pd.DataFrame(
        [
            {"component_index": 0, "x": 20.0, "y": 40.0, "pixel_scale_arcsec": 1.0},
            {"component_index": 1, "x": 90.0, "y": 40.0, "pixel_scale_arcsec": 1.0},
        ]
    )
    return groups, components, segmentation


def high_seed_parent_inputs() -> tuple[pd.DataFrame, pd.DataFrame, SimpleNamespace]:
    groups, components, segmentation = synthetic_parent_inputs()
    groups.loc[1, ["bounding_box", "centroid_x"]] = ["50,30,69,49", 59.5]
    components.loc[1, "x"] = 60.0
    segmentation.labels_by_threshold[:, 30:50, 80:100] = 0
    segmentation.labels_by_threshold[:, 30:50, 50:70] = 2
    return groups, components, segmentation


class _EmptyHostClient:
    def __init__(self) -> None:
        self.catalogues: list[str] = []

    def query_catalogue(self, ra: float, dec: float, radius_arcsec: float, catalogue: str):
        self.catalogues.append(catalogue)
        return SimpleNamespace(
            results=pd.DataFrame(),
            log=pd.DataFrame(
                [
                    {
                        "ra": ra,
                        "dec": dec,
                        "radius_arcsec": radius_arcsec,
                        "catalogue": catalogue,
                        "status": "empty",
                        "n_results": 0,
                    }
                ]
            ),
        )


def test_parent_seed_generates_deterministic_candidate_and_scores_missing_ratios() -> None:
    config = release_config()
    groups, components, segmentation = synthetic_parent_inputs()
    first = run_parent_seed("c0", groups, components, config, segmentation)
    second = run_parent_seed("c0", groups, components, config, segmentation)

    assert len(first.parent_seed_table) == 2
    assert first.parent_seed_table["is_parent_seed"].all()
    assert first.parent_seed_table["robust_bbox_source"].tolist() == ["3sigma_bbox", "3sigma_bbox"]
    assert len(first.candidates) == 1
    assert first.candidates.iloc[0]["parent_candidate_id"] == "c0_seed_pc000"
    assert first.candidates.iloc[0]["parent_candidate_quality"] in {"high", "medium"}
    pd.testing.assert_frame_equal(first.candidates, second.candidates)
    assert _ratio_score(float("nan"), 5.0) == pytest.approx(0.35)


def test_run_parent_links_requires_host_for_radio_candidate() -> None:
    config = release_config()
    groups, components, segmentation = high_seed_parent_inputs()
    client = _EmptyHostClient()

    result = run_parent_links(
        "c0",
        groups,
        components,
        segmentation,
        config,
        client,
        {"count": 0},
    )

    edge = result.edges_debug.iloc[0]
    assert edge["independent_parent_evidence"] == "residual_radio_bridge"
    assert edge["parent_acceptance_class"] == "rejected_parent_candidate"
    assert edge["parent_acceptance_reason"] == "no_midpoint_host"
    assert result.candidates.empty
    assert client.catalogues


def test_run_parent_links_accepts_radio_candidate_when_host_is_optional() -> None:
    config = release_config()
    config["host_support"]["require_host_for_parent_link"] = False
    groups, components, segmentation = high_seed_parent_inputs()

    result = run_parent_links(
        "c0",
        groups,
        components,
        segmentation,
        config,
        _EmptyHostClient(),
        {"count": 0},
    )

    edge = result.edges_debug.iloc[0]
    assert edge["parent_acceptance_class"] == "accepted_high_confidence_parent"
    assert edge["parent_candidate_quality"] == "high"
    assert edge["no_host_detected"]
    assert len(result.candidates) == 1


def test_run_parent_links_accepts_radio_candidate_when_host_is_disabled() -> None:
    config = release_config()
    config["host_support"].update(
        {"enabled": False, "require_host_for_parent_link": False}
    )
    groups, components, segmentation = high_seed_parent_inputs()
    client = _EmptyHostClient()

    result = run_parent_links(
        "c0",
        groups,
        components,
        segmentation,
        config,
        client,
        {"count": 0},
    )

    edge = result.edges_debug.iloc[0]
    assert edge["host_status"] == "disabled"
    assert edge["parent_acceptance_class"] == "accepted_high_confidence_parent"
    assert edge["parent_candidate_quality"] == "high"
    assert len(result.candidates) == 1
    assert client.catalogues == []


def test_high_seed_quality_does_not_determine_final_parent_quality() -> None:
    config = release_config()
    groups, components, segmentation = high_seed_parent_inputs()
    seed = run_parent_seed("c0", groups, components, config, segmentation)

    final = run_parent_links(
        "c0",
        groups,
        components,
        segmentation,
        config,
        _EmptyHostClient(),
        {"count": 0},
    )

    assert seed.candidates.iloc[0]["parent_candidate_quality"] == "high"
    assert final.edges_debug.iloc[0]["parent_candidate_quality"] == "rejected"
    assert final.candidates.empty


def test_endpoint_thresholds_control_seed_and_final_endpoint_eligibility() -> None:
    default_config = release_config()
    strict_config = release_config()
    strict_config["parent_linking"]["endpoint_thresholds"]["noise_peak_snr_min"] = 13.0
    groups, components, segmentation = synthetic_parent_inputs()
    groups = groups.iloc[[0]].copy()
    groups.loc[:, "n_gaussians"] = 1
    groups.loc[:, "axis_ratio"] = 2.0
    groups.loc[:, "association_quality"] = "low"
    groups.loc[:, "association_type"] = "weak_association"
    components = components.iloc[[0]].copy()

    default_seed = run_parent_seed("c0", groups, components, default_config, segmentation)
    strict_seed = run_parent_seed("c0", groups, components, strict_config, segmentation)
    default_final = run_parent_links(
        "c0", groups, components, segmentation, default_config, object(), {"count": 0}
    )
    strict_final = run_parent_links(
        "c0", groups, components, segmentation, strict_config, object(), {"count": 0}
    )

    assert bool(default_seed.parent_seed_table.iloc[0]["is_parent_seed"])
    assert not bool(strict_seed.parent_seed_table.iloc[0]["is_parent_seed"])
    assert bool(default_final.source_morph_table.iloc[0]["is_parent_endpoint_allowed"])
    assert not bool(strict_final.source_morph_table.iloc[0]["is_parent_endpoint_allowed"])
    assert "peak_snr_below_threshold" in strict_final.source_morph_table.iloc[0]["endpoint_veto_reason"]


def test_parent_seed_uses_five_sigma_bbox_when_three_sigma_is_absent() -> None:
    config = release_config()
    groups, components, segmentation = synthetic_parent_inputs()
    segmentation.labels_by_threshold[0, ...] = 0
    row = groups.iloc[0]

    robust = _robust_bbox_for_group(row, components, segmentation, config)

    assert robust["robust_bbox_source"] == "5sigma_core"
    assert robust["robust_bbox"] == (10.0, 30.0, 29.0, 49.0)
    assert robust["missing_area_3sigma_beam"] is True


def test_parent_seed_candidate_search_filters_bbox_gap_and_is_sorted() -> None:
    groups = pd.DataFrame(
        {
            "robust_bbox": ["0,0,9,9", "20,0,29,9", "12,0,19,9", "bad"],
        }
    )

    assert _candidate_search_pairs(groups, max_gap_pix=3.0) == [(0, 2), (1, 2)]


def test_parent_seed_rejects_missing_pixel_scale_instead_of_guessing() -> None:
    config = release_config()
    groups, components, segmentation = synthetic_parent_inputs()
    groups = groups.drop(columns=["pixel_scale_arcsec"])
    components = components.drop(columns=["pixel_scale_arcsec"])
    config["pixel_scale_arcsec"] = None

    with pytest.raises(ValueError, match="pixel scale"):
        run_parent_seed("c0", groups, components, config, segmentation)


def test_endpoint_hard_gates_are_not_counted_as_extension_indicators() -> None:
    cfg = parent_link_config(release_config())
    row = pd.Series(
        {
            "n_gaussians": 1,
            "LAS_beam": 4.0,
            "peak_snr": 10.0,
            "association_quality": "low",
            "association_type": "weak_association",
        }
    )
    artifact = {"artifact_environment_score": 0.0}

    one_indicator = _source_morph(row, 1.6, 1.6, 1.1, 1.8, 1.6, artifact, cfg)
    two_indicators = _source_morph(row, 3.0, 3.0, 1.1, 1.8, 1.6, artifact, cfg)

    assert one_indicator["is_lobe_candidate"] is False
    assert two_indicators["is_lobe_candidate"] is True


@pytest.mark.parametrize(
    ("peak_snr", "area_3sigma", "reason"),
    [
        (float("nan"), 3.0, "peak_snr_missing"),
        (10.0, float("nan"), "area_3sigma_beam_missing"),
        (5.9, 3.0, "peak_snr_below_threshold"),
        (10.0, 1.4, "area_3sigma_beam_below_threshold"),
    ],
)
def test_endpoint_hard_gates_block_standard_and_rescue_paths(
    peak_snr: float,
    area_3sigma: float,
    reason: str,
) -> None:
    cfg = parent_link_config(release_config())
    row = pd.Series(
        {
            "n_gaussians": 2,
            "LAS_beam": 4.0,
            "peak_snr": peak_snr,
            "association_quality": "high",
            "association_type": "continuous_extended",
        }
    )

    result = _source_morph(row, area_3sigma, 3.0, 2.0, 3.0, 1.5, {"artifact_environment_score": 0.0}, cfg)

    assert result["is_lobe_candidate"] is False
    assert result["near_extended_lobe_candidate"] is False
    assert result["endpoint_veto_final"] is True
    assert reason in result["endpoint_veto_reason"]


def test_parent_pair_facing_uses_inward_bbox_extent_independently_of_pa() -> None:
    config = release_config()
    groups = pd.DataFrame(
        [
            {
                "association_group_id": "g1",
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "group_PA": 0.0,
                "bounding_box": "-2,-1,3,1",
                "pixel_scale_arcsec": 1.0,
                "n_gaussians": 2,
                "LAS_beam": 4.0,
                "total_flux_gaussian": 10.0,
                "peak_flux": 8.0,
            },
            {
                "association_group_id": "g2",
                "centroid_x": 10.0,
                "centroid_y": 0.0,
                "group_PA": 0.0,
                "bounding_box": "7,-1,12,1",
                "pixel_scale_arcsec": 1.0,
                "n_gaussians": 2,
                "LAS_beam": 4.0,
                "total_flux_gaussian": 10.0,
                "peak_flux": 8.0,
            },
        ]
    )
    seeds = {
        row["association_group_id"]: pd.Series(
            {"robust_bbox": row["bounding_box"], "is_parent_seed": True, "is_compact_singleton": False}
        )
        for _, row in groups.iterrows()
    }

    record = _compute_pair("c0", 0, groups.iloc[0], groups.iloc[1], seeds["g1"], seeds["g2"], groups, seeds, config)

    assert record["axis_alignment_score"] == pytest.approx(1.0)
    assert record["facing_score"] == pytest.approx(0.6)


def test_midpoint_symmetry_uses_compact_midpoint_core_not_flux_or_size_balance() -> None:
    config = release_config()
    groups = pd.DataFrame(
        [
            {
                "association_group_id": "g1",
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "group_PA": 0.0,
                "bounding_box": "-2,-1,2,1",
                "pixel_scale_arcsec": 1.0,
                "n_gaussians": 2,
                "LAS_beam": 4.0,
                "total_flux_gaussian": 10.0,
                "peak_flux": 8.0,
            },
            {
                "association_group_id": "g2",
                "centroid_x": 10.0,
                "centroid_y": 0.0,
                "group_PA": 0.0,
                "bounding_box": "8,-1,12,1",
                "pixel_scale_arcsec": 1.0,
                "n_gaussians": 2,
                "LAS_beam": 4.0,
                "total_flux_gaussian": 10.0,
                "peak_flux": 8.0,
            },
        ]
    )
    endpoint_seeds = {
        row["association_group_id"]: pd.Series(
            {"robust_bbox": row["bounding_box"], "is_parent_seed": True, "is_compact_singleton": False}
        )
        for _, row in groups.iterrows()
    }

    without_core = _compute_pair(
        "c0", 0, groups.iloc[0], groups.iloc[1], endpoint_seeds["g1"], endpoint_seeds["g2"], groups, endpoint_seeds, config
    )
    _, midpoint_without_core, flux_score, size_score = _symmetry_scores(without_core, parent_link_config(config))
    assert flux_score == pytest.approx(1.0)
    assert size_score == pytest.approx(1.0)
    assert midpoint_without_core == pytest.approx(0.5)
    assert without_core["midpoint_symmetry_source"] == "no_midpoint_core"

    core = {
        "association_group_id": "g3",
        "centroid_x": 5.0,
        "centroid_y": 0.0,
        "group_PA": 0.0,
        "bounding_box": "4,-1,6,1",
        "pixel_scale_arcsec": 1.0,
        "n_gaussians": 1,
        "LAS_beam": 1.0,
        "total_flux_gaussian": 1.0,
        "peak_flux": 1.0,
    }
    groups_with_core = pd.concat([groups, pd.DataFrame([core])], ignore_index=True)
    seeds_with_core = {**endpoint_seeds, "g3": pd.Series({"is_compact_singleton": True})}
    with_core = _compute_pair(
        "c0",
        0,
        groups_with_core.iloc[0],
        groups_with_core.iloc[1],
        endpoint_seeds["g1"],
        endpoint_seeds["g2"],
        groups_with_core,
        seeds_with_core,
        config,
    )

    assert with_core["core_candidate_near_midpoint"] is True
    assert with_core["midpoint_symmetry_source"] == "compact_midpoint_core"
    assert with_core["midpoint_symmetry_score"] == pytest.approx(1.0)


def test_near_boundary_rescue_requires_extended_support_and_valid_endpoints() -> None:
    config = release_config()
    from lotss_association.parent_links import parent_link_config

    cfg = parent_link_config(config)
    record = {
        "near_boundary_pair": True,
        "axis_alignment_score": 0.8,
        "facing_score": 0.8,
        "flux_ratio": 2.0,
        "size_ratio": 2.0,
        "same_3sigma_region_as_neighbor": True,
    }
    endpoint = pd.Series(
        {
            "is_lobe_candidate": False,
            "near_extended_lobe_candidate": True,
            "endpoint_veto_final": False,
            "hard_point_source_veto": False,
            "hard_compact_veto": False,
            "isolated_compact_veto": False,
            "noise_artifact_veto": False,
            "is_artifact_risk": False,
        }
    )
    accepted, reason = _classify_near_boundary_rescue(record, endpoint, endpoint, cfg)
    assert accepted is True
    assert reason == ""

    vetoed = endpoint.copy()
    vetoed["noise_artifact_veto"] = True
    accepted, reason = _classify_near_boundary_rescue(record, vetoed, endpoint, cfg)
    assert accepted is False
    assert reason == "endpoint_vetoed_or_not_extended_for_near_boundary_rescue"


def test_near_boundary_rescue_consumes_configured_area_threshold() -> None:
    config = release_config()
    from lotss_association.parent_links import parent_link_config

    cfg = parent_link_config(config)
    record = {"area_3sigma_beam_1": 3.0, "area_3sigma_beam_2": 3.0}
    endpoints = pd.Series({"n_gaussians": 1, "is_lobe_candidate": False})

    cfg["rescue_thresholds"]["min_area_each_beam"] = 2.5
    assert _near_boundary_strong_support(record, endpoints, endpoints, cfg) is True
    cfg["rescue_thresholds"]["min_area_each_beam"] = 3.5
    assert _near_boundary_strong_support(record, endpoints, endpoints, cfg) is False


class _FakeHostClient:
    def __init__(self) -> None:
        self.catalogues: list[str] = []

    def query_catalogue(self, ra: float, dec: float, radius_arcsec: float, catalogue: str):
        self.catalogues.append(catalogue)
        if catalogue == "catwise2020":
            result = pd.DataFrame()
            status = "empty"
        else:
            result = pd.DataFrame([{"catalogue": catalogue, "host_id": "wise-1", "host_ra": ra, "host_dec": dec}])
            status = "ok"
        log = pd.DataFrame(
            [{"catalogue": catalogue, "status": status, "n_results": len(result)}]
        )
        return SimpleNamespace(results=result, log=log)


def test_host_query_falls_back_to_allwise_after_empty_catwise() -> None:
    client = _FakeHostClient()
    cfg = {"catalog_priority": ["catwise2020", "allwise"]}
    state = {"count": 0}


    results, logs, status, failed = _query_position_hosts(
        1.0,
        2.0,
        10.0,
        client,
        cfg,
        state,
        max_host_queries=5,
    )

    assert client.catalogues == ["catwise2020", "allwise"]
    assert len(results) == 1
    assert status == "allwise_results"
    assert failed is False
    assert logs["catalogue"].tolist() == ["catwise2020", "allwise"]
