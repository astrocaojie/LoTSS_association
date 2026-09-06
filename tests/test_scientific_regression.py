from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from lotss_association.association import _flux_ratio, compute_pair_association_features, run_component_association
from lotss_association.graph_merge import _flux_ratio_score
from lotss_association.measurements import measure_merged_sources
from lotss_association.utils import load_yaml, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _stage1_reference_inputs() -> tuple[dict, SimpleNamespace, SimpleNamespace, pd.DataFrame]:
    config = validate_config(load_yaml(PROJECT_ROOT / "configs/default.yaml"))
    image = np.zeros((40, 80), dtype=float)
    image[18:23, 8:68] = 4.0
    thresholds = np.asarray(config["snr_thresholds"], dtype=float)
    labels = [ndi.label(image >= threshold, structure=np.ones((3, 3), dtype=int))[0] for threshold in thresholds]
    segmentation = SimpleNamespace(
        snr_map=image,
        thresholds=thresholds,
        labels_by_threshold=np.stack(labels),
    )
    components = pd.DataFrame(
        [
            {
                "component_index": index,
                "_gaussian_id": f"g{index}",
                "x": float(x),
                "y": 20.0,
                "pixel_scale_arcsec": 1.0,
                "_total_flux": float(10 - index),
                "_peak_flux": float(8 - index),
                "_maj": 6.0,
                "_min": 6.0,
                "_pa": 0.0,
                "_dc_maj": np.nan,
                "_dc_min": np.nan,
                "_dc_pa": np.nan,
                "_ra": 1.0,
                "_dec": 1.0,
                "_island_id": 1,
                "cutout_id": "c0",
                "cutout_index": 0,
            }
            for index, x in enumerate((10, 35, 60))
        ]
    )
    cutout = SimpleNamespace(
        image=image,
        rms=1.0,
        mean=0.0,
        cutout_id="c0",
        index=0,
        metadata={"pixel_scale_arcsec": 1.0},
        wcs=None,
        ra=None,
        dec=None,
    )
    return config, cutout, segmentation, components


def test_stage1_reference_fixture_is_deterministic() -> None:
    """The minimal reference scene keeps grouping decisions stable across runs."""

    config, cutout, segmentation, components = _stage1_reference_inputs()

    first = run_component_association(cutout, segmentation, components, config)
    second = run_component_association(cutout, segmentation, components, config)
    assert len(first.groups) == len(second.groups) == 1
    assert len(first.edges) == len(second.edges) == 3
    assert int((first.edges["edge_type"] == "strong").sum()) == 3
    assert int((first.edges["edge_type"] == "weak").sum()) == 0
    assert int((first.edges["edge_type"] == "rejected").sum()) == 0
    summary_columns = ["association_group_id", "component_ids"]
    first_hash = sha256(first.groups[summary_columns].to_csv(index=False).encode()).hexdigest()
    second_hash = sha256(second.groups[summary_columns].to_csv(index=False).encode()).hexdigest()
    assert first_hash == second_hash
    assert first_hash == "1b10544e0448150461ddd23458e7abfcc445aced801ea262aeed664e8bf861c6"

    local_diagnostic = first.local_sanity_diagnostics.iloc[0]
    assert local_diagnostic[
        [
            "n_gaussians_before",
            "n_groups_after_split",
            "split_applied",
            "local_overmerge_risk_score",
            "saddle_to_peak_ratio",
            "ridge_gap_fraction",
            "weak_edge_fraction",
            "only_2sigma_edge_fraction",
        ]
    ].to_dict() == {
        "n_gaussians_before": 3,
        "n_groups_after_split": 1,
        "split_applied": False,
        "local_overmerge_risk_score": 0.35,
        "saddle_to_peak_ratio": 1.0,
        "ridge_gap_fraction": 0.0,
        "weak_edge_fraction": 0.0,
        "only_2sigma_edge_fraction": 0.0,
    }

    without_sanity = deepcopy(config)
    without_sanity["local_association"]["local_sanity"]["enabled"] = False
    original = run_component_association(cutout, segmentation, components, without_sanity)
    preserved_columns = [
        "morphology_class",
        "resolved_probability",
        "beam_like_score",
        "classification_reason",
        "artifact_risk_flags",
    ]
    pd.testing.assert_frame_equal(
        first.groups[preserved_columns].reset_index(drop=True),
        original.groups[preserved_columns].reset_index(drop=True),
    )


def test_stage1_flux_evidence_preserves_valid_scores_and_rejects_invalid_values() -> None:
    config, cutout, segmentation, components = _stage1_reference_inputs()

    valid = compute_pair_association_features(
        components.iloc[0],
        components.iloc[1],
        cutout.image,
        segmentation.snr_map,
        segmentation,
        config,
    )
    assert valid["flux_ratio"] == pytest.approx(0.9)
    assert valid["flux_continuity_score"] == pytest.approx(np.sqrt(0.9))
    assert valid["association_score"] == pytest.approx(5.928306470656955)

    peak_fallback_i = pd.Series({"_total_flux": np.nan, "_peak_flux": 8.0})
    peak_fallback_j = pd.Series({"_total_flux": 4.0, "_peak_flux": 2.0})
    assert _flux_ratio(peak_fallback_i, peak_fallback_j) == pytest.approx(0.25)

    mixed_measurements_i = pd.Series({"_total_flux": 8.0, "_peak_flux": np.nan})
    mixed_measurements_j = pd.Series({"_total_flux": np.nan, "_peak_flux": 2.0})
    assert np.isnan(_flux_ratio(mixed_measurements_i, mixed_measurements_j))

    for invalid_flux in (0.0, -1.0, np.nan, np.inf):
        invalid_components = components.copy()
        invalid_components.loc[0, ["_total_flux", "_peak_flux"]] = invalid_flux
        features = compute_pair_association_features(
            invalid_components.iloc[0],
            invalid_components.iloc[1],
            cutout.image,
            segmentation.snr_map,
            segmentation,
            config,
        )
        assert np.isnan(features["flux_ratio"])
        assert features["flux_continuity_score"] == 0.0
        assert np.isfinite(features["association_score"])
        assert np.isnan(_flux_ratio_score(invalid_components.iloc[0], invalid_components.iloc[1]))


def test_stage1_split_rewrites_edges_and_remeasures_final_groups(monkeypatch) -> None:
    from lotss_association import association as association_module, local_sanity as local_sanity_module
    from scripts.run_pipeline import association_diagnostic_record

    config, cutout, segmentation, components = _stage1_reference_inputs()
    compute_features = association_module.compute_pair_association_features

    def edge_features(*args, **kwargs):
        record = compute_features(*args, **kwargs)
        if 0 in {int(record["component_index_1"]), int(record["component_index_2"])}:
            record["artifact_risk_flags"] = "sidelobe_risk"
        return record

    def force_split(_group, _components, _edges, features, _config):
        assert features["component_nodes"] == [0, 1, 2]
        return [[0], [1, 2]], "regression_split", True

    monkeypatch.setattr(association_module, "compute_pair_association_features", edge_features)
    monkeypatch.setattr(local_sanity_module, "split_overmerged_local_group", force_split)

    result = run_component_association(cutout, segmentation, components, config)

    assert result.clusters == [[0], [1, 2]]
    assert result.groups["component_ids"].tolist() == ["0", "1,2"]
    assert result.groups["association_group_id"].tolist() == ["c0_l000", "c0_l001"]
    assert result.groups["original_association_group_id"].tolist() == ["c0_a000", "c0_a000"]

    cut_edges = result.edges[
        result.edges["component_index_1"].eq(0) | result.edges["component_index_2"].eq(0)
    ]
    assert not cut_edges["association_decision"].any()
    assert set(cut_edges["edge_type"]) == {"rejected"}
    assert set(cut_edges["rejection_reason"]) == {"local_sanity_split"}
    assert set(cut_edges["local_edge_type"]) == {"cut"}
    assert not cut_edges["local_edge_decision"].any()

    retained_edge = result.edges[
        result.edges["component_index_1"].eq(1) & result.edges["component_index_2"].eq(2)
    ].iloc[0]
    assert retained_edge["association_decision"]
    assert retained_edge["edge_type"] == "strong"

    singleton = result.groups[result.groups["component_ids"].eq("0")].iloc[0]
    component = result.components[result.components["component_index"].eq(0)].iloc[0]
    assert singleton["morphology_class"] == component["morphology_class"] == "unresolved"
    assert singleton["classification_reason"] == component["classification_reason"]
    assert singleton["artifact_risk_flags"] == ""
    assert result.groups["n_strong_edges"].tolist() == [0, 1]

    component_groups = result.components.set_index("component_index")["association_group_id"].to_dict()
    assert component_groups == {0: "c0_l000", 1: "c0_l001", 2: "c0_l001"}
    assert result.components["original_association_group_id"].tolist() == ["c0_a000"] * 3

    diagnostic = association_diagnostic_record("c0", result.components, result.groups, result.edges)
    assert diagnostic["n_strong_edges"] == 1
    assert diagnostic["n_rejected_edges"] == 2
    assert diagnostic["n_decision_edges"] == 1


def test_merged_source_support_matches_group_catalog_2p5sigma_preference() -> None:
    """Legacy merged sources must pick support masks like the group catalogue."""

    image = np.zeros((6, 6), dtype=float)
    segmentation = SimpleNamespace(
        thresholds=np.asarray([2.5, 2.0]),
        labels_by_threshold=np.asarray(
            [
                [[1, 1, 0, 0, 0, 0]] * 2 + [[0] * 6] * 4,
                [[1, 1, 1, 0, 0, 0]] * 3 + [[0] * 6] * 3,
            ]
        ),
    )
    components = pd.DataFrame(
        [
            {
                "component_index": 0,
                "_gaussian_id": "g0",
                "_island_id": 1,
                "_total_flux": 2.0,
                "_peak_flux": 1.0,
                "_pa": 0.0,
                "x": 1.0,
                "y": 1.0,
                "pixel_scale_arcsec": 1.0,
                "label_at_2sigma": 1,
                "label_at_2p5sigma": 1,
            }
        ]
    )
    cutout = SimpleNamespace(cutout_id="c0", image=image, wcs=None)
    measured = measure_merged_sources(cutout, segmentation, components, [[0]], pd.DataFrame())
    # The 2.5 sigma label spans x in [0, 1]; the 2 sigma label spans x in [0, 2].
    assert measured.loc[0, "bounding_box"] == "0,0,1,1"


def test_stage1_split_reprojects_subgroup_sky_coordinates() -> None:
    """Split subgroups must report their own centroid sky position."""

    from astropy.wcs import WCS

    config, cutout, segmentation, components = _stage1_reference_inputs()
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [20.0, 20.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.crval = [150.0, 2.0]
    cutout.wcs = wcs

    from lotss_association import local_sanity as local_sanity_module

    original_split = local_sanity_module.split_overmerged_local_group

    def force_split(_group, _components, _edges, features, _config):
        assert features["component_nodes"] == [0, 1, 2]
        return [[0], [1, 2]], "regression_split", True

    local_sanity_module.split_overmerged_local_group = force_split
    try:
        result = run_component_association(cutout, segmentation, components, config)
    finally:
        local_sanity_module.split_overmerged_local_group = original_split

    groups = result.groups.set_index("association_group_id")
    singleton = groups.loc["c0_l000"]
    pair = groups.loc["c0_l001"]
    assert np.isfinite(singleton["ra"]) and np.isfinite(pair["ra"])
    # RA decreases with pixel x (CDELT1 < 0): the singleton at x=10 sits at a
    # higher RA than the pair centroid near x=47.5.
    assert singleton["ra"] > pair["ra"]
    expected_ra, expected_dec = wcs.celestial.pixel_to_world_values(10.0, 20.0)
    assert singleton["ra"] == pytest.approx(expected_ra)
    assert singleton["dec"] == pytest.approx(expected_dec)
