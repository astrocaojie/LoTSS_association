import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.table import Table

from lotss_association import __version__
from lotss_association.catalog import (
    normalized_gaussian_dataframe,
    read_gaussian_catalog,
    validate_normalized_gaussian_dataframe,
)
from lotss_association.io import Cutout
from lotss_association.morphology import beam_like_score, classify_gaussian_component
from lotss_association.parent_links import _noise_artifact_veto, parent_link_config
from lotss_association.utils import load_yaml, packaged_config_uri, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_are_consistent() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = re.search(r"^version:\s*(.+)$", citation, re.MULTILINE)
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert package_version is not None and citation_version is not None
    release_version = package_version.group(1)
    assert __version__ == release_version == citation_version.group(1).strip()
    assert f"## {release_version} - Unreleased" in changelog


def test_packaged_configs_load_and_cli_defaults_stay_out_of_site_packages(monkeypatch, tmp_path) -> None:
    for filename in ("default.yaml",):
        packaged = load_yaml(packaged_config_uri(filename))
        source = load_yaml(PROJECT_ROOT / "configs" / filename)
        assert packaged == source
        assert validate_config(packaged)

    from scripts import run_parent_linking, run_pipeline

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lotss-association", "--h5-path", "input.h5", "--gaus-catalog", "gaussians.fits"])
    pipeline_args = run_pipeline.parse_args()
    assert pipeline_args.config == "package://data/default.yaml"
    assert Path(pipeline_args.output_dir) == tmp_path / "outputs"

    monkeypatch.setattr(sys, "argv", ["lotss-parent-link", "--h5-path", "input.h5", "--input-dir", "input"])
    parent_args = run_parent_linking.parse_args()
    assert parent_args.config == "package://data/default.yaml"
    assert Path(parent_args.output_dir) == tmp_path / "outputs" / "parent_linking"


def test_duplicate_gaussian_ids_fail_validation() -> None:
    # Use an astropy table only at test time, keeping the fixture tiny.
    astropy_table = Table(
        {
            "Gaussian_id": ["dup", "dup"],
            "RA": [1.0, 1.1],
            "DEC": [2.0, 2.1],
            "Total_flux": [3.0, 4.0],
            "Peak_flux": [2.0, 2.5],
            "Maj": [1.0 / 3600.0, 1.0 / 3600.0],
            "Min": [1.0 / 3600.0, 1.0 / 3600.0],
        }
    )
    frame, _ = normalized_gaussian_dataframe(astropy_table)
    with pytest.raises(ValueError, match="duplicate Gaussian IDs"):
        validate_normalized_gaussian_dataframe(frame)


def test_pybdsf_fits_sizes_are_normalized_from_degrees_to_arcsec(tmp_path) -> None:
    """The production FITS reader must apply the PyBDSF size-unit contract."""

    path = tmp_path / "pybdsf_gaussians.fits"
    table = Table(
        {
            "Gaussian_id": ["g0"],
            "RA": [150.0],
            "DEC": [2.0],
            "Total_flux": [10.0],
            "Peak_flux": [8.0],
            "Maj": [6.0 / 3600.0],
            "Min": [4.0 / 3600.0],
            "DC_Maj": [3.0 / 3600.0],
            "DC_Min": [2.0 / 3600.0],
            "PA": [20.0],
            "DC_PA": [20.0],
        }
    )
    for name in ("Maj", "Min", "DC_Maj", "DC_Min"):
        table[name].unit = u.deg
    table.write(path, overwrite=True)

    frame, _ = normalized_gaussian_dataframe(read_gaussian_catalog(path))
    row = frame.iloc[0]
    assert np.isclose(row["_maj"], 6.0)
    assert np.isclose(row["_min"], 4.0)
    assert np.isclose(row["_dc_maj"], 3.0)
    assert np.isclose(row["_dc_min"], 2.0)

    config = {"beam": {"major_arcsec": 6.0, "minor_arcsec": 4.0, "pa_deg": 20.0}}
    morphology = classify_gaussian_component(row, config)
    assert morphology["morphology_class"] == "resolved"
    assert morphology["observed_major_arcsec"] == pytest.approx(6.0)
    assert morphology["observed_minor_arcsec"] == pytest.approx(4.0)
    assert beam_like_score(row, config) > 0.99


def test_pybdsf_xposn_yposn_are_supported_for_pixel_matching(tmp_path) -> None:
    path = tmp_path / "pixel_only_gaussians.fits"
    Table(
        {
            "Gaussian_id": ["g0"],
            "Xposn": [3.0],
            "Yposn": [4.0],
            "Total_flux": [10.0],
            "Peak_flux": [8.0],
            "Maj": [6.0 / 3600.0],
            "Min": [4.0 / 3600.0],
        }
    ).write(path, overwrite=True)
    frame, columns = normalized_gaussian_dataframe(read_gaussian_catalog(path))
    assert columns.x == "Xposn"
    assert columns.y == "Yposn"
    assert frame.loc[0, "_maj"] == pytest.approx(6.0)
    assert frame.loc[0, "_min"] == pytest.approx(4.0)
    validate_normalized_gaussian_dataframe(frame)

    from lotss_association.matching import match_gaussians_to_cutout

    cutout = Cutout(cutout_id="c0", image=np.zeros((8, 8)), index=0, wcs=None)
    components, mode = match_gaussians_to_cutout(frame, cutout, pixel_scale_arcsec=1.0)
    assert mode == "pixel"
    assert components.loc[0, "x"] == 3.0
    assert components.loc[0, "y"] == 4.0


def test_image_mask_sums_are_not_reported_as_integrated_flux() -> None:
    from lotss_association.measurements import measure_merged_sources

    cutout = SimpleNamespace(
        cutout_id="c0",
        image=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        wcs=None,
    )
    segmentation = SimpleNamespace(
        thresholds=np.asarray([2.0, 2.5, 3.0]),
        labels_by_threshold=np.asarray(
            [
                [[1, 1], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 0], [0, 0]],
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
                "x": 0.0,
                "y": 0.0,
                "pixel_scale_arcsec": 1.0,
                "label_at_2sigma": 1,
                "label_at_2p5sigma": 1,
            }
        ]
    )
    measured = measure_merged_sources(cutout, segmentation, components, [[0]], pd.DataFrame())
    assert measured.loc[0, "pixel_sum_2sigma"] == 3.0
    assert measured.loc[0, "pixel_sum_2p5sigma"] == 1.0
    assert "total_flux_pixel_2sigma" not in measured


def test_stage2_noise_reasons_and_threshold_metadata_follow_config() -> None:
    config = validate_config(load_yaml(PROJECT_ROOT / "configs/default.yaml"))
    config["parent_linking"]["endpoint_thresholds"]["noise_peak_snr_min"] = 9.0
    config["parent_linking"]["endpoint_thresholds"]["noise_area_min_beam"] = 2.25
    cfg = parent_link_config(config)
    veto, reason = _noise_artifact_veto(
        pd.Series({"artifact_risk": False}),
        area3=2.0,
        mask_area=2.0,
        peak_snr=8.0,
        artifact=False,
        cfg=cfg,
    )
    assert veto is True
    assert set(reason.split(";")) == {"peak_snr_below_threshold", "area_3sigma_beam_below_threshold"}


def test_synthetic_stage1_fixture_separates_unrelated_components() -> None:
    """Controlled catalogue: two close pairs and one distant component."""
    from lotss_association.association import candidate_pairs

    config = validate_config(load_yaml(PROJECT_ROOT / "configs/default.yaml"))
    components = pd.DataFrame(
        [
            {"component_index": 0, "x": 10.0, "y": 10.0, "pixel_scale_arcsec": 1.0},
            {"component_index": 1, "x": 12.0, "y": 10.0, "pixel_scale_arcsec": 1.0},
            {"component_index": 2, "x": 200.0, "y": 10.0, "pixel_scale_arcsec": 1.0},
            {"component_index": 3, "x": 202.0, "y": 10.0, "pixel_scale_arcsec": 1.0},
        ]
    )
    assert candidate_pairs(components, config) == [(0, 1), (2, 3)]


@pytest.mark.parametrize("indices", [[1.5, 2.0], [np.nan, 2.0], [np.inf, 2.0], [1.2, 1.8]])
def test_fractional_or_nonfinite_component_indices_fail_fast(indices: list[float]) -> None:
    from lotss_association.association import candidate_pairs

    config = validate_config(load_yaml(PROJECT_ROOT / "configs/default.yaml"))
    components = pd.DataFrame(
        [
            {"component_index": indices[0], "x": 0.0, "y": 0.0, "pixel_scale_arcsec": 1.0},
            {"component_index": indices[1], "x": 1.0, "y": 0.0, "pixel_scale_arcsec": 1.0},
        ]
    )
    with pytest.raises(ValueError, match="component_index"):
        candidate_pairs(components, config)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [(0, 0, "self-loop"), (0, 1, "duplicate pair")],
)
def test_graph_builder_rejects_invalid_edge_invariants(left: int, right: int, message: str) -> None:
    from lotss_association.association import build_association_graph

    config = validate_config(load_yaml(PROJECT_ROOT / "configs/default.yaml"))
    components = pd.DataFrame(
        [
            {"component_index": 0, "x": 0.0, "y": 0.0, "pixel_scale_arcsec": 1.0},
            {"component_index": 1, "x": 1.0, "y": 0.0, "pixel_scale_arcsec": 1.0},
        ]
    )
    rows = [{"component_index_1": left, "component_index_2": right, "association_score": 4.0}]
    if message == "duplicate pair":
        rows.append({"component_index_1": right, "component_index_2": left, "association_score": 4.0})
    with pytest.raises(ValueError, match=message):
        build_association_graph(components, pd.DataFrame(rows), config)


def test_label_at_group_ignores_nonfinite_centroids() -> None:
    from lotss_association.parent_links import _label_at_group

    segmentation = SimpleNamespace(
        thresholds=np.asarray([3.0]),
        labels_by_threshold=np.asarray([[[1, 0], [0, 0]]]),
    )
    valid = pd.Series({"centroid_x": 0.2, "centroid_y": 0.2})
    missing = pd.Series({"centroid_x": np.nan, "centroid_y": 0.9})
    assert _label_at_group(segmentation, valid, 3.0) == 1
    assert _label_at_group(segmentation, missing, 3.0) == 0


def test_parent_conflict_resolution_assigns_each_local_group_once() -> None:
    from lotss_association.parent_links import resolve_parent_conflicts

    config = validate_config(load_yaml(PROJECT_ROOT / "configs/default.yaml"))
    edges = pd.DataFrame(
        [
            {
                "parent_acceptance_class": "accepted_high_confidence_parent",
                "local_group_id_1": "g1",
                "local_group_id_2": "g2",
                "independent_parent_evidence": "common_diffuse_envelope",
                "parent_score_final": 5.0,
                "symmetry_score": 0.8,
                "box_gap_beam_robust": 2.0,
                "parent_candidate_quality": "high",
                "rejection_reason": "",
                "conflict_resolution_status": "unresolved",
            },
            {
                "parent_acceptance_class": "accepted_high_confidence_parent",
                "local_group_id_1": "g1",
                "local_group_id_2": "g3",
                "independent_parent_evidence": "common_diffuse_envelope",
                "parent_score_final": 3.0,
                "symmetry_score": 0.7,
                "box_gap_beam_robust": 3.0,
                "parent_candidate_quality": "high",
                "rejection_reason": "",
                "conflict_resolution_status": "unresolved",
            },
        ]
    )
    resolved = resolve_parent_conflicts(edges, config)
    kept = resolved[resolved["conflict_resolution_status"] == "kept"]
    removed = resolved[resolved["conflict_resolution_status"] == "removed"]
    assert len(kept) == 1
    assert len(removed) == 1
    assert removed.iloc[0]["parent_acceptance_class"] == "rejected_parent_candidate"


def test_parent_conflict_resolution_uses_canonical_tie_break() -> None:
    from lotss_association.parent_links import resolve_parent_conflicts

    config = validate_config(load_yaml(PROJECT_ROOT / "configs/default.yaml"))
    rows = [
        {
            "parent_acceptance_class": "accepted_high_confidence_parent",
            "local_group_id_1": "g1",
            "local_group_id_2": "g3",
            "independent_parent_evidence": "common_diffuse_envelope",
            "parent_score_final": 5.0,
            "symmetry_score": 0.8,
            "box_gap_beam_robust": 2.0,
            "parent_candidate_quality": "high",
            "rejection_reason": "",
            "conflict_resolution_status": "unresolved",
        },
        {
            "parent_acceptance_class": "accepted_high_confidence_parent",
            "local_group_id_1": "g1",
            "local_group_id_2": "g2",
            "independent_parent_evidence": "common_diffuse_envelope",
            "parent_score_final": 5.0,
            "symmetry_score": 0.8,
            "box_gap_beam_robust": 2.0,
            "parent_candidate_quality": "high",
            "rejection_reason": "",
            "conflict_resolution_status": "unresolved",
        },
    ]
    resolved = resolve_parent_conflicts(pd.DataFrame(rows), config)
    kept = resolved.loc[resolved["conflict_resolution_status"] == "kept"]
    assert len(kept) == 1
    assert kept.iloc[0]["local_group_id_2"] == "g2"
