from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from lotss_association.beam import beam_area_arcsec2
from lotss_association.catalog import select_catalog_region, validate_normalized_gaussian_dataframe
from lotss_association.config import ASSOCIATION_DEFAULTS
from lotss_association.io import Cutout
from lotss_association.morphology import beam_like_score
from lotss_association.segmentation import build_snr_map, segment_snr_map
from lotss_association.utils import (
    effective_config_sha256,
    load_yaml,
    packaged_config_uri,
    resolve_effective_config,
    resolve_pixel_scale_arcsec,
    snr_map_config_kwargs,
    validate_config,
    validate_identifier,
    write_effective_config,
)
from scripts import (
    build_component_graph as graph_script,
    match_gaussians_to_cutouts as match_script,
    run_pipeline as pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_morphology_tolerances_control_beam_like_score() -> None:
    row = pd.Series({"_maj": 8.0, "_min": 4.0, "_pa": 0.0})
    base = {"beam": {"major_arcsec": 6.0, "minor_arcsec": 6.0, "pa_deg": 0.0}}
    strict = {**base, "beam_aware_classification": {"beam_like_axis_ratio_tolerance": 0.1, "beam_like_size_tolerance": 0.1}}
    loose = {**base, "beam_aware_classification": {"beam_like_axis_ratio_tolerance": 1.0, "beam_like_size_tolerance": 1.0}}
    assert beam_like_score(row, strict) < beam_like_score(row, loose)


def test_host_color_requirement_is_consumed_by_host_quality() -> None:
    from lotss_association.host_support import host_support_config, score_host_candidates

    parent = pd.Series(
        {
            "cutout_id": "c0",
            "parent_candidate_id": "p0",
            "midpoint_ra": 10.0,
            "midpoint_dec": 20.0,
            "host_search_radius_arcsec": 20.0,
            "local_group_id_1": "g1",
            "local_group_id_2": "g2",
        }
    )
    groups = {
        "g1": pd.Series({"ra": 9.999, "dec": 20.0}),
        "g2": pd.Series({"ra": 10.001, "dec": 20.0}),
    }
    hosts = pd.DataFrame(
        [{"catalogue": "catwise2020", "host_id": "h0", "host_ra": 10.0, "host_dec": 20.0, "W1": 15.0}]
    )
    permissive = host_support_config({"host_support": {"wise_color": {"require_agn_color": False}}})
    required = host_support_config({"host_support": {"wise_color": {"require_agn_color": True}}})
    free = score_host_candidates(parent, hosts, groups, 6.0, permissive)
    restricted = score_host_candidates(parent, hosts, groups, 6.0, required)
    assert free.iloc[0]["host_quality"] in {"high", "medium"}
    assert restricted.iloc[0]["host_quality"] == "low"


def test_association_thresholds_and_pair_distance_are_consumed() -> None:
    from lotss_association.association import build_association_graph, candidate_pairs

    config = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    components = pd.DataFrame(
        [
            {"component_index": 0, "x": 0.0, "y": 0.0, "pixel_scale_arcsec": 1.0},
            {"component_index": 1, "x": 10.0, "y": 0.0, "pixel_scale_arcsec": 1.0},
        ]
    )
    features = pd.DataFrame(
        [
            {
                "component_index_1": 0,
                "component_index_2": 1,
                "association_score": 3.5,
                "distance_beam": 1.0,
                "distance_arcsec": 6.0,
                "unresolved_pair_veto": False,
            }
        ]
    )
    _, default_edges = build_association_graph(components, features, config)
    assert default_edges.iloc[0]["edge_type"] == "strong"

    strict = {**config, "association": {**config["association"], "threshold_strong": 4.0}}
    _, strict_edges = build_association_graph(components, features, strict)
    assert strict_edges.iloc[0]["edge_type"] == "weak"

    near = {**config, "association": {**config["association"], "max_pair_distance_beam": 1.0}}
    assert candidate_pairs(components, config)
    assert candidate_pairs(components, near) == []


def test_matching_preselection_margin_is_configurable() -> None:
    from lotss_association.matching import match_gaussians_to_cutout

    class FixedProjectionWCS:
        def __init__(self):
            self.celestial = self

        def world_to_pixel_values(self, ra, dec):
            return np.full(len(np.atleast_1d(ra)), 5.0), np.full(len(np.atleast_1d(dec)), 5.0)

    config = {
        "matching": {"preselect_margin_arcsec": 0.0},
        "pixel_scale_arcsec": 1.0,
    }
    cutout = Cutout(
        cutout_id="c0",
        image=np.zeros((10, 10)),
        index=0,
        ra=0.0,
        dec=0.0,
        wcs=FixedProjectionWCS(),
    )
    gaussians = pd.DataFrame(
        {
            "_ra": [1.0 / 60.0],
            "_dec": [0.0],
        }
    )
    components, mode = match_gaussians_to_cutout(gaussians, cutout, pixel_scale_arcsec=1.0, config=config)
    assert mode == "sky"
    assert components.empty

    expanded = {"matching": {"preselect_margin_arcsec": 120.0}, "pixel_scale_arcsec": 1.0}
    components, mode = match_gaussians_to_cutout(gaussians, cutout, pixel_scale_arcsec=1.0, config=expanded)
    assert mode == "sky"
    assert len(components) == 1


def test_matching_combines_sky_and_pixel_only_rows() -> None:
    from lotss_association.matching import match_gaussians_to_cutout

    class FixedProjectionWCS:
        def __init__(self):
            self.celestial = self

        def world_to_pixel_values(self, ra, dec):
            return np.full(len(np.atleast_1d(ra)), 1.0), np.full(len(np.atleast_1d(dec)), 2.0)

    cutout = Cutout(
        cutout_id="c0",
        image=np.zeros((8, 8)),
        index=0,
        ra=0.0,
        dec=0.0,
        wcs=FixedProjectionWCS(),
    )
    gaussians = pd.DataFrame(
        {
            "_ra": [0.0, np.nan],
            "_dec": [0.0, np.nan],
            "_x": [np.nan, 3.0],
            "_y": [np.nan, 4.0],
        }
    )
    components, mode = match_gaussians_to_cutout(gaussians, cutout, pixel_scale_arcsec=1.0)
    assert mode == "mixed"
    assert components[["x", "y"]].to_dict("records") == [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]


def test_strict_metadata_rejects_implicit_pixel_scale_and_invalid_rms() -> None:
    with pytest.raises(ValueError, match="pixel scale"):
        resolve_pixel_scale_arcsec(None, {"runtime": {"strict_metadata": True}})
    with pytest.raises(ValueError, match="RMS"):
        build_snr_map(np.arange(64, dtype=float).reshape(8, 8), strict_metadata=True)
    with pytest.raises(ValueError, match="scalar rms"):
        build_snr_map(np.zeros((8, 8)), rms=0.0, strict_metadata=True)
    bad_map = np.ones((8, 8))
    bad_map[0, 0] = np.nan
    with pytest.raises(ValueError, match="rms map"):
        build_snr_map(np.arange(64, dtype=float).reshape(8, 8), rms=bad_map, strict_metadata=True)
    with pytest.raises(ValueError, match="beam major/minor"):
        beam_area_arcsec2(config={"runtime": {"strict_metadata": True}, "beam": {}})


def test_h5_array_reader_exposes_root_physical_metadata(tmp_path) -> None:
    from lotss_association.io import H5CutoutReader

    path = tmp_path / "cutouts.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("image", data=np.ones((1, 8, 8), dtype=np.float32))
        handle.create_dataset("rms", data=np.asarray([2.0], dtype=np.float32))
        handle.attrs["pixel_scale_arcsec"] = 1.25
        handle.attrs["beam_major_arcsec"] = 6.0
        handle.attrs["beam_minor_arcsec"] = 5.0
        handle.attrs["beam_pa_deg"] = 12.0
    cutout = H5CutoutReader(path).read(0)
    assert cutout.metadata["pixel_scale_arcsec"] == 1.25
    assert cutout.metadata["beam_major_arcsec"] == 6.0
    assert cutout.metadata["beam_minor_arcsec"] == 5.0
    assert cutout.metadata["beam_pa_deg"] == 12.0


def test_h5_array_reader_squeezes_singleton_rms_channel(tmp_path) -> None:
    from lotss_association.io import H5CutoutReader

    path = tmp_path / "channel_first.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("image", data=np.ones((1, 1, 8, 8), dtype=np.float32))
        handle.create_dataset("rms", data=np.full((1, 1, 8, 8), 2.0, dtype=np.float32))
    cutout = H5CutoutReader(path).read(0)
    assert np.asarray(cutout.rms).shape == (8, 8)


def test_h5_reader_rejects_empty_spatial_arrays_and_out_of_range_indices(tmp_path) -> None:
    from lotss_association.io import H5CutoutReader

    empty_path = tmp_path / "empty.h5"
    with h5py.File(empty_path, "w") as handle:
        handle.create_dataset("image", shape=(1, 0, 8), dtype=float)
    with pytest.raises(ValueError, match="empty spatial"):
        H5CutoutReader(empty_path)

    path = tmp_path / "one.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("image", data=np.ones((8, 8), dtype=float))
    reader = H5CutoutReader(path)
    with pytest.raises(IndexError, match="out of range"):
        reader.read(1)


def test_h5_group_reader_prefers_group_physical_metadata(tmp_path) -> None:
    from lotss_association.io import H5CutoutReader

    path = tmp_path / "groups.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("c0")
        group.create_dataset("image", data=np.ones((8, 8), dtype=float))
        group.attrs["rms"] = 2.0
        group.attrs["pixel_scale_arcsec"] = 0.8
        group.attrs["beam_major_arcsec"] = 5.0
        group.attrs["beam_minor_arcsec"] = 4.0
        handle.attrs["pixel_scale_arcsec"] = 1.5
    cutout = H5CutoutReader(path).read(0)
    assert cutout.metadata["pixel_scale_arcsec"] == 0.8
    assert cutout.metadata["beam_major_arcsec"] == 5.0
    assert cutout.rms == 2.0


def test_h5_group_reader_accepts_basename_image_override(tmp_path) -> None:
    from lotss_association.io import H5CutoutReader

    path = tmp_path / "groups_override.h5"
    with h5py.File(path, "w") as handle:
        for name in ("c0", "c1"):
            group = handle.create_group(name)
            group.create_dataset("image", data=np.ones((4, 4), dtype=float))
    reader = H5CutoutReader(path, config_h5={"image_key": "image"})
    assert len(reader) == 2
    assert reader.read(1).image.shape == (4, 4)


def test_matching_uses_pixel_coordinates_when_wcs_has_no_finite_sky_rows() -> None:
    from lotss_association.matching import match_gaussians_to_cutout

    cutout = Cutout(cutout_id="c0", image=np.zeros((8, 8)), wcs=object(), index=0)
    gaussians = pd.DataFrame({"_ra": [np.nan], "_dec": [np.nan], "_x": [2.0], "_y": [3.0]})
    components, mode = match_gaussians_to_cutout(gaussians, cutout, pixel_scale_arcsec=1.0)
    assert mode == "pixel"
    assert len(components) == 1


def test_catalog_region_handles_ra_wraparound() -> None:
    rows = pd.DataFrame({"_ra": [359.99, 0.01, 1.0], "_dec": [0.0, 0.0, 0.0]})
    selected = select_catalog_region(rows, 0.0, 0.0, radius_arcsec=100.0)
    assert selected.index.tolist() == [0, 1]


def test_output_identifiers_reject_path_traversal() -> None:
    with pytest.raises(ValueError, match="path separator"):
        validate_identifier("../private", context="cutout_id")


def test_visualization_rejects_unsafe_cutout_id() -> None:
    from lotss_association.visualize import plot_cutout_all

    with pytest.raises(ValueError, match="path separator"):
        plot_cutout_all(
            SimpleNamespace(cutout_id="../private", image=np.zeros((2, 2))),
            SimpleNamespace(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            ".",
            zoom_only=True,
        )


def test_visualization_cleanup_rejects_unsafe_cutout_id(tmp_path) -> None:
    from scripts import visualize_results

    with pytest.raises(ValueError, match="path separator"):
        visualize_results.clean_figures(tmp_path, cutout_id="../private")


def test_snr_map_configuration_is_forwarded_consistently() -> None:
    config = {
        "mean_mode": "zero",
        "rms_mode": "std",
        "smooth_before_segmentation": False,
        "gaussian_smooth_sigma_pix": 2.5,
        "runtime": {"strict_metadata": False},
    }
    assert snr_map_config_kwargs(config) == {
        "mean_mode": "zero",
        "rms_mode": "std",
        "smooth_before_segmentation": False,
        "gaussian_smooth_sigma_pix": 2.5,
        "strict_metadata": False,
    }


@pytest.mark.parametrize("entrypoint", [graph_script, match_script])
def test_public_entrypoints_forward_snr_options(monkeypatch, tmp_path, entrypoint) -> None:
    captured: dict[str, object] = {}
    config = {
        "mean_mode": "zero",
        "rms_mode": "std",
        "smooth_before_segmentation": False,
        "gaussian_smooth_sigma_pix": 2.5,
        "min_mask_area_pix": 1,
        "snr_thresholds": [5.0, 4.0, 3.0, 2.5, 2.0],
        "beam": {"major_arcsec": 6.0, "minor_arcsec": 6.0, "pa_deg": 0.0},
        "runtime": {"strict_metadata": False},
    }

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            pass

        def iter_indices(self, *_args, **_kwargs):
            return [0]

        def read(self, index):
            return SimpleNamespace(
                cutout_id="c0",
                index=index,
                image=np.ones((4, 4)),
                rms=1.0,
                mean=0.0,
                metadata={"pixel_scale_arcsec": 1.0},
                wcs=None,
            )

    def fake_snr(*_args, **kwargs):
        captured.update(kwargs)
        return np.zeros((4, 4), dtype=float), 0.0, 1.0

    monkeypatch.setattr(entrypoint, "parse_args", lambda: SimpleNamespace(
        h5_path="input.h5", gaus_catalog="gaussians.fits", config="config.yaml", output_dir=str(tmp_path),
        start_index=0, end_index=None, limit=1, debug=False,
    ))
    monkeypatch.setattr(entrypoint, "setup_logging", lambda **_kwargs: SimpleNamespace(info=lambda *_a, **_k: None))
    monkeypatch.setattr(entrypoint, "load_yaml", lambda _path: config)
    monkeypatch.setattr(
        entrypoint,
        "normalized_gaussian_dataframe",
        lambda _table: (
            pd.DataFrame(
                {
                    "_x": [1.0],
                    "_y": [1.0],
                    "_total_flux": [1.0],
                    "_peak_flux": [1.0],
                    "_maj": [6.0],
                    "_min": [6.0],
                }
            ),
            None,
        ),
    )
    monkeypatch.setattr(entrypoint, "read_gaussian_catalog", lambda _path: object())
    monkeypatch.setattr(entrypoint, "H5CutoutReader", FakeReader)
    monkeypatch.setattr(entrypoint, "build_snr_map", fake_snr)
    monkeypatch.setattr(entrypoint, "segment_snr_map", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(entrypoint, "match_gaussians_to_cutout", lambda *_args, **_kwargs: (pd.DataFrame(), "pixel"))
    if entrypoint is graph_script:
        monkeypatch.setattr(entrypoint, "build_component_graph", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(entrypoint, "write_dataframe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(entrypoint, "ensure_dir", lambda path: Path(path).mkdir(parents=True, exist_ok=True) or Path(path))

    entrypoint.main()
    expected = snr_map_config_kwargs(config)
    assert {key: captured[key] for key in expected} == expected


def test_catalog_schema_rejects_missing_scientific_fields() -> None:
    with pytest.raises(ValueError, match="required normalized column"):
        validate_normalized_gaussian_dataframe(
            pd.DataFrame({"_ra": [1.0], "_dec": [2.0], "_total_flux": [1.0]}),
        )
    with pytest.raises(ValueError, match="positive observed"):
        validate_normalized_gaussian_dataframe(
            pd.DataFrame(
                {
                    "_ra": [1.0],
                    "_dec": [2.0],
                    "_total_flux": [1.0],
                    "_peak_flux": [1.0],
                    "_maj": [0.0],
                    "_min": [1.0],
                }
            ),
        )


def test_catalog_coordinate_validation_accepts_sky_and_mixed_rows() -> None:
    all_sky = pd.DataFrame(
        {
            "_gaussian_id": ["g0", "g1"],
            "_ra": [10.0, 10.1],
            "_dec": [20.0, 20.1],
            "_x": [np.nan, np.nan],
            "_y": [np.nan, np.nan],
            "_total_flux": [2.0, 3.0],
            "_peak_flux": [1.0, 1.5],
            "_maj": [6.0, 6.0],
            "_min": [4.0, 4.0],
        }
    )
    validate_normalized_gaussian_dataframe(all_sky)

    mixed = all_sky.copy()
    mixed.loc[1, ["_ra", "_dec"]] = np.nan
    mixed.loc[1, ["_x", "_y"]] = [3.0, 4.0]
    validate_normalized_gaussian_dataframe(mixed)


def test_catalog_coordinate_validation_reports_every_unusable_row() -> None:
    frame = pd.DataFrame(
        {
            "_gaussian_id": ["g0", "g1", "g2"],
            "_ra": [10.0, 10.1, np.nan],
            "_dec": [20.0, 20.1, np.nan],
            "_x": [np.nan, np.nan, np.nan],
            "_y": [np.nan, np.nan, np.nan],
            "_total_flux": [2.0, 3.0, 4.0],
            "_peak_flux": [1.0, 1.5, 2.0],
            "_maj": [6.0, 6.0, 6.0],
            "_min": [4.0, 4.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match=r"1 Gaussian row .*g2"):
        validate_normalized_gaussian_dataframe(frame)

    from lotss_association.matching import match_gaussians_to_cutout

    cutout = Cutout(cutout_id="c0", image=np.zeros((8, 8)), index=0, wcs=object())
    with pytest.raises(ValueError, match=r"1 Gaussian row .*g2"):
        match_gaussians_to_cutout(frame, cutout, pixel_scale_arcsec=1.0)


@pytest.mark.parametrize("invalid_flux", [0.0, -1.0, np.nan, np.inf])
def test_catalog_validation_allows_unavailable_flux(invalid_flux: float) -> None:
    frame = pd.DataFrame(
        {
            "_ra": [10.0],
            "_dec": [20.0],
            "_total_flux": [invalid_flux],
            "_peak_flux": [invalid_flux],
            "_maj": [6.0],
            "_min": [4.0],
        }
    )
    validate_normalized_gaussian_dataframe(frame)


def test_matching_rejects_catalog_without_coordinates() -> None:
    from lotss_association.matching import match_gaussians_to_cutout

    cutout = Cutout(cutout_id="c0", image=np.zeros((8, 8)), index=0)
    gaussians = pd.DataFrame({"_ra": [np.nan], "_dec": [np.nan]})
    with pytest.raises(ValueError, match="1 Gaussian row"):
        match_gaussians_to_cutout(gaussians, cutout, pixel_scale_arcsec=1.0)


def test_matching_without_wcs_requires_pixels_for_every_row() -> None:
    from lotss_association.matching import match_gaussians_to_cutout

    cutout = Cutout(cutout_id="c0", image=np.zeros((8, 8)), index=0, wcs=None)
    gaussians = pd.DataFrame(
        {
            "_gaussian_id": ["sky-only", "pixel"],
            "_ra": [10.0, np.nan],
            "_dec": [20.0, np.nan],
            "_x": [np.nan, 3.0],
            "_y": [np.nan, 4.0],
        }
    )
    with pytest.raises(ValueError, match=r"1 Gaussian row .*sky-only"):
        match_gaussians_to_cutout(gaussians, cutout, pixel_scale_arcsec=1.0)


def test_association_mode_rejects_disabled_association_sections(tmp_path, monkeypatch) -> None:
    import sys

    import yaml

    from scripts import run_pipeline

    config = load_yaml(packaged_config_uri("default.yaml"))
    for section, key in (("association", "enabled"), ("local_association", "enabled")):
        disabled = deepcopy(config)
        disabled[section][key] = False
        config_path = tmp_path / f"disabled_{section}.yaml"
        config_path.write_text(yaml.safe_dump(disabled), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lotss-association",
                "--h5-path",
                "input.h5",
                "--gaus-catalog",
                "input.fits",
                "--config",
                str(config_path),
            ],
        )
        with pytest.raises(SystemExit, match="association.enabled=true"):
            run_pipeline.main()


def test_pipeline_rejects_ambiguous_resume_overwrite(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "parse_args", lambda: SimpleNamespace(resume=True, overwrite=True))
    with pytest.raises(SystemExit, match="mutually exclusive"):
        pipeline.main()


def test_status_rejects_duplicate_ids_and_partial_merge_reads_parquet_or_csv(tmp_path) -> None:
    output = tmp_path / "output"
    dirs = pipeline.ensure_output_tree(output)
    pd.DataFrame([{"cutout_id": "c0", "status": "done"}, {"cutout_id": "c0", "status": "failed"}]).to_csv(
        dirs["logs"] / "status.csv", index=False
    )
    with pytest.raises(ValueError, match="duplicate cutout_id"):
        pipeline.done_cutouts(dirs["logs"] / "status.csv")

    (dirs["logs"] / "status.csv").unlink()
    pd.DataFrame([{"cutout_id": "c0", "status": "done"}]).to_csv(dirs["logs"] / "status.csv", index=False)
    partials = dirs["partials"]
    # A parquet-only completed partial is valid when a parquet engine is
    # available; the test uses CSV fallback so it remains portable.
    for stem, frame in [
        ("merged_sources", pd.DataFrame({"merged_source_id": ["m0"]})),
        ("edges", pd.DataFrame(columns=pipeline.EDGE_COLUMNS)),
        ("components", pd.DataFrame(columns=pipeline.ASSOCIATION_COMPONENT_COLUMNS)),
        ("radio_association_groups", pd.DataFrame(columns=pipeline.ASSOCIATION_GROUP_COLUMNS)),
        ("radio_association_edges", pd.DataFrame(columns=pipeline.ASSOCIATION_EDGE_COLUMNS)),
        ("radio_association_components", pd.DataFrame(columns=pipeline.ASSOCIATION_COMPONENT_COLUMNS)),
    ]:
        frame.to_csv(partials / f"c0_{stem}.csv", index=False)
    pipeline.combine_partials(output)
    merged = pd.read_csv(dirs["catalogs"] / "lotss_association_merged_sources.csv")
    assert merged["merged_source_id"].tolist() == ["m0"]


def test_overwrite_removes_stale_catalog_representations(tmp_path) -> None:
    output = tmp_path / "output"
    dirs = pipeline.ensure_output_tree(output)
    for name in [
        "lotss_association_edges.csv",
        "lotss_association_edges.parquet",
        "radio_association_groups.csv",
        "local_sanity_diagnostics.csv",
        "local_needs_visual_check.csv",
    ]:
        (dirs["catalogs"] / name).write_text("stale", encoding="utf-8")
    pipeline.reset_overwrite_outputs(dirs)
    assert not any(
        (dirs["catalogs"] / name).exists()
        for name in [
            "lotss_association_edges.csv",
            "lotss_association_edges.parquet",
            "radio_association_groups.csv",
            "local_sanity_diagnostics.csv",
            "local_needs_visual_check.csv",
        ]
    )


def test_append_dataframe_writes_local_sanity_tables(tmp_path) -> None:
    path = tmp_path / "local_sanity_diagnostics.csv"
    pipeline.append_dataframe(path, pd.DataFrame([{"cutout_id": "c0", "split_applied": True}]))
    pipeline.append_dataframe(path, pd.DataFrame([{"cutout_id": "c1", "split_applied": False}]))

    stored = pd.read_csv(path)
    assert stored["cutout_id"].tolist() == ["c0", "c1"]


def test_rewriting_partials_removes_stale_parquet_siblings(tmp_path) -> None:
    partials = tmp_path / "partials"
    partials.mkdir()
    stale = partials / "c0_edges.parquet"
    stale.write_text("stale", encoding="utf-8")
    pipeline.write_cutout_partials(
        partials,
        "c0",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert not stale.exists()


def test_segmentation_requires_explicit_min_area() -> None:
    with pytest.raises(ValueError, match="min_mask_area_pix"):
        segment_snr_map(np.zeros((4, 4)), [3.0])


def test_segmentation_rejects_invalid_thresholds_and_metadata_shapes() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        segment_snr_map(np.zeros((4, 4)), [])
    with pytest.raises(ValueError, match="rms map shape"):
        build_snr_map(np.zeros((4, 4)), rms=np.ones((3, 3)))
    with pytest.raises(ValueError, match="mean map shape"):
        build_snr_map(np.zeros((4, 4)), mean=np.ones((3, 3)))


def test_effective_config_expands_core_defaults_without_private_trees() -> None:
    effective = resolve_effective_config(
        {"association": {"threshold_strong": 4.0}, "parent_linking": {"enabled": True}}
    )
    assert effective["association"]["threshold_strong"] == 4.0
    assert "weights_association" in effective
    assert "beam_aware_classification" in effective
    assert effective["parent_linking"]["enabled"] is True


def test_pair_distance_default_is_consistent_across_python_and_packaged_configs() -> None:
    expected = 120.0
    assert ASSOCIATION_DEFAULTS["max_pair_distance_arcsec"] == expected
    assert resolve_effective_config({})["association"]["max_pair_distance_arcsec"] == expected
    for filename in ("default.yaml",):
        source = load_yaml(PROJECT_ROOT / "configs" / filename)
        packaged = load_yaml(packaged_config_uri(filename))
        assert source["association"]["max_pair_distance_arcsec"] == expected
        assert packaged["association"]["max_pair_distance_arcsec"] == expected


def test_release_config_includes_formal_stage2_sections() -> None:
    config = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    assert "parent_seed_selection" in config
    assert "parent_linking" in config
    assert "host_support" in config


def test_config_rejects_mandatory_host_when_host_support_is_disabled() -> None:
    config = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    config["host_support"].update(
        {"enabled": False, "require_host_for_parent_link": True}
    )

    with pytest.raises(ValueError, match="require_host_for_parent_link"):
        validate_config(config)


def test_config_schema_rejects_unknown_keys_and_non_descending_thresholds() -> None:
    config = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    unknown = dict(config)
    unknown["snr_thesholds"] = [5.0]
    with pytest.raises(ValueError, match="Unknown configuration key"):
        from lotss_association.utils import validate_config

        validate_config(unknown)
    bad_order = dict(config)
    bad_order["snr_thresholds"] = [2.0, 3.0]
    with pytest.raises(ValueError, match="strictly descending"):
        from lotss_association.utils import validate_config

        validate_config(bad_order)


def test_pixel_scale_requires_metadata_or_explicit_config() -> None:
    with pytest.raises(ValueError, match="pixel_scale_arcsec"):
        resolve_pixel_scale_arcsec(None, {"runtime": {"strict_metadata": False}})


def test_runtime_label_cache_does_not_mutate_scientific_config() -> None:
    from lotss_association.association import _label_count_cache

    labels = np.zeros((1, 4, 4), dtype=np.int32)
    config = {"beam": {"major_arcsec": 6.0, "minor_arcsec": 6.0}}
    before = repr(config)
    _label_count_cache(labels, np.asarray([2.0]), config)
    assert repr(config) == before


def test_effective_config_provenance_is_stable_and_complete(tmp_path) -> None:
    config = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    expected_hash = effective_config_sha256(resolve_effective_config(config))
    output = tmp_path / "effective.yaml"
    write_effective_config(output, config, metadata={"config_file": "default.yaml"})
    persisted = load_yaml(output)
    metadata = persisted["runtime"]["effective_metadata"]
    assert metadata["config_sha256"] == expected_hash
    assert metadata["software_version"]
    assert metadata["git_commit"]


def test_git_commit_uses_build_environment_revision(monkeypatch) -> None:
    from lotss_association import utils

    monkeypatch.setenv("LOTSS_ASSOCIATION_SOURCE_COMMIT", "0123456789abcdef0123456789abcdef01234567")
    assert utils._git_commit() == "0123456789abcdef0123456789abcdef01234567"


def test_combine_partials_rejects_cross_cutout_gaussian_ownership(tmp_path) -> None:
    output = tmp_path / "output"
    dirs = pipeline.ensure_output_tree(output)
    pd.DataFrame(
        [
            {"cutout_id": "c0", "status": "done"},
            {"cutout_id": "c1", "status": "done"},
        ]
    ).to_csv(dirs["logs"] / "status.csv", index=False)
    for cutout_id in ("c0", "c1"):
        pd.DataFrame(
            [{"cutout_id": cutout_id, "_gaussian_id": "global-1", "component_index": 0}]
        ).to_csv(dirs["partials"] / f"{cutout_id}_components.csv", index=False)
        empty_columns = {
            "merged_sources": pipeline.MERGED_COLUMNS,
            "edges": pipeline.EDGE_COLUMNS,
            "radio_association_groups": pipeline.ASSOCIATION_GROUP_COLUMNS,
            "radio_association_edges": pipeline.ASSOCIATION_EDGE_COLUMNS,
            "radio_association_components": pipeline.ASSOCIATION_COMPONENT_COLUMNS,
        }
        for stem, columns in empty_columns.items():
            pd.DataFrame(columns=columns).to_csv(dirs["partials"] / f"{cutout_id}_{stem}.csv", index=False)
    with pytest.raises(ValueError, match="owned by multiple cutouts"):
        pipeline.combine_partials(output)
