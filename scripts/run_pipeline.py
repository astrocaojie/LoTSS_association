#!/usr/bin/env python
"""Run the rule-based LoTSS Association pipeline."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lotss_association.association import (
    EDGE_COLUMNS as ASSOCIATION_EDGE_COLUMNS,
    GROUP_COLUMNS as ASSOCIATION_GROUP_COLUMNS,
    run_component_association,
)
from lotss_association.catalog import (
    normalized_gaussian_dataframe,
    read_gaussian_catalog,
    validate_normalized_gaussian_dataframe,
)
from lotss_association.graph_merge import build_component_graph
from lotss_association.io import H5CutoutReader
from lotss_association.matching import match_gaussians_to_cutout
from lotss_association.measurements import measure_merged_sources
from lotss_association.segmentation import build_snr_map, save_segmentation, segment_snr_map, segmentation_diagnostics
from lotss_association.utils import (
    ensure_dir,
    failed_status_message,
    load_yaml,
    packaged_config_uri,
    segmentation_config_kwargs,
    setup_logging,
    snr_map_config_kwargs,
    strict_metadata_enabled,
    validate_config,
    validate_cutout_physical_metadata,
    validate_identifier,
    write_dataframe,
    write_effective_config,
)
from lotss_association.visualize import plot_cutout_all

EDGE_COLUMNS = [
    "cutout_id",
    "gaussian_id_1",
    "gaussian_id_2",
    "distance_arcsec",
    "same_pybdsf_island",
    "connected_at_3sigma",
    "connected_at_2p5sigma",
    "connected_at_2sigma",
    "bridge_snr_mean",
    "merge_score",
    "merge_decision",
    "positive_evidence",
    "negative_evidence",
]

# Diagnostic-only QA gates for the per-cutout edge/association summaries.
EDGE_QA_ONLY_2SIGMA_FRACTION = 0.5
EDGE_QA_MAX_GROUP_SIZE = 20

MERGED_COLUMNS = [
    "cutout_id",
    "merged_source_id",
    "n_components",
    "gaussian_ids",
    "island_ids",
    "ra",
    "dec",
    "centroid_x",
    "centroid_y",
    "total_flux_gaussian",
    "pixel_sum_2sigma",
    "pixel_sum_2p5sigma",
    "peak_flux",
    "LAS_arcsec",
    "PA",
    "merge_confidence",
    "flags",
    "debug_info",
]

ASSOCIATION_COMPONENT_COLUMNS = [
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
    "_dc_maj",
    "_dc_min",
    "_dc_pa",
    "x",
    "y",
    "pixel_scale_arcsec",
    "association_group_id",
    "original_association_group_id",
    "association_group_index",
    "association_group_size",
    "association_quality",
    "association_type",
    "morphology_class",
    "resolved_probability",
    "resolved_significance",
    "beam_like_score",
    "classification_reason",
    "observed_major_arcsec",
    "observed_minor_arcsec",
    "observed_pa_pixel_deg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True, help="H5 file containing LoTSS cutouts")
    parser.add_argument(
        "--gaus-catalog",
        required=True,
        help="PyBDSF Gaussian FITS catalog",
    )
    parser.add_argument("--config", default=packaged_config_uri("default.yaml"))
    parser.add_argument("--output-dir", default=str(Path.cwd() / "outputs"))
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Deprecated compatibility option; only 1 worker is supported",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip cutouts marked done in status.csv")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess selected cutouts even if outputs exist")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--association-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write beam-aware radio association catalogs and use them for figures.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_status(path: Path) -> pd.DataFrame:
    if path.exists():
        status = pd.read_csv(path)
        required = {"cutout_id", "status"}
        missing = sorted(required - set(status.columns))
        if missing:
            raise ValueError(f"status file is missing required column(s): {', '.join(missing)}: {path}")
        if status["cutout_id"].isna().any() or status["cutout_id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"status file contains an empty cutout_id: {path}")
        status["cutout_id"] = status["cutout_id"].astype(str).str.strip()
        for value in status["cutout_id"]:
            validate_identifier(value, context="status cutout_id")
        if status["cutout_id"].duplicated().any():
            duplicates = sorted(set(status.loc[status["cutout_id"].duplicated(keep=False), "cutout_id"].astype(str)))
            raise ValueError(f"status file contains duplicate cutout_id values: {', '.join(duplicates[:10])}")
        if status["status"].isna().any() or status["status"].astype(str).str.strip().eq("").any():
            raise ValueError(f"status file contains an empty status value: {path}")
        return status
    return pd.DataFrame(
        columns=[
            "cutout_id",
            "status",
            "time_start",
            "time_end",
            "n_gaussians",
            "n_merged_sources",
            "n_association_groups",
            "failure_reason",
        ]
    )


def update_status(path: Path, record: dict) -> None:
    status = load_status(path)
    status = status[status["cutout_id"].astype(str) != str(record["cutout_id"])]
    status = pd.concat([status, pd.DataFrame([record])], ignore_index=True)
    status.to_csv(path, index=False)


def done_cutouts(path: Path) -> set[str]:
    status = load_status(path)
    if status.empty or "status" not in status:
        return set()
    if "cutout_id" not in status:
        raise ValueError(f"status file is missing required cutout_id column: {path}")
    done = status[status["status"].astype(str).str.strip().str.lower() == "done"]
    return set(done["cutout_id"].astype(str).tolist())


def ensure_output_tree(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "segmentation": ensure_dir(output_dir / "segmentation"),
        "graphs": ensure_dir(output_dir / "graphs"),
        "catalogs": ensure_dir(output_dir / "catalogs"),
        "figures": ensure_dir(output_dir / "figures"),
        "logs": ensure_dir(output_dir / "logs"),
        "partials": ensure_dir(output_dir / "catalogs" / "partials"),
    }
    return dirs


def reset_overwrite_outputs(dirs: dict[str, Path]) -> None:
    """Clear append-style outputs so --overwrite starts a clean production run."""

    for path in dirs["catalogs"].glob("lotss_association_*.csv"):
        path.unlink()
    for path in dirs["catalogs"].glob("lotss_association_*.parquet"):
        path.unlink()
    for path in dirs["catalogs"].glob("radio_association_*.csv"):
        path.unlink()
    for path in dirs["catalogs"].glob("radio_association_*.parquet"):
        path.unlink()
    for name in [
        "matching_diagnostics.csv",
        "segmentation_diagnostics.csv",
        "edge_diagnostics.csv",
        "association_diagnostics.csv",
        "local_sanity_diagnostics.csv",
        "local_needs_visual_check.csv",
    ]:
        path = dirs["catalogs"] / name
        if path.exists():
            path.unlink()
    for path in list(dirs["partials"].glob("*.csv")) + list(dirs["partials"].glob("*.parquet")):
        path.unlink()
    for path in dirs["figures"].glob("*_association.png"):
        path.unlink()
    for folder_name in ["overview", "zoom"]:
        folder = dirs["figures"] / folder_name
        if folder.exists():
            for path in folder.glob("*.png"):
                path.unlink()
    for path in dirs["segmentation"].glob("*_seg.npz"):
        path.unlink()
    status_path = dirs["logs"] / "status.csv"
    if status_path.exists():
        status_path.unlink()


def append_diagnostics(path: Path, records: list[dict]) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def append_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Append a non-empty diagnostic table while preserving its schema."""

    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = pd.Series(dtype=object)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out


def write_cutout_partials(
    partial_dir: Path,
    cutout_id: str,
    merged: pd.DataFrame,
    edges: pd.DataFrame,
    components: pd.DataFrame,
    association_groups: pd.DataFrame | None = None,
    association_edges: pd.DataFrame | None = None,
    association_components: pd.DataFrame | None = None,
) -> None:
    cutout_id = validate_identifier(cutout_id, context="cutout_id")
    merged = _with_columns(merged, MERGED_COLUMNS)
    edges = _with_columns(edges, EDGE_COLUMNS)
    association_groups = _with_columns(association_groups if association_groups is not None else pd.DataFrame(), ASSOCIATION_GROUP_COLUMNS)
    association_edges = _with_columns(association_edges if association_edges is not None else pd.DataFrame(), ASSOCIATION_EDGE_COLUMNS)
    association_components = _with_columns(
        association_components if association_components is not None else pd.DataFrame(),
        ASSOCIATION_COMPONENT_COLUMNS,
    )
    # CSV is the canonical partial representation. Remove an older parquet
    # sibling before replacing a cutout so a resumed run cannot merge stale
    # rows from a previous environment that happened to have parquet support.
    for stem in (
        "merged_sources",
        "edges",
        "components",
        "radio_association_groups",
        "radio_association_edges",
        "radio_association_components",
    ):
        stale_parquet = partial_dir / f"{cutout_id}_{stem}.parquet"
        if stale_parquet.exists():
            stale_parquet.unlink()
    merged.to_csv(partial_dir / f"{cutout_id}_merged_sources.csv", index=False)
    edges.to_csv(partial_dir / f"{cutout_id}_edges.csv", index=False)
    components.to_csv(partial_dir / f"{cutout_id}_components.csv", index=False)
    association_groups.to_csv(partial_dir / f"{cutout_id}_radio_association_groups.csv", index=False)
    association_edges.to_csv(partial_dir / f"{cutout_id}_radio_association_edges.csv", index=False)
    association_components.to_csv(partial_dir / f"{cutout_id}_radio_association_components.csv", index=False)


def matching_diagnostic_record(cutout, components: pd.DataFrame, match_mode: str, n_outside: int = 0) -> dict:
    has_ra_dec = cutout.ra is not None and cutout.dec is not None
    warning = ""
    if len(components) == 0:
        warning = "no_gaussians_matched"
    return {
        "cutout_id": cutout.cutout_id,
        "n_gaussians_matched": int(len(components)),
        "image_shape": "x".join(map(str, cutout.image.shape)),
        "has_wcs": bool(cutout.wcs is not None),
        "has_ra_dec": bool(has_ra_dec),
        "match_mode": match_mode,
        "min_x": float(components["x"].min()) if len(components) else float("nan"),
        "max_x": float(components["x"].max()) if len(components) else float("nan"),
        "min_y": float(components["y"].min()) if len(components) else float("nan"),
        "max_y": float(components["y"].max()) if len(components) else float("nan"),
        "n_outside_after_projection": int(n_outside),
        "warning": warning,
    }


def edge_diagnostic_record(cutout_id: str, components: pd.DataFrame, edges: pd.DataFrame) -> dict:
    n_nodes = int(len(components))
    if edges.empty:
        return {
            "cutout_id": cutout_id,
            "n_nodes": n_nodes,
            "n_candidate_pairs": 0,
            "n_edges_merged": 0,
            "merge_score_min": float("nan"),
            "merge_score_median": float("nan"),
            "merge_score_max": float("nan"),
            "connected_at_3sigma_count": 0,
            "connected_at_2p5sigma_count": 0,
            "connected_at_2sigma_count": 0,
            "only_2sigma_connected_count": 0,
            "median_distance_arcsec": float("nan"),
            "max_distance_arcsec": float("nan"),
            "max_merged_component_size": 1 if n_nodes else 0,
            "warning": "no_candidate_pairs",
        }
    decision_col = "merge_decision" if "merge_decision" in edges else "association_decision"
    score_col = "merge_score" if "merge_score" in edges else "association_score"
    distance_col = "distance_arcsec" if "distance_arcsec" in edges else "pair_separation_arcsec"
    merged_edges = edges[edges[decision_col].astype(bool)]
    only_2 = edges[
        edges["connected_at_2sigma"].astype(bool)
        & ~edges["connected_at_2p5sigma"].astype(bool)
        & ~edges["connected_at_3sigma"].astype(bool)
    ]
    max_component_size = 0
    if "merged_component_group" in components and len(components):
        max_component_size = int(components.groupby("merged_component_group").size().max())
    warning = ""
    if len(merged_edges) and len(only_2) / max(len(edges), 1) > EDGE_QA_ONLY_2SIGMA_FRACTION:
        warning = "many_only_2sigma_pairs"
    if max_component_size > EDGE_QA_MAX_GROUP_SIZE:
        warning = f"{warning};large_connected_component".strip(";")
    return {
        "cutout_id": cutout_id,
        "n_nodes": n_nodes,
        "n_candidate_pairs": int(len(edges)),
        "n_edges_merged": int(len(merged_edges)),
        "merge_score_min": float(pd.to_numeric(edges[score_col], errors="coerce").min()),
        "merge_score_median": float(pd.to_numeric(edges[score_col], errors="coerce").median()),
        "merge_score_max": float(pd.to_numeric(edges[score_col], errors="coerce").max()),
        "connected_at_3sigma_count": int(edges["connected_at_3sigma"].astype(bool).sum()),
        "connected_at_2p5sigma_count": int(edges["connected_at_2p5sigma"].astype(bool).sum()),
        "connected_at_2sigma_count": int(edges["connected_at_2sigma"].astype(bool).sum()),
        "only_2sigma_connected_count": int(len(only_2)),
        "median_distance_arcsec": float(pd.to_numeric(edges[distance_col], errors="coerce").median()),
        "max_distance_arcsec": float(pd.to_numeric(edges[distance_col], errors="coerce").max()),
        "max_merged_component_size": max_component_size,
        "warning": warning,
    }


def association_diagnostic_record(
    cutout_id: str,
    components: pd.DataFrame,
    groups: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict:
    n_nodes = int(len(components))
    if edges.empty:
        return {
            "cutout_id": cutout_id,
            "n_nodes": n_nodes,
            "n_candidate_pairs": 0,
            "n_association_groups": int(len(groups)),
            "n_strong_edges": 0,
            "n_weak_edges": 0,
            "n_rejected_edges": 0,
            "n_decision_edges": 0,
            "n_only_2sigma_edges": 0,
            "n_unresolved": 0,
            "n_marginally_resolved": 0,
            "n_resolved": 0,
            "n_artifact_like": 0,
            "n_unresolved_pair_veto": 0,
            "association_score_min": float("nan"),
            "association_score_median": float("nan"),
            "association_score_max": float("nan"),
            "max_group_size": int(groups["n_gaussians"].max()) if not groups.empty and "n_gaussians" in groups else (1 if n_nodes else 0),
            "warning": "no_candidate_pairs",
        }
    only_2 = edges[
        edges["only_2sigma_connected"].astype(bool)
        & edges["association_decision"].astype(bool)
    ]
    morph = components.get("morphology_class", pd.Series(dtype=str)).astype(str) if not components.empty else pd.Series(dtype=str)
    max_group_size = int(groups["n_gaussians"].max()) if not groups.empty and "n_gaussians" in groups else 0
    warning = ""
    if len(only_2) > max(1, int(EDGE_QA_ONLY_2SIGMA_FRACTION * max(len(edges[edges["association_decision"].astype(bool)]), 1))):
        warning = "many_only_2sigma_associations"
    if max_group_size > 20:
        warning = f"{warning};large_association_group".strip(";")
    return {
        "cutout_id": cutout_id,
        "n_nodes": n_nodes,
        "n_candidate_pairs": int(len(edges)),
        "n_association_groups": int(len(groups)),
        "n_strong_edges": int((edges["edge_type"].astype(str) == "strong").sum()),
        "n_weak_edges": int((edges["edge_type"].astype(str) == "weak").sum()),
        "n_rejected_edges": int((edges["edge_type"].astype(str) == "rejected").sum()),
        "n_decision_edges": int(edges["association_decision"].astype(bool).sum()),
        "n_only_2sigma_edges": int(len(only_2)),
        "n_unresolved": int((morph == "unresolved").sum()),
        "n_marginally_resolved": int((morph == "marginally_resolved").sum()),
        "n_resolved": int((morph == "resolved").sum()),
        "n_artifact_like": int((morph == "artifact_like").sum()),
        "n_unresolved_pair_veto": int(edges.get("unresolved_pair_veto", pd.Series(dtype=bool)).astype(bool).sum()),
        "association_score_min": float(pd.to_numeric(edges["association_score"], errors="coerce").min()),
        "association_score_median": float(pd.to_numeric(edges["association_score"], errors="coerce").median()),
        "association_score_max": float(pd.to_numeric(edges["association_score"], errors="coerce").max()),
        "max_group_size": max_group_size,
        "warning": warning,
    }


def combine_partials(output_dir: Path, allowed_cutouts: set[str] | None = None) -> None:
    partial_dir = output_dir / "catalogs" / "partials"
    catalogs_dir = output_dir / "catalogs"

    status_path = output_dir / "logs" / "status.csv"
    if allowed_cutouts is None and status_path.exists():
        # A status file is authoritative: failed/stale cutouts must never be
        # republished from an older partial left in the output directory.
        allowed_cutouts = done_cutouts(status_path)
    elif allowed_cutouts is not None:
        allowed_cutouts = {str(cutout_id) for cutout_id in allowed_cutouts}

    partial_stems = (
        "merged_sources",
        "edges",
        "components",
        "radio_association_groups",
        "radio_association_edges",
        "radio_association_components",
    )

    def partial_paths(stem: str) -> dict[str, dict[str, Path]]:
        fields: dict[str, dict[str, Path]] = {}
        for extension in ("csv", "parquet"):
            marker = f"_{stem}.{extension}"
            for path in partial_dir.glob(f"*{marker}"):
                field = path.name[: -len(marker)]
                fields.setdefault(field, {})[extension] = path
        return fields

    if allowed_cutouts is not None:
        missing = {
            stem: sorted(set(map(str, allowed_cutouts)) - set(partial_paths(stem)))
            for stem in partial_stems
        }
        missing = {stem: fields for stem, fields in missing.items() if fields}
        if missing:
            details = "; ".join(f"{stem}: {', '.join(fields[:10])}" for stem, fields in missing.items())
            raise RuntimeError(f"Missing partial output(s) for completed cutout(s): {details}")

    def combine(stem: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for field, candidates in sorted(partial_paths(stem).items()):
            if allowed_cutouts is not None and field not in allowed_cutouts:
                continue
            frame = None
            empty_frame = None
            errors: list[str] = []
            # Prefer parquet, but fall back to CSV when parquet dependencies or
            # an older partial prevent reading it.  Exactly one frame per field
            # is included even when both formats exist.
            for extension in ("parquet", "csv"):
                path = candidates.get(extension)
                if path is None:
                    continue
                try:
                    candidate_frame = pd.read_parquet(path) if extension == "parquet" else pd.read_csv(path)
                    if candidate_frame.empty and len(candidates) > 1:
                        empty_frame = candidate_frame
                        continue
                    frame = candidate_frame
                    break
                except pd.errors.EmptyDataError as exc:
                    errors.append(f"{path}: empty table ({exc})")
                except Exception as exc:
                    errors.append(f"{path}: {type(exc).__name__}: {exc}")
            if frame is None:
                frame = empty_frame
            if frame is None:
                if allowed_cutouts is not None and field in allowed_cutouts:
                    raise RuntimeError(f"Could not read completed {stem} partial for {field}: {'; '.join(errors)}")
                continue
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    merged = combine("merged_sources")
    edges = combine("edges")
    components = combine("components")
    association_groups = combine("radio_association_groups")
    association_edges = combine("radio_association_edges")
    association_components = combine("radio_association_components")

    def validate_gaussian_ownership(frame: pd.DataFrame, label: str) -> None:
        """Reject duplicate global Gaussian ownership in the published tables."""

        if frame.empty or "_gaussian_id" not in frame.columns or "cutout_id" not in frame.columns:
            return
        ownership = frame[["_gaussian_id", "cutout_id"]].copy()
        ownership = ownership.dropna(subset=["_gaussian_id", "cutout_id"])
        if ownership.empty:
            return
        ownership["_gaussian_id"] = ownership["_gaussian_id"].astype(str)
        ownership["cutout_id"] = ownership["cutout_id"].astype(str)
        duplicate_rows = ownership.duplicated(subset=["_gaussian_id", "cutout_id"], keep=False)
        if duplicate_rows.any():
            duplicates = ownership.loc[duplicate_rows, ["_gaussian_id", "cutout_id"]].drop_duplicates()
            preview = "; ".join(
                f"{row['_gaussian_id']}@{row['cutout_id']}"
                for _, row in duplicates.head(10).iterrows()
            )
            raise ValueError(f"{label} contains duplicate Gaussian ownership rows: {preview}")
        conflicts = ownership.groupby("_gaussian_id")["cutout_id"].unique()
        conflicts = conflicts[conflicts.map(len) > 1]
        if not conflicts.empty:
            preview = "; ".join(
                f"{gaussian_id}: {', '.join(sorted(map(str, cutouts)))}"
                for gaussian_id, cutouts in conflicts.head(10).items()
            )
            raise ValueError(f"{label} contains Gaussian IDs owned by multiple cutouts: {preview}")

    validate_gaussian_ownership(components, "component partials")
    validate_gaussian_ownership(association_components, "association component partials")

    merged = _with_columns(merged, MERGED_COLUMNS)
    edges = _with_columns(edges, EDGE_COLUMNS)
    association_groups = _with_columns(association_groups, ASSOCIATION_GROUP_COLUMNS)
    association_edges = _with_columns(association_edges, ASSOCIATION_EDGE_COLUMNS)
    association_components = _with_columns(association_components, ASSOCIATION_COMPONENT_COLUMNS)

    merged.to_csv(catalogs_dir / "lotss_association_merged_sources.csv", index=False)
    write_dataframe(merged, catalogs_dir / "lotss_association_merged_sources.parquet")
    write_dataframe(edges, catalogs_dir / "lotss_association_edges.parquet")
    write_dataframe(components, catalogs_dir / "lotss_association_components.parquet")
    edges.to_csv(catalogs_dir / "lotss_association_edges.csv", index=False)
    components.to_csv(catalogs_dir / "lotss_association_components.csv", index=False)
    association_groups.to_csv(catalogs_dir / "radio_association_groups.csv", index=False)
    write_dataframe(association_groups, catalogs_dir / "radio_association_groups.parquet")
    write_dataframe(association_edges, catalogs_dir / "radio_association_edges.parquet")
    write_dataframe(association_components, catalogs_dir / "radio_association_components.parquet")
    association_edges.to_csv(catalogs_dir / "radio_association_edges.csv", index=False)
    association_components.to_csv(catalogs_dir / "radio_association_components.csv", index=False)


def process_cutout(
    cutout,
    gaussians: pd.DataFrame,
    config: dict,
    dirs: dict[str, Path],
    make_figures: bool,
    association_mode: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Process each cutout through segmentation, matching, association, and output.
    validate_identifier(cutout.cutout_id, context="cutout_id")
    validate_cutout_physical_metadata(cutout, config)
    snr, mean_value, rms_value = build_snr_map(
        cutout.image,
        rms=cutout.rms,
        mean=cutout.mean,
        **snr_map_config_kwargs(config),
    )
    segmentation = segment_snr_map(snr, **segmentation_config_kwargs(config))
    save_segmentation(dirs["segmentation"] / f"{cutout.cutout_id}_seg.npz", segmentation)
    append_diagnostics(
        dirs["catalogs"] / "segmentation_diagnostics.csv",
        segmentation_diagnostics(cutout.cutout_id, segmentation),
    )

    components, matching_mode = match_gaussians_to_cutout(
        gaussians,
        cutout,
        segmentation,
        pixel_scale_arcsec=(cutout.metadata or {}).get("pixel_scale_arcsec"),
        strict_metadata=strict_metadata_enabled(config),
        config=config,
    )
    components["matching_mode"] = matching_mode
    components["snr_mean_used"] = str(mean_value if not hasattr(mean_value, "shape") else "map")
    components["snr_rms_used"] = str(rms_value if not hasattr(rms_value, "shape") else "map")
    append_diagnostics(
        dirs["catalogs"] / "matching_diagnostics.csv",
        [
            matching_diagnostic_record(
                cutout,
                components,
                matching_mode,
                int(components.attrs.get("n_outside_after_projection", 0)),
            )
        ],
    )

    if len(components) == 0:
        edges = pd.DataFrame(columns=EDGE_COLUMNS)
        merged = pd.DataFrame(columns=MERGED_COLUMNS)
        association_groups = pd.DataFrame(columns=ASSOCIATION_GROUP_COLUMNS)
        association_edges = pd.DataFrame(columns=ASSOCIATION_EDGE_COLUMNS)
        association_components = pd.DataFrame(columns=ASSOCIATION_COMPONENT_COLUMNS)
        local_sanity_diagnostics = pd.DataFrame()
        local_needs_visual_check = pd.DataFrame()
    else:
        if association_mode:
            # The formal release path is Stage 1 association; main() has already
            # rejected --association-mode combined with disabled association
            # sections.  The historical graph-merge path remains available
            # through build_component_graph and --no-association-mode only.
            association_result = run_component_association(cutout, segmentation, components, config)
            association_groups = association_result.groups
            association_edges = association_result.edges
            association_components = association_result.components
            local_sanity_diagnostics = association_result.local_sanity_diagnostics
            local_needs_visual_check = association_result.local_needs_visual_check
            components = association_components
            edges = association_edges
            merged = measure_merged_sources(
                cutout,
                segmentation,
                association_result.components,
                association_result.clusters,
                association_result.edges,
                config=config,
            )
        else:
            graph_result = build_component_graph(components, segmentation, config)
            edges = graph_result.edges
            components = graph_result.components
            merged = measure_merged_sources(
                cutout,
                segmentation,
                graph_result.components,
                graph_result.clusters,
                graph_result.edges,
                config=config,
            )
            association_groups = pd.DataFrame(columns=ASSOCIATION_GROUP_COLUMNS)
            association_edges = pd.DataFrame(columns=ASSOCIATION_EDGE_COLUMNS)
            association_components = pd.DataFrame(columns=ASSOCIATION_COMPONENT_COLUMNS)
            local_sanity_diagnostics = pd.DataFrame()
            local_needs_visual_check = pd.DataFrame()
    append_diagnostics(
        dirs["catalogs"] / "edge_diagnostics.csv",
        [edge_diagnostic_record(cutout.cutout_id, components, edges)],
    )
    if association_mode:
        append_diagnostics(
            dirs["catalogs"] / "association_diagnostics.csv",
            [association_diagnostic_record(cutout.cutout_id, components, association_groups, association_edges)],
        )
        append_dataframe(dirs["catalogs"] / "local_sanity_diagnostics.csv", local_sanity_diagnostics)
        append_dataframe(dirs["catalogs"] / "local_needs_visual_check.csv", local_needs_visual_check)

    write_cutout_partials(
        dirs["partials"],
        cutout.cutout_id,
        merged,
        edges,
        components,
        association_groups=association_groups,
        association_edges=association_edges,
        association_components=association_components,
    )

    if make_figures:
        figure_edges = association_edges if association_mode else edges
        figure_groups = association_groups if association_mode else merged
        plot_cutout_all(
            cutout,
            segmentation,
            components,
            figure_edges,
            figure_groups,
            dirs["figures"],
            config,
        )
    return merged, edges, components, association_groups


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.n_workers < 1:
        raise SystemExit("--n-workers must be >= 1")
    if args.n_workers != 1:
        raise SystemExit("--n-workers values greater than 1 are not supported by this sequential entry point")
    output_dir = Path(args.output_dir)
    dirs = ensure_output_tree(output_dir)
    logger = setup_logging(debug=args.debug, log_path=dirs["logs"] / "run_pipeline.log")
    config = validate_config(load_yaml(args.config))
    if args.association_mode and (
        not bool(config["association"]["enabled"]) or not bool(config["local_association"]["enabled"])
    ):
        raise SystemExit(
            "--association-mode requires association.enabled=true and local_association.enabled=true "
            "in the configuration; use --no-association-mode for the legacy graph path"
        )
    write_effective_config(
        output_dir / "run_metadata" / "effective_config.yaml",
        config,
        metadata={"config_file": Path(args.config).name, "strict_metadata": strict_metadata_enabled(config)},
    )
    if args.overwrite and not args.resume:
        reset_overwrite_outputs(dirs)

    logger.info("Reading Gaussian catalog: %s", args.gaus_catalog)
    gaussians, _columns = normalized_gaussian_dataframe(read_gaussian_catalog(args.gaus_catalog))
    validate_normalized_gaussian_dataframe(gaussians, context=f"Gaussian catalogue {args.gaus_catalog}")
    logger.info("Gaussian catalog rows: %d", len(gaussians))

    reader = H5CutoutReader(args.h5_path, config_h5=config.get("h5", {}))
    logger.info("H5 image key: %s", reader.keys.image_key)
    logger.info("Number of cutouts available: %d", len(reader))

    status_path = dirs["logs"] / "status.csv"
    completed = done_cutouts(status_path) if args.resume else set()

    indices = reader.iter_indices(args.start_index, args.end_index, args.limit)
    logger.info("Selected %d cutouts", len(indices))
    if not indices:
        raise SystemExit("No cutouts selected; check --start-index/--end-index/--limit and the H5 input")
    run_status_records: list[dict] = []
    selected_cutout_ids: set[str] = set()
    seen_cutout_ids: set[str] = set()
    for index in indices:
        # Resume skips cutouts already marked complete in status.csv.
        start = now_iso()
        cutout_id = f"index_{index}"
        try:
            cutout = reader.read(index)
            validate_cutout_physical_metadata(cutout, config)
            cutout_id = str(cutout.cutout_id)
            if cutout_id in seen_cutout_ids:
                duplicate_id = cutout_id
                cutout_id = f"index_{index}"
                raise ValueError(f"duplicate cutout_id encountered in selected H5 indices: {duplicate_id}")
            validate_identifier(cutout_id, context="cutout_id")
            seen_cutout_ids.add(cutout_id)
            selected_cutout_ids.add(cutout_id)
            if args.resume and not args.overwrite and cutout_id in completed:
                logger.info("Skipping %s because status is done", cutout_id)
                continue
            logger.info("Processing %s index=%s", cutout_id, index)
            merged, _edges, components, association_groups = process_cutout(
                cutout,
                gaussians,
                config,
                dirs,
                args.make_figures,
                association_mode=args.association_mode,
            )
            record = {
                "cutout_id": cutout_id,
                "status": "done",
                "time_start": start,
                "time_end": now_iso(),
                "n_gaussians": int(len(components)),
                "n_merged_sources": int(len(merged)),
                "n_association_groups": int(len(association_groups)),
                "failure_reason": "",
            }
            update_status(status_path, record)
            run_status_records.append(record)
            logger.info(
                "Done %s: n_gaussians=%d n_merged_sources=%d n_association_groups=%d",
                cutout_id,
                len(components),
                len(merged),
                len(association_groups),
            )
        except Exception as exc:
            selected_cutout_ids.add(cutout_id)
            reason = str(exc)
            if args.debug:
                reason = traceback.format_exc()
                logger.error("Failed %s\n%s", cutout_id, reason)
            else:
                logger.error("Failed %s: %s", cutout_id, reason)
            record = {
                "cutout_id": cutout_id,
                "status": "failed",
                "time_start": start,
                "time_end": now_iso(),
                "n_gaussians": 0,
                "n_merged_sources": 0,
                "n_association_groups": 0,
                "failure_reason": reason,
            }
            update_status(status_path, record)
            run_status_records.append(record)

    combine_partials(output_dir, allowed_cutouts=done_cutouts(status_path) if status_path.exists() else set())
    logger.info("Wrote merged catalogs under %s", output_dir / "catalogs")
    # Include failures already present in status.csv when --resume skips them;
    # a resumed command must not report success while selected items remain
    # failed from an earlier attempt.
    final_status = load_status(status_path)
    selected_status = final_status[final_status["cutout_id"].astype(str).isin(selected_cutout_ids)] if selected_cutout_ids else final_status.iloc[0:0]
    failure = failed_status_message(selected_status.to_dict(orient="records"), "Local association")
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
