"""Diagnostic overview and zoom visualizations for local associations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle

from .beam import sky_pa_to_pixel_angle
from .utils import resolve_pixel_scale_arcsec, validate_identifier

DEFAULT_OVERVIEW = {
    "draw_all_labels": False,
    "draw_singletons": False,
    "label_min_components": 2,
    "max_labels": 30,
    "label_top_by": "LAS_arcsec",
    "max_gaussians_drawn": 500,
    "gaussian_marker_size": 4,
    "max_edges_drawn": 300,
    "draw_nonmerged_edges": False,
    "edge_min_score": None,
    "contour_thresholds": [2.5, 3.0],
}

DEFAULT_ZOOM = {
    "enabled": True,
    "max_zoom_per_cutout": 20,
    "select_min_components": 2,
    "select_top_las": 10,
    "select_top_confidence": 10,
    "padding_pix": 100,
    "min_size_pix": 256,
    "max_size_pix": 1024,
    "contour_thresholds": [2.0, 2.5, 3.0, 5.0],
    "draw_gaussian_ids": True,
    "draw_edge_scores": True,
}


def _plot_pixel_scale(config: dict[str, Any] | None, *frames: pd.DataFrame | None) -> float:
    for frame in frames:
        if frame is not None and not frame.empty and "pixel_scale_arcsec" in frame:
            return resolve_pixel_scale_arcsec(frame["pixel_scale_arcsec"].iloc[0], config, context="visualization pixel scale")
    return resolve_pixel_scale_arcsec(None, config, context="visualization pixel scale")


def _display_image(image: np.ndarray, stretch: str = "asinh", percent_clip: tuple[float, float] = (1, 99.5)) -> np.ndarray:
    data = np.asarray(image, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data)
    lo, hi = np.nanpercentile(finite, percent_clip)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    clipped = np.nan_to_num(np.clip(data, lo, hi), nan=lo)
    if stretch == "asinh":
        scale = np.nanstd(clipped)
        if not np.isfinite(scale) or scale <= 0:
            scale = max(hi - lo, 1.0)
        clipped = np.arcsinh((clipped - lo) / scale)
    return clipped


def _viz_config(config: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = (config or {}).get("visualization", {}) or {}
    viz = dict(raw)
    overview = dict(DEFAULT_OVERVIEW)
    overview.update(raw.get("overview", {}) or {})
    zoom = dict(DEFAULT_ZOOM)
    zoom.update(raw.get("zoom", {}) or {})
    return viz, overview, zoom


def _short_source_id(value: Any, fallback_idx: int = 0) -> str:
    text = str(value) if value is not None and str(value) != "nan" else ""
    match = re.search(r"(?:_a|a)(\d+)$", text)
    if match:
        return f"a{int(match.group(1)):03d}"
    match = re.search(r"(?:_m|m)(\d+)$", text)
    if match:
        return f"m{int(match.group(1)):03d}"
    return f"a{fallback_idx:03d}"


def _as_bool(value: Any) -> Any:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value
        return value.astype(str).str.lower().isin(["true", "1", "yes"])
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    try:
        values = [float(item.strip()) for item in str(value).split(",")]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 4 or not np.isfinite(values).all():
        return None
    return tuple(values)  # type: ignore[return-value]


def _component_ids(row: pd.Series) -> set[int]:
    raw = row.get("component_ids", "")
    ids: set[int] = set()
    for item in str(raw).split(","):
        try:
            ids.add(int(float(item.strip())))
        except (TypeError, ValueError, OverflowError):
            continue
    return ids


def _is_association_catalog(sources: pd.DataFrame | None) -> bool:
    return sources is not None and not sources.empty and "association_group_id" in sources


def _n_components_column(sources: pd.DataFrame) -> str:
    return "n_gaussians" if "n_gaussians" in sources else "n_components"


def _score_column(sources: pd.DataFrame) -> str:
    return "association_score_mean" if "association_score_mean" in sources else "merge_confidence"


def _edge_score_column(edges: pd.DataFrame) -> str:
    return "association_score" if edges is not None and not edges.empty and "association_score" in edges else "merge_score"


def _edge_decision_column(edges: pd.DataFrame) -> str:
    return "association_decision" if edges is not None and not edges.empty and "association_decision" in edges else "merge_decision"


def _source_id_column(sources: pd.DataFrame) -> str:
    return "association_group_id" if "association_group_id" in sources else "merged_source_id"


def _select_gaussians(components: pd.DataFrame, overview: dict[str, Any]) -> pd.DataFrame:
    if components is None or components.empty:
        return pd.DataFrame()
    limit = int(overview.get("max_gaussians_drawn", 500) or 0)
    if limit <= 0 or len(components) <= limit:
        return components
    if "_peak_flux" in components:
        return components.sort_values("_peak_flux", ascending=False).head(limit)
    return components.head(limit)


def _select_edges(edges: pd.DataFrame, overview: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    if edges is None or edges.empty:
        return pd.DataFrame(), 0
    work = edges.copy()
    decision = _edge_decision_column(work)
    score = _edge_score_column(work)
    if not overview.get("draw_nonmerged_edges", False) and decision in work:
        work = work[_as_bool(work[decision])]
    minimum = overview.get("edge_min_score")
    if minimum is not None and score in work:
        work = work[pd.to_numeric(work[score], errors="coerce") >= float(minimum)]
    total = len(work)
    limit = int(overview.get("max_edges_drawn", 300) or 0)
    if limit > 0 and len(work) > limit:
        work = work.sort_values(score, ascending=False).head(limit)
    return work, total


def select_important_sources(merged_sources: pd.DataFrame, overview: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """Select groups that deserve labels in an overview figure."""

    if merged_sources is None or merged_sources.empty:
        return pd.DataFrame(), 0
    work = merged_sources.copy()
    n_col = _n_components_column(work)
    score_col = _score_column(work)
    for col in [n_col, "LAS_arcsec", "LAS_beam", score_col]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    selected = pd.Series(False, index=work.index)
    if overview.get("draw_all_labels", False):
        selected[:] = True
    elif overview.get("draw_singletons", False):
        selected[:] = True
    elif n_col in work:
        selected |= work[n_col].fillna(0) >= int(overview.get("label_min_components", 2))
    if "association_quality" in work:
        selected |= work["association_quality"].astype(str).isin(["high", "medium", "suspicious", "artifact_risk"])
    if "quality_flags" in work:
        selected |= work["quality_flags"].astype(str).str.contains("large_connected_component", na=False)
    top_by = str(overview.get("label_top_by", "LAS_arcsec"))
    if top_by == "merge_confidence":
        top_by = score_col
    for col in [top_by, score_col]:
        if col in work:
            selected.loc[work.sort_values(col, ascending=False).head(int(overview.get("max_labels", 30))).index] = True
    out = work.loc[selected].copy()
    total = len(out)
    if not out.empty:
        sort_cols = [col for col in [top_by, score_col, n_col] if col in out]
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=False)
        limit = int(overview.get("max_labels", 30) or 0)
        if limit > 0:
            out = out.head(limit)
    return out, total


def _draw_contours(ax: Any, segmentation: Any, thresholds: list[float], xlim: tuple[int, int] | None = None, ylim: tuple[int, int] | None = None) -> None:
    colors = {2.0: "lime", 2.5: "orange", 3.0: "red", 5.0: "white"}
    for threshold in thresholds:
        if len(segmentation.thresholds) == 0:
            continue
        idx = int(np.argmin(np.abs(np.asarray(segmentation.thresholds, dtype=float) - float(threshold))))
        mask = segmentation.masks[idx].astype(bool)
        if xlim is not None and ylim is not None:
            mask = mask[ylim[0] : ylim[1], xlim[0] : xlim[1]]
        if mask.any():
            ax.contour(mask, levels=[0.5], colors=[colors.get(float(threshold), "yellow")], linewidths=0.55, alpha=0.65)


def _draw_gaussian_ellipse(
    ax: Any,
    row: pd.Series,
    pixel_scale_arcsec: float,
    offset: tuple[float, float] = (0, 0),
    config: dict[str, Any] | None = None,
) -> None:
    try:
        major = float(row.get("_dc_maj", row.get("_maj", np.nan)))
        minor = float(row.get("_dc_min", row.get("_min", np.nan)))
        pa = float(row.get("_dc_pa", row.get("_pa", np.nan)))
    except (TypeError, ValueError, OverflowError):
        return
    if not np.isfinite([major, minor, pa]).all() or major <= 0 or minor <= 0:
        return
    beam = (config or {}).get("beam", {}) or {}
    pa_pixel = sky_pa_to_pixel_angle(
        pa,
        ra_axis_sign=float(beam.get("ra_axis_sign", 1.0) or 1.0),
        dec_axis_sign=float(beam.get("dec_axis_sign", 1.0) or 1.0),
    )
    ax.add_patch(Ellipse(
        (float(row["x"]) - offset[0], float(row["y"]) - offset[1]),
        width=max(major / max(pixel_scale_arcsec, 1e-6), 1.0),
        height=max(minor / max(pixel_scale_arcsec, 1e-6), 1.0),
        angle=pa_pixel,
        fill=False,
        lw=0.7,
        edgecolor="cyan",
        alpha=0.75,
    ))


def _text_effects() -> list[Any]:
    return [pe.withStroke(linewidth=1.6, foreground="black", alpha=0.8)]


def _draw_edges(ax: Any, edges: pd.DataFrame, components: pd.DataFrame, offset: tuple[float, float] = (0, 0), draw_scores: bool = False) -> None:
    if edges is None or edges.empty or components is None or components.empty or "component_index" not in components:
        return
    by_idx = components.set_index("component_index")
    for _, edge in edges.iterrows():
        try:
            left, right = int(edge["component_index_1"]), int(edge["component_index_2"])
            ri, rj = by_idx.loc[left], by_idx.loc[right]
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        x = [float(ri["x"]) - offset[0], float(rj["x"]) - offset[0]]
        y = [float(ri["y"]) - offset[1], float(rj["y"]) - offset[1]]
        edge_type = str(edge.get("edge_type", "strong"))
        color, alpha, width = {"weak": ("deepskyblue", 0.55, 0.55), "rejected": ("gray", 0.18, 0.4)}.get(edge_type, ("white", 0.42, 0.65))
        ax.plot(x, y, color=color, lw=width, alpha=alpha)
        score = _edge_score_column(pd.DataFrame([edge]))
        if draw_scores and score in edge:
            ax.text(sum(x) / 2, sum(y) / 2, f"{float(edge[score]):.1f}", color="white", fontsize=5, path_effects=_text_effects())


def _overview_title(cutout_id: str, components: pd.DataFrame, edges: pd.DataFrame, sources: pd.DataFrame, shown_edges: int, total_edges: int, shown_labels: int, total_labels: int) -> str:
    decision = _edge_decision_column(edges) if edges is not None and not edges.empty else ""
    n_edges = int(_as_bool(edges[decision]).sum()) if decision in edges else 0
    n_col = _n_components_column(sources) if sources is not None else "n_components"
    n_values = pd.to_numeric(sources.get(n_col, pd.Series(dtype=float)), errors="coerce") if sources is not None else pd.Series(dtype=float)
    n_multi = int((n_values >= 2).sum())
    max_size = int(n_values.max()) if len(n_values.dropna()) else 0
    return (
        f"{cutout_id} | gauss={len(components)} assoc_edges={n_edges} groups={len(sources)} "
        f"multi={n_multi} max_group={max_size}\nlabels shown: {shown_labels} / {total_labels} | "
        f"edges shown: {shown_edges} / {total_edges}"
    )


def plot_cutout_overview(cutout: Any, segmentation: Any, components: pd.DataFrame, edges: pd.DataFrame, merged_sources: pd.DataFrame, output_path: str | Path, config: dict[str, Any] | None = None) -> Path:
    """Generate a full-cutout radio and S/N overview."""

    viz, overview, _zoom = _viz_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels, total_labels = select_important_sources(merged_sources, overview)
    edges_to_draw, total_edges = _select_edges(edges, overview)
    gaussians = _select_gaussians(components, overview)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.7), constrained_layout=True)
    axes[0].imshow(_display_image(cutout.image, viz.get("stretch", "asinh"), tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    finite_snr = np.asarray(segmentation.snr_map)[np.isfinite(segmentation.snr_map)]
    if finite_snr.size:
        vmin, vmax = np.nanpercentile(finite_snr, [1, 99.5])
        axes[1].imshow(segmentation.snr_map, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    else:
        axes[1].imshow(segmentation.snr_map, origin="lower", cmap="magma")
    ellipse_scale = _plot_pixel_scale(config, gaussians) if viz.get("draw_gaussian_ellipses", True) and not gaussians.empty else None
    for ax in axes:
        _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
        if not gaussians.empty:
            ax.scatter(gaussians["x"], gaussians["y"], s=float(overview.get("gaussian_marker_size", 4)), c="cyan", alpha=0.28, linewidths=0)
            if ellipse_scale is not None:
                for _, gaussian in gaussians.iterrows():
                    _draw_gaussian_ellipse(ax, gaussian, ellipse_scale, config=config)
            if overview.get("draw_gaussian_ids", False) or overview.get("draw_component_ids", False):
                for _, gaussian in gaussians.iterrows():
                    labels: list[str] = []
                    if overview.get("draw_gaussian_ids", False):
                        labels.append(str(gaussian.get("_gaussian_id", "")))
                    if overview.get("draw_component_ids", False):
                        labels.append(f"c{int(float(gaussian.get('component_index', 0)))}")
                    label = "/".join(item for item in labels if item and item != "nan")
                    if label:
                        ax.text(
                            float(gaussian["x"]) + 3,
                            float(gaussian["y"]) + 3,
                            label,
                            color="cyan",
                            fontsize=5,
                            path_effects=_text_effects(),
                        )
        _draw_edges(ax, edges_to_draw, components)
        for idx, (_, row) in enumerate(labels.iterrows()):
            bbox = _bbox_tuple(row.get("bounding_box", ""))
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            ax.add_patch(Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1, fill=False, lw=0.75, edgecolor="yellow", alpha=0.65))
            label = _short_source_id(row.get(_source_id_column(labels), row.get("merged_source_id")), idx)
            ax.text(x0, max(0, y0 - 4), label, color="yellow", fontsize=6, va="top", path_effects=_text_effects())
        ax.set_xlim(0, cutout.image.shape[1] - 1)
        ax.set_ylim(0, cutout.image.shape[0] - 1)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
    axes[0].set_title("radio")
    axes[1].set_title("S/N")
    fig.suptitle(_overview_title(cutout.cutout_id, components, edges, merged_sources, len(edges_to_draw), total_edges, len(labels), total_labels), fontsize=9)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _zoom_window(bbox: tuple[float, float, float, float], image_shape: tuple[int, int], zoom: dict[str, Any]) -> tuple[int, int, int, int]:
    height, width = image_shape
    x0, y0, x1, y1 = bbox
    pad = int(zoom.get("padding_pix", 100))
    size = min(max(max(x1 - x0, y1 - y0) + 2 * pad + 1, int(zoom.get("min_size_pix", 256))), int(zoom.get("max_size_pix", 1024)))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    left, bottom = int(round(cx - size / 2)), int(round(cy - size / 2))
    left = min(max(left, 0), max(width - size, 0))
    bottom = min(max(bottom, 0), max(height - size, 0))
    return left, bottom, min(width, left + size), min(height, bottom + size)


def select_zoom_sources(merged_sources: pd.DataFrame, zoom: dict[str, Any]) -> pd.DataFrame:
    if merged_sources is None or merged_sources.empty or not zoom.get("enabled", True):
        return pd.DataFrame()
    work = merged_sources.copy()
    n_col, score_col = _n_components_column(work), _score_column(work)
    for col in [n_col, "LAS_arcsec", "LAS_beam", score_col]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    selected = pd.Series(False, index=work.index)
    if n_col in work:
        selected |= work[n_col].fillna(0) >= int(zoom.get("select_min_components", 2))
    if "association_quality" in work:
        selected |= work["association_quality"].astype(str).isin(["high", "medium", "suspicious", "artifact_risk"])
    for col, limit in [("LAS_arcsec", "select_top_las"), (score_col, "select_top_confidence")]:
        if col in work:
            selected.loc[work.sort_values(col, ascending=False).head(int(zoom.get(limit, 10))).index] = True
    out = work.loc[selected]
    if out.empty:
        return out
    return out.sort_values([col for col in [n_col, "LAS_arcsec", score_col] if col in out], ascending=False).head(int(zoom.get("max_zoom_per_cutout", 20)))


def plot_source_zoom(cutout: Any, segmentation: Any, components: pd.DataFrame, edges: pd.DataFrame, source_row: pd.Series, output_path: str | Path, config: dict[str, Any] | None = None, fallback_idx: int = 0) -> Path | None:
    """Generate one local association zoom figure."""

    viz, _overview, zoom = _viz_config(config)
    bbox = _bbox_tuple(source_row.get("bounding_box", ""))
    if bbox is None:
        return None
    x0, y0, x1, y1 = _zoom_window(bbox, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids = _component_ids(source_row)
    src = components[components["component_index"].astype(int).isin(ids)].copy() if ids and "component_index" in components else pd.DataFrame()
    if ids and edges is not None and not edges.empty:
        decision = _edge_decision_column(edges)
        src_edges = edges[_as_bool(edges[decision]) & edges["component_index_1"].astype(int).isin(ids) & edges["component_index_2"].astype(int).isin(ids)]
    else:
        src_edges = pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(_display_image(cutout.image[y0:y1, x0:x1], viz.get("stretch", "asinh"), tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src.empty:
        ax.scatter(src["x"] - x0, src["y"] - y0, s=18, c="cyan", alpha=0.8, edgecolors="black")
        scale = _plot_pixel_scale(config, src)
        for _, comp in src.iterrows():
            _draw_gaussian_ellipse(ax, comp, scale, offset=(x0, y0), config=config)
            if zoom.get("draw_gaussian_ids", True):
                ax.text(float(comp["x"]) - x0 + 4, float(comp["y"]) - y0 + 4, str(int(float(comp.get("_gaussian_id", comp.get("component_index", 0))))), color="cyan", fontsize=5, path_effects=_text_effects())
    _draw_edges(ax, src_edges, components, offset=(x0, y0), draw_scores=bool(zoom.get("draw_edge_scores", True)))
    group_id = source_row.get("association_group_id", source_row.get("merged_source_id"))
    ax.set_title(f"{cutout.cutout_id} {_short_source_id(group_id, fallback_idx)} | n={int(source_row.get('n_gaussians', source_row.get('n_components', 0)))} | quality={source_row.get('association_quality', source_row.get('merge_confidence', ''))}", fontsize=9)
    ax.set_xlim(0, max(0, x1 - x0 - 1))
    ax.set_ylim(0, max(0, y1 - y0 - 1))
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_cutout_result(cutout: Any, segmentation: Any, components: pd.DataFrame, edges: pd.DataFrame, merged_sources: pd.DataFrame, output_path: str | Path, config: dict[str, Any] | None = None) -> Path:
    """Backward-compatible alias for :func:`plot_cutout_overview`."""

    return plot_cutout_overview(cutout, segmentation, components, edges, merged_sources, output_path, config)


def plot_cutout_all(cutout: Any, segmentation: Any, components: pd.DataFrame, edges: pd.DataFrame, merged_sources: pd.DataFrame, output_dir: str | Path, config: dict[str, Any] | None = None, overview_only: bool = False, zoom_only: bool = False) -> dict[str, list[Path]]:
    """Generate local association overview and zoom figures."""

    validate_identifier(cutout.cutout_id, context="cutout_id")
    _viz, _overview, zoom = _viz_config(config)
    output_dir = Path(output_dir)
    overview_paths: list[Path] = []
    zoom_paths: list[Path] = []
    if not zoom_only:
        overview_paths.append(plot_cutout_overview(cutout, segmentation, components, edges, merged_sources, output_dir / "overview" / f"{cutout.cutout_id}.png", config))
    if not overview_only and zoom.get("enabled", True):
        for idx, (_, row) in enumerate(select_zoom_sources(merged_sources, zoom).iterrows()):
            path = output_dir / "zoom" / f"{cutout.cutout_id}_{_short_source_id(row.get(_source_id_column(merged_sources), row.get('merged_source_id')), idx)}.png"
            written = plot_source_zoom(cutout, segmentation, components, edges, row, path, config, fallback_idx=idx)
            if written is not None:
                zoom_paths.append(written)
    return {"overview": overview_paths, "zoom": zoom_paths}


def _local_group_ids(row: pd.Series) -> set[str]:
    """Return the stable local-group identifiers referenced by a parent row."""

    values = {str(row.get("local_group_id_1", "")).strip(), str(row.get("local_group_id_2", "")).strip()}
    raw = row.get("local_group_ids")
    if raw is not None and str(raw).strip() not in {"", "nan"}:
        values.update(item.strip() for item in str(raw).split(","))
    return {value for value in values if value and value.lower() != "nan"}


def _int_list(value: Any) -> list[int]:
    """Parse a comma-separated component-id field without raising on missing values."""

    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return []
    output: list[int] = []
    for item in str(value).split(","):
        try:
            output.append(int(float(item.strip())))
        except (TypeError, ValueError):
            continue
    return output


def _parent_link_display_groups(
    local_groups: pd.DataFrame,
    source_morph_table: pd.DataFrame | None,
    parent_candidates: pd.DataFrame | None,
) -> pd.DataFrame:
    """Select non-point-like local groups for the Stage 2 overview."""

    if local_groups is None or local_groups.empty:
        return pd.DataFrame()
    work = local_groups.copy()
    if source_morph_table is not None and not source_morph_table.empty and "association_group_id" in source_morph_table:
        columns = [
            "association_group_id",
            "source_morph_class",
            "is_point_like",
            "is_lobe_candidate",
            "is_artifact_risk",
            "hard_compact_veto",
            "noise_artifact_veto",
            "isolated_compact_veto",
            "near_extended_lobe_candidate",
        ]
        available = [column for column in columns if column in source_morph_table]
        work = work.merge(
            source_morph_table[available].drop_duplicates("association_group_id"),
            on="association_group_id",
            how="left",
        )
    for column in (
        "is_point_like",
        "is_lobe_candidate",
        "is_artifact_risk",
        "hard_compact_veto",
        "noise_artifact_veto",
        "isolated_compact_veto",
        "near_extended_lobe_candidate",
    ):
        if column not in work:
            work[column] = False
    if "source_morph_class" not in work:
        work["source_morph_class"] = "resolved_single"
    candidate_ids: set[str] = set()
    if parent_candidates is not None and not parent_candidates.empty:
        for _, row in parent_candidates.iterrows():
            candidate_ids.update(_local_group_ids(row))
    quality = work.get("association_quality", pd.Series("", index=work.index)).astype(str)
    hidden_classes = {"point_like", "point_like_or_compact", "compact_resolved_single", "noise_or_artifact"}
    hard_hidden = work["source_morph_class"].astype(str).isin(hidden_classes)
    for column in ("is_point_like", "hard_compact_veto", "noise_artifact_veto", "isolated_compact_veto"):
        hard_hidden |= _as_bool(work[column])
    candidate_member = work["association_group_id"].astype(str).isin(candidate_ids)
    show = ~hard_hidden | _as_bool(work["is_lobe_candidate"]) | _as_bool(work["near_extended_lobe_candidate"])
    show |= quality.eq("suspicious") | candidate_member
    out = work.loc[show].copy()
    out["local_quality"] = out.get("association_quality", "medium")
    out["local_group_id"] = out.get("association_group_id", "")
    return out


def _draw_parent_link_local_boxes(
    ax: Any,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    label: bool = True,
) -> None:
    if local_groups is None or local_groups.empty:
        return
    for index, (_, row) in enumerate(local_groups.iterrows()):
        bbox = _bbox_tuple(row.get("bounding_box", ""))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        morph = str(row.get("source_morph_class", "resolved_single"))
        color = "lime" if morph == "lobe_candidate" else "gold"
        if morph == "artifact_risk" or str(row.get("association_quality", "")).lower() == "suspicious":
            color = "magenta"
        ax.add_patch(
            Rectangle(
                (x0 - offset[0], y0 - offset[1]),
                x1 - x0 + 1,
                y1 - y0 + 1,
                fill=False,
                lw=0.85,
                edgecolor=color,
                alpha=0.82,
            )
        )
        if label:
            group_id = str(row.get("local_group_id", f"l{index:03d}")).rsplit("_", 1)[-1]
            ax.text(
                x0 - offset[0],
                max(0, y0 - offset[1] - 4),
                group_id,
                color=color,
                fontsize=5.7,
                va="top",
                ha="left",
                path_effects=_text_effects(),
            )


def _draw_parent_link_parent_links(
    ax: Any,
    candidates: pd.DataFrame,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    draw_scores: bool = False,
) -> None:
    if candidates is None or candidates.empty or local_groups is None or local_groups.empty:
        return
    if "association_group_id" not in local_groups:
        return
    by_id = local_groups.set_index("association_group_id")
    quality = candidates.get("parent_candidate_quality", pd.Series("", index=candidates.index)).astype(str)
    work = candidates.loc[quality.isin({"high", "medium", "needs_host_check", "suspicious"})]
    for _, row in work.sort_values("parent_score_final", ascending=False).iterrows():
        left, right = str(row.get("local_group_id_1")), str(row.get("local_group_id_2"))
        if left not in by_id.index or right not in by_id.index:
            continue
        first, second = by_id.loc[left], by_id.loc[right]
        x = [float(first["centroid_x"]) - offset[0], float(second["centroid_x"]) - offset[0]]
        y = [float(first["centroid_y"]) - offset[1], float(second["centroid_y"]) - offset[1]]
        candidate_quality = str(row.get("parent_candidate_quality", "medium"))
        color = {"high": "cyan", "medium": "deepskyblue", "needs_host_check": "orange", "suspicious": "magenta"}.get(candidate_quality, "white")
        ax.plot(x, y, color=color, lw=1.35, alpha=0.86, linestyle="-" if candidate_quality in {"high", "medium"} else "--")
        bounds = [
            float(row.get("parent_bbox_xmin", np.nan)) - offset[0],
            float(row.get("parent_bbox_xmax", np.nan)) - offset[0],
            float(row.get("parent_bbox_ymin", np.nan)) - offset[1],
            float(row.get("parent_bbox_ymax", np.nan)) - offset[1],
        ]
        if np.all(np.isfinite(bounds)):
            ax.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1, fill=False, lw=1.1, edgecolor="cyan", alpha=0.72, linestyle=":"))
        if draw_scores:
            label = (
                f"sym={float(row.get('symmetry_score', np.nan)):.2f} score={float(row.get('lobe_pair_score', np.nan)):.2f}\n"
                f"host={row.get('host_evidence', '')} peak_host={row.get('lobe_peak_host_found', False)}\n"
                f"{candidate_quality} reason={row.get('rejection_reason', '')}"
            )
            ax.text(0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), label, color=color, fontsize=6, path_effects=_text_effects())


def plot_parent_link_cutout_overview(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    source_morph_table: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate the Stage 2 overview with local groups and parent candidates."""

    viz, overview, _zoom = _viz_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 8.0), constrained_layout=True)
    ax.imshow(_display_image(cutout.image, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
    gaussians = _select_gaussians(components, overview)
    if not gaussians.empty:
        ax.scatter(gaussians["x"], gaussians["y"], s=float(overview.get("gaussian_marker_size", 4)), c="cyan", alpha=0.13, linewidths=0)
    display_groups = _parent_link_display_groups(local_groups, source_morph_table, parent_candidates)
    _draw_parent_link_local_boxes(ax, display_groups)
    _draw_parent_link_parent_links(ax, parent_candidates, local_groups)
    ax.set_xlim(0, cutout.image.shape[1] - 1)
    ax.set_ylim(0, cutout.image.shape[0] - 1)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(f"{cutout.cutout_id} | local={len(local_groups)} | production parent-linking candidates={len(parent_candidates)} | shown_nonpoint={len(display_groups)}", fontsize=9)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_parent_link_parent_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidate_row: pd.Series,
    host_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one Stage 2 parent-candidate zoom figure."""

    viz, _overview, zoom = _viz_config(config)
    ids = _local_group_ids(parent_candidate_row)
    if "association_group_id" not in local_groups:
        return None
    member_groups = local_groups[local_groups["association_group_id"].astype(str).isin(ids)].copy()
    if member_groups.empty:
        return None
    boxes = [_bbox_tuple(row.get("bounding_box", "")) for _, row in member_groups.iterrows()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    union_box = (
        float(parent_candidate_row.get("parent_bbox_xmin", min(box[0] for box in boxes))),
        float(parent_candidate_row.get("parent_bbox_ymin", min(box[1] for box in boxes))),
        float(parent_candidate_row.get("parent_bbox_xmax", max(box[2] for box in boxes))),
        float(parent_candidate_row.get("parent_bbox_ymax", max(box[3] for box in boxes))),
    )
    x0, y0, x1, y1 = _zoom_window(union_box, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comp_ids: set[int] = set()
    for _, row in member_groups.iterrows():
        comp_ids.update(_int_list(row.get("component_ids", "")))
    src_components = components[components["component_index"].astype(int).isin(comp_ids)].copy() if comp_ids and "component_index" in components else pd.DataFrame()
    fig, ax = plt.subplots(1, 1, figsize=(7.7, 7.7), constrained_layout=True)
    ax.imshow(_display_image(cutout.image[y0:y1, x0:x1], stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=16, c="cyan", alpha=0.72, linewidths=0.2, edgecolors="black")
    local_boxes = member_groups.copy()
    local_boxes["local_group_id"] = local_boxes["association_group_id"]
    _draw_parent_link_local_boxes(ax, local_boxes, offset=(x0, y0))
    _draw_parent_link_parent_links(ax, pd.DataFrame([parent_candidate_row]), local_groups, offset=(x0, y0), draw_scores=True)
    short_id = str(parent_candidate_row.get("parent_candidate_id", f"pc{fallback_idx:03d}")).rsplit("_", 1)[-1]
    title = (
        f"{cutout.cutout_id} {short_id} | production parent-linking physics-aware parent candidate\n"
        f"quality={parent_candidate_row.get('parent_candidate_quality', '')} host_evidence={parent_candidate_row.get('host_evidence', '')} "
        f"sym={float(parent_candidate_row.get('symmetry_score', np.nan)):.2f} gap={float(parent_candidate_row.get('box_gap_beam_robust', np.nan)):.2f} beam"
    )
    ax.set_title(title, fontsize=8.0)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_parent_link_cutout_all(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    source_morph_table: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    host_candidates: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, list[Path]]:
    """Generate Stage 2 parent-linking overview and zoom figures."""

    output_dir = Path(output_dir)
    paths: dict[str, list[Path]] = {"overview": [], "parent_zoom": []}
    overview_path = output_dir / "overview" / f"{cutout.cutout_id}.png"
    paths["overview"].append(
        plot_parent_link_cutout_overview(
            cutout,
            segmentation,
            components,
            local_groups,
            source_morph_table,
            parent_candidates,
            overview_path,
            config,
        )
    )
    if parent_candidates is None or parent_candidates.empty:
        return paths
    score_column = "parent_score_final" if "parent_score_final" in parent_candidates else "parent_score"
    for index, (_, row) in enumerate(parent_candidates.sort_values(score_column, ascending=False).iterrows()):
        short_id = str(row.get("parent_candidate_id", f"pc{index:03d}")).rsplit("_", 1)[-1]
        written = plot_parent_link_parent_zoom(
            cutout,
            segmentation,
            components,
            local_groups,
            row,
            host_candidates,
            output_dir / "parent_zoom" / f"{cutout.cutout_id}_{short_id}.png",
            config,
            fallback_idx=index,
        )
        if written is not None:
            paths["parent_zoom"].append(written)
    return paths
