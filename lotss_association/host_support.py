"""Host-catalogue scoring for parent-link candidates.

The production Stage 2 path in ``parent_links`` queries CatWISE/AllWISE through
the helpers defined here.  Host evidence is recorded as a diagnostic; it can
support or contradict a radio candidate but never replaces the radio gate.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from ._shared import select_columns as _with_columns
from .config import HOST_SUPPORT_DEFAULTS
from .host_query import HOST_QUERY_LOG_COLUMNS, HOST_RAW_COLUMNS, HostQueryClient
from .utils import safe_float

HOST_CANDIDATE_COLUMNS = [
    "cutout_id",
    "parent_candidate_id",
    "host_catalog",
    "host_id",
    "host_ra",
    "host_dec",
    "host_sep_midpoint_arcsec",
    "host_perp_offset_beam",
    "host_fractional_position",
    "W1",
    "W2",
    "W1_W2",
    "W1_snr",
    "W2_snr",
    "host_score",
    "host_quality",
    "host_flags",
]


def host_support_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = copy.deepcopy(HOST_SUPPORT_DEFAULTS)
    raw = (config.get("host_support", {}) or {}).copy()
    out = dict(defaults)
    nested_keys = {
        "host_quality_thresholds",
        "geometry",
        "wise_color",
        "score_weights",
        "detection_scoring",
        "parent_quality",
    }
    out.update({key: value for key, value in raw.items() if key not in nested_keys})
    out["host_quality_thresholds"] = dict(defaults["host_quality_thresholds"])
    out["host_quality_thresholds"].update(raw.get("host_quality_thresholds", {}) or {})
    out["geometry"] = dict(defaults["geometry"])
    out["geometry"].update(raw.get("geometry", {}) or {})
    out["wise_color"] = dict(defaults["wise_color"])
    out["wise_color"].update(raw.get("wise_color", {}) or {})
    out["score_weights"] = dict(defaults["score_weights"])
    out["score_weights"].update(raw.get("score_weights", {}) or {})
    out["detection_scoring"] = dict(defaults["detection_scoring"])
    out["detection_scoring"].update(raw.get("detection_scoring", {}) or {})
    out["parent_quality"] = dict(defaults["parent_quality"])
    out["parent_quality"].update(raw.get("parent_quality", {}) or {})
    return out


def _quality_rank(value: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(str(value), 0)


def min_host_quality_rank(cfg: dict[str, Any]) -> int:
    """Return the configured minimum quality rank for midpoint host support."""

    return _quality_rank(str(cfg.get("min_host_quality_for_default_candidate", "medium")))


def _midpoint_ra_dec(row: pd.Series, group_by_id: dict[str, pd.Series]) -> tuple[float, float]:
    g1 = group_by_id.get(str(row.get("local_group_id_1")))
    g2 = group_by_id.get(str(row.get("local_group_id_2")))
    if g1 is None or g2 is None:
        return float("nan"), float("nan")
    ra1, dec1 = safe_float(g1.get("ra")), safe_float(g1.get("dec"))
    ra2, dec2 = safe_float(g2.get("ra")), safe_float(g2.get("dec"))
    if not np.all(np.isfinite([ra1, dec1, ra2, dec2])):
        return float("nan"), float("nan")
    c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
    c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")
    sep = c1.separation(c2)
    pa = c1.position_angle(c2)
    mid = c1.directional_offset_by(pa, sep / 2.0)
    return float(mid.ra.deg), float(mid.dec.deg)


def _host_search_radius(row: pd.Series, cfg: dict[str, Any]) -> float:
    sep = safe_float(row.get("center_distance_arcsec"), 0.0)
    radius = float(cfg["search_radius_fraction_of_sep"]) * sep
    radius = max(float(cfg["min_search_radius_arcsec"]), radius)
    radius = min(float(cfg["max_search_radius_arcsec"]), radius)
    return float(radius)


def _axis_geometry(
    host_ra: float,
    host_dec: float,
    row: pd.Series,
    group_by_id: dict[str, pd.Series],
    beam_arcsec: float,
) -> tuple[float, float, float]:
    g1 = group_by_id.get(str(row.get("local_group_id_1")))
    g2 = group_by_id.get(str(row.get("local_group_id_2")))
    if g1 is None or g2 is None:
        return float("nan"), float("nan"), float("nan")
    c1 = SkyCoord(safe_float(g1.get("ra")) * u.deg, safe_float(g1.get("dec")) * u.deg, frame="icrs")
    c2 = SkyCoord(safe_float(g2.get("ra")) * u.deg, safe_float(g2.get("dec")) * u.deg, frame="icrs")
    ch = SkyCoord(float(host_ra) * u.deg, float(host_dec) * u.deg, frame="icrs")
    sep12 = c1.separation(c2).arcsec
    if sep12 <= 0:
        return float("nan"), float("nan"), float("nan")
    sep1h = c1.separation(ch).arcsec
    pa12 = c1.position_angle(c2).rad
    pa1h = c1.position_angle(ch).rad
    along = sep1h * np.cos(pa1h - pa12)
    perp = abs(sep1h * np.sin(pa1h - pa12))
    frac = along / sep12
    return float(perp), float(perp / max(beam_arcsec, 1e-6)), float(frac)


def score_host_candidates(
    parent_row: pd.Series,
    raw_hosts: pd.DataFrame,
    group_by_id: dict[str, pd.Series],
    beam_arcsec: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    if raw_hosts is None or raw_hosts.empty:
        return pd.DataFrame(columns=HOST_CANDIDATE_COLUMNS)
    midpoint_ra = safe_float(parent_row.get("midpoint_ra"))
    midpoint_dec = safe_float(parent_row.get("midpoint_dec"))
    if not np.all(np.isfinite([midpoint_ra, midpoint_dec])):
        return pd.DataFrame(columns=HOST_CANDIDATE_COLUMNS)
    midpoint = SkyCoord(midpoint_ra * u.deg, midpoint_dec * u.deg, frame="icrs")
    radius = safe_float(parent_row.get("host_search_radius_arcsec"), 10.0)
    geom = cfg["geometry"]
    wise_cfg = cfg["wise_color"]
    score_weights = cfg["score_weights"]
    detection_cfg = cfg["detection_scoring"]
    thresholds = cfg["host_quality_thresholds"]
    records: list[dict[str, Any]] = []
    for _, host in raw_hosts.iterrows():
        # Record midpoint proximity and double-lobe axis consistency.
        host_ra = safe_float(host.get("host_ra"), float("nan"))
        host_dec = safe_float(host.get("host_dec"), float("nan"))
        if not np.all(np.isfinite([host_ra, host_dec])):
            continue
        hc = SkyCoord(host_ra * u.deg, host_dec * u.deg, frame="icrs")
        sep_mid = float(midpoint.separation(hc).arcsec)
        perp_arcsec, perp_beam, frac = _axis_geometry(host_ra, host_dec, parent_row, group_by_id, beam_arcsec)
        midpoint_closeness = float(np.clip(1.0 - sep_mid / max(radius, 1e-6), 0.0, 1.0))
        axis_consistency = float(np.clip(1.0 - perp_beam / max(float(geom["medium_max_perp_offset_beam"]), 1e-6), 0.0, 1.0))
        w1snr = safe_float(host.get("W1_snr"), float("nan"))
        w2snr = safe_float(host.get("W2_snr"), float("nan"))
        det_scores = []
        for value in [w1snr, w2snr]:
            if np.isfinite(value):
                det_scores.append(float(np.clip(value / max(float(detection_cfg["snr_scale"]), 1e-6), 0.0, 1.0)))
        wise_detection = (
            float(np.mean(det_scores))
            if det_scores
            else (
                float(detection_cfg["catalogued_w1_fallback"])
                if np.isfinite(safe_float(host.get("W1"), float("nan")))
                else 0.0
            )
        )
        w1_w2 = safe_float(host.get("W1_W2"), float("nan"))
        has_required_agn_color = bool(
            np.isfinite(w1_w2) and w1_w2 >= float(wise_cfg["agn_bonus_w1_w2_min"])
        )
        color_bonus = float(wise_cfg["agn_bonus"]) if np.isfinite(w1_w2) and w1_w2 >= float(wise_cfg["agn_bonus_w1_w2_min"]) else 0.0
        flags: list[str] = []
        cc_flags = str(host.get("cc_flags", "")).strip()
        artifact_penalty = 0.0
        if cc_flags and cc_flags.lower() not in {"0000", "0", "nan", "none"}:
            artifact_penalty = float(score_weights["artifact_penalty"])
            flags.append(f"cc_flags={cc_flags}")
        score = (
            float(score_weights["midpoint_closeness"]) * midpoint_closeness
            + float(score_weights["axis_consistency"]) * axis_consistency
            + float(score_weights["wise_detection"]) * wise_detection
            + color_bonus
            - artifact_penalty
        )
        if (
            score >= float(thresholds["high"])
            and perp_beam <= float(geom["high_max_perp_offset_beam"])
            and float(geom["high_fractional_position_min"]) <= frac <= float(geom["high_fractional_position_max"])
            and (not bool(wise_cfg["require_agn_color"]) or has_required_agn_color)
        ):
            quality = "high"
        elif (
            score >= float(thresholds["medium"])
            and perp_beam <= float(geom["medium_max_perp_offset_beam"])
            and float(geom["medium_fractional_position_min"]) <= frac <= float(geom["medium_fractional_position_max"])
            and (not bool(wise_cfg["require_agn_color"]) or has_required_agn_color)
        ):
            quality = "medium"
        elif sep_mid <= radius:
            quality = "low"
        else:
            quality = "none"
        records.append(
            {
                "cutout_id": parent_row.get("cutout_id"),
                "parent_candidate_id": parent_row.get("parent_candidate_id"),
                "host_catalog": host.get("catalogue", ""),
                "host_id": host.get("host_id", ""),
                "host_ra": host_ra,
                "host_dec": host_dec,
                "host_sep_midpoint_arcsec": sep_mid,
                "host_perp_offset_beam": perp_beam,
                "host_fractional_position": frac,
                "W1": safe_float(host.get("W1"), float("nan")),
                "W2": safe_float(host.get("W2"), float("nan")),
                "W1_W2": w1_w2,
                "W1_snr": w1snr,
                "W2_snr": w2snr,
                "host_score": float(score),
                "host_quality": quality,
                "host_flags": ";".join(flags),
            }
        )
    frame = _with_columns(pd.DataFrame(records), HOST_CANDIDATE_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["host_score", "host_sep_midpoint_arcsec"], ascending=[False, True])


def _query_hosts_for_pair(
    parent_row: pd.Series,
    host_client: HostQueryClient,
    cfg: dict[str, Any],
    max_host_queries_state: dict[str, int],
    max_host_queries: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    logs: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    any_failed = False
    query_status = "not_queried"
    for catalogue in cfg["catalog_priority"]:
        if max_host_queries is not None and max_host_queries_state["count"] >= max_host_queries:
            query_status = "max_host_queries_reached"
            break
        result = host_client.query_catalogue(
            safe_float(parent_row.get("midpoint_ra")),
            safe_float(parent_row.get("midpoint_dec")),
            safe_float(parent_row.get("host_search_radius_arcsec")),
            str(catalogue),
        )
        max_host_queries_state["count"] += 1
        logs.append(result.log)
        status = str(result.log["status"].iloc[0]) if not result.log.empty else ""
        if status == "failed":
            any_failed = True
        if not result.results.empty:
            raw_frames.append(result.results)
            query_status = f"{catalogue}_results"
            break
        query_status = status or f"{catalogue}_empty"
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame(columns=HOST_RAW_COLUMNS)
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=HOST_QUERY_LOG_COLUMNS)
    return raw, log, query_status, any_failed
