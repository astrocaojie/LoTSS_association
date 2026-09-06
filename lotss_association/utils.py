"""Shared utilities for the LoTSS Association pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .config import SCIENTIFIC_DEFAULTS

LOGGER_NAME = "lotss_association"


def strict_metadata_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether missing physical metadata must abort a run.

    Production configurations should enable this flag.  The default remains
    permissive for backwards compatibility with small synthetic examples.
    """

    runtime = (config or {}).get("runtime", {}) or {}
    return bool(runtime.get("strict_metadata", False))


def snr_map_config_kwargs(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the canonical S/N-map options for every public entry point."""

    config = config or {}
    return {
        "mean_mode": config.get("mean_mode", SCIENTIFIC_DEFAULTS["mean_mode"]),
        "rms_mode": config.get("rms_mode", SCIENTIFIC_DEFAULTS["rms_mode"]),
        "smooth_before_segmentation": bool(config.get("smooth_before_segmentation", SCIENTIFIC_DEFAULTS["smooth_before_segmentation"])),
        "gaussian_smooth_sigma_pix": float(config.get("gaussian_smooth_sigma_pix", SCIENTIFIC_DEFAULTS["gaussian_smooth_sigma_pix"])),
        "strict_metadata": strict_metadata_enabled(config),
    }


def segmentation_config_kwargs(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return canonical segmentation options for public entry points."""

    config = config or {}
    return {
        "thresholds": config.get("snr_thresholds", SCIENTIFIC_DEFAULTS["snr_thresholds"]),
        "min_mask_area_pix": int(config.get("min_mask_area_pix", SCIENTIFIC_DEFAULTS["min_mask_area_pix"])),
        "connectivity": int(config.get("connectivity", SCIENTIFIC_DEFAULTS["connectivity"])),
        "binary_opening": bool(config.get("binary_opening", SCIENTIFIC_DEFAULTS["binary_opening"])),
        "binary_closing": bool(config.get("binary_closing", SCIENTIFIC_DEFAULTS["binary_closing"])),
    }


def validate_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate user-facing association settings before processing any data.

    The scientific entry points call this once so malformed configurations do
    not produce an apparently successful empty run or fail halfway through a
    batch after writing partial outputs.
    """

    from .config import validate_mapping

    return validate_mapping(config)


def failed_status_message(
    records: Iterable[Mapping[str, Any]],
    label: str,
    *,
    id_key: str = "cutout_id",
) -> str | None:
    """Return a concise batch-failure message when any record is not done."""

    failed = [record for record in records if str(record.get("status", "")).strip().lower() != "done"]
    if not failed:
        return None
    identifiers = [str(record.get(id_key, "")).strip() for record in failed]
    identifiers = [identifier for identifier in identifiers if identifier]
    preview = ", ".join(identifiers[:10])
    if len(identifiers) > 10:
        preview += f", ... (+{len(identifiers) - 10} more)"
    suffix = f": {preview}" if preview else ""
    return f"{label} completed with {len(failed)} failed item(s){suffix}"


def resolve_pixel_scale_arcsec(
    value: Any,
    config: Mapping[str, Any] | None = None,
    *,
    context: str = "pixel scale",
) -> float:
    """Resolve a positive pixel scale from metadata or explicit configuration."""

    scale = safe_float(value, float("nan"))
    if np.isfinite(scale) and scale > 0:
        return float(scale)
    if value is not None:
        raise ValueError(f"{context} metadata must be finite and positive")
    configured = safe_float((config or {}).get("pixel_scale_arcsec"), float("nan"))
    if np.isfinite(configured) and configured > 0:
        return float(configured)
    raise ValueError(f"{context} is required: provide input metadata/WCS or pixel_scale_arcsec in config")


def validate_cutout_physical_metadata(cutout: Any, config: Mapping[str, Any] | None) -> None:
    """Validate physical metadata against the configured science contract.

    A cutout may provide pixel scale and restoring-beam values as H5 metadata,
    while the validated YAML provides the reference values used by the scoring
    code.  When both are present they must agree; otherwise one source could
    silently override the other and change beam-normalized decisions.
    """

    metadata = getattr(cutout, "metadata", None) or {}
    beam = (config or {}).get("beam", {}) or {}

    def finite_positive(value: Any, label: str) -> float | None:
        if value is None:
            return None
        number = safe_float(value, float("nan"))
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{label} metadata must be finite and positive")
        return float(number)

    metadata_scale = finite_positive(metadata.get("pixel_scale_arcsec"), "pixel scale")
    configured_scale = finite_positive((config or {}).get("pixel_scale_arcsec"), "pixel_scale_arcsec config")
    if metadata_scale is not None and configured_scale is not None and not np.isclose(
        metadata_scale, configured_scale, rtol=1e-3, atol=1e-6
    ):
        raise ValueError(
            "pixel scale mismatch between cutout metadata and config: "
            f"{metadata_scale:g} vs {configured_scale:g} arcsec/pixel"
        )

    metadata_beam = {
        "major_arcsec": metadata.get("beam_major_arcsec"),
        "minor_arcsec": metadata.get("beam_minor_arcsec"),
    }
    present_beam = [value is not None for value in metadata_beam.values()]
    if any(present_beam):
        if not all(present_beam):
            raise ValueError("cutout beam metadata must provide both major and minor axes")
        metadata_major = finite_positive(metadata_beam["major_arcsec"], "beam major")
        metadata_minor = finite_positive(metadata_beam["minor_arcsec"], "beam minor")
        configured_major = finite_positive(beam.get("major_arcsec"), "beam.major_arcsec config")
        configured_minor = finite_positive(beam.get("minor_arcsec"), "beam.minor_arcsec config")
        if configured_major is None or configured_minor is None:
            raise ValueError("beam.major_arcsec and beam.minor_arcsec must be configured when cutout metadata provides a beam")
        if not np.isclose(metadata_major, configured_major, rtol=1e-3, atol=1e-6) or not np.isclose(
            metadata_minor, configured_minor, rtol=1e-3, atol=1e-6
        ):
            raise ValueError(
                "beam mismatch between cutout metadata and config: "
                f"metadata=({metadata_major:g}, {metadata_minor:g}) arcsec, "
                f"config=({configured_major:g}, {configured_minor:g}) arcsec"
            )
        metadata_pa = metadata.get("beam_pa_deg")
        configured_pa = beam.get("pa_deg")
        if metadata_pa is not None and configured_pa is not None:
            metadata_pa_value = safe_float(metadata_pa, float("nan"))
            configured_pa_value = safe_float(configured_pa, float("nan"))
            if not np.isfinite(metadata_pa_value) or not np.isfinite(configured_pa_value):
                raise ValueError("beam position angle metadata and config must be finite")
            pa_delta = abs((metadata_pa_value - configured_pa_value + 90.0) % 180.0 - 90.0)
            if pa_delta > 0.1:
                raise ValueError(
                    "beam position-angle mismatch between cutout metadata and config: "
                    f"{metadata_pa_value:g} vs {configured_pa_value:g} degrees"
                )


def write_effective_config(
    path: str | Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist the resolved configuration and run metadata as YAML."""

    import yaml

    resolved = resolve_effective_config(config)
    config_hash = effective_config_sha256(resolved)
    metadata_out = dict(metadata or {})
    metadata_out.setdefault("config_sha256", config_hash)
    metadata_out.setdefault("software_version", _software_version())
    metadata_out.setdefault("git_commit", _git_commit())
    runtime = dict(resolved.get("runtime", {}) or {})
    runtime["effective_metadata"] = metadata_out
    resolved["runtime"] = runtime
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")


def effective_config_sha256(config: Mapping[str, Any]) -> str:
    """Return a stable hash of the effective scientific configuration."""

    clean = copy.deepcopy(dict(config))
    runtime = clean.get("runtime")
    if isinstance(runtime, Mapping):
        runtime = dict(runtime)
        runtime.pop("effective_metadata", None)
        clean["runtime"] = runtime
    clean.pop("_label_count_cache", None)
    payload = json.dumps(clean, default=str, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _software_version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except (ImportError, AttributeError, TypeError, ValueError):
        return "unknown"


def _git_commit() -> str:
    # Build systems can provide the source revision explicitly.  This is the
    # preferred provenance path for wheels, where the installed tree has no
    # .git directory.  Ignore malformed values rather than recording a value
    # that cannot identify a source revision.
    for env_name in ("LOTSS_ASSOCIATION_SOURCE_COMMIT", "SOURCE_COMMIT", "GITHUB_SHA"):
        candidate = os.environ.get(env_name, "").strip()
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", candidate):
            return candidate
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown"


def resolve_effective_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the package's scientific configuration defaults.

    Imports are local to avoid a module cycle: the scientific modules use the
    utilities in this file, while run wrappers call this function only after
    package import has completed.
    """

    from .association import _association_config, _association_weights
    from .config import (
        ASSOCIATION_TYPE_DEFAULTS,
        BEAM_DEFAULTS,
        GRAPH_DEFAULTS,
        GRAPH_WEIGHT_DEFAULTS,
        HOST_SUPPORT_DEFAULTS,
        LOCAL_SANITY_DEFAULTS,
        MATCHING_DEFAULTS,
        PARENT_LINK_DEFAULTS,
        PARENT_SEED_DEFAULTS,
        SCIENTIFIC_DEFAULTS,
    )
    from .feature_flags import feature_config
    from .morphology import _classification_config

    def merge_defaults(raw: Mapping[str, Any] | None, defaults: Mapping[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(dict(defaults))
        for key, value in (raw or {}).items():
            if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
                out[key] = merge_defaults(value, out[key])
            else:
                out[key] = copy.deepcopy(value)
        return out

    resolved = copy.deepcopy(dict(config))
    # Scientific scoring may attach private numpy caches to the live config;
    # never persist those transient implementation details in provenance YAML.
    resolved.pop("_label_count_cache", None)
    resolved["features"] = feature_config(resolved)
    resolved["matching"] = merge_defaults(resolved.get("matching"), MATCHING_DEFAULTS)
    for key, value in SCIENTIFIC_DEFAULTS.items():
        resolved.setdefault(key, copy.deepcopy(value))
    resolved["association"] = _association_config(resolved)
    resolved["association_types"] = merge_defaults(resolved.get("association_types"), ASSOCIATION_TYPE_DEFAULTS)
    resolved["weights_association"] = _association_weights(resolved)
    resolved["weights_graph"] = merge_defaults(resolved.get("weights_graph"), GRAPH_WEIGHT_DEFAULTS)
    resolved["graph_merge"] = merge_defaults(resolved.get("graph_merge"), GRAPH_DEFAULTS)
    resolved["beam_aware_classification"] = _classification_config(resolved)
    resolved["beam"] = merge_defaults(resolved.get("beam"), BEAM_DEFAULTS)
    resolved["local_association"] = merge_defaults(
        resolved.get("local_association"),
        {"enabled": True, "local_sanity": LOCAL_SANITY_DEFAULTS},
    )
    resolved["parent_seed_selection"] = merge_defaults(resolved.get("parent_seed_selection"), PARENT_SEED_DEFAULTS)
    resolved["host_support"] = merge_defaults(resolved.get("host_support"), HOST_SUPPORT_DEFAULTS)
    resolved["parent_linking"] = merge_defaults(resolved.get("parent_linking"), PARENT_LINK_DEFAULTS)
    return resolved


def setup_logging(debug: bool = False, log_path: str | Path | None = None) -> logging.Logger:
    """Configure package logging."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(stream_handler)

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the package logger."""

    return logging.getLogger(LOGGER_NAME)


def packaged_config_uri(filename: str) -> str:
    """Return the URI for a configuration shipped inside the package."""

    name = Path(filename).name
    if name != filename or not name.endswith(".yaml"):
        raise ValueError("packaged configuration filename must be a plain .yaml name")
    return f"package://data/{name}"


def validate_integer_indices(values: Iterable[Any], *, context: str = "component_index") -> np.ndarray:
    """Validate and normalize identifiers used as graph node indices."""

    normalized: list[int] = []
    int_info = np.iinfo(np.int64)
    for position, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{context}[{position}] must be an integer, not boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context}[{position}] must be finite and integer-valued") from exc
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError(f"{context}[{position}] must be finite and integer-valued")
        if numeric < int_info.min or numeric > int_info.max:
            raise ValueError(f"{context}[{position}] is outside the supported integer range")
        normalized.append(int(numeric))
    return np.asarray(normalized, dtype=np.int64)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        """SafeLoader variant that rejects duplicate mapping keys."""

        ...

    def construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate YAML configuration key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping,
    )
    source = str(path)
    if source.startswith("package://"):
        resource_name = source.removeprefix("package://").strip("/")
        parts = tuple(part for part in resource_name.split("/") if part)
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"invalid packaged configuration URI: {source}")
        from importlib.resources import files

        resource = files("lotss_association").joinpath(*parts)
        if not resource.is_file():
            raise FileNotFoundError(f"packaged configuration not found: {source}")
        handle = resource.open("r", encoding="utf-8")
    else:
        handle = open(path, encoding="utf-8")
    with handle:
        data = yaml.load(handle, Loader=UniqueKeyLoader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping, got {type(data).__name__}")
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_dumps_safe(value: Any) -> str:
    """Serialize simple debug payloads, converting numpy scalars and arrays."""

    def default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return str(obj)

    return json.dumps(value, default=default, sort_keys=True)


def first_existing_key(mapping: Mapping[str, Any], candidates: list[str]) -> str | None:
    """Return the first candidate key present in a mapping, case-insensitively."""

    lower_to_key = {str(key).lower(): str(key) for key in mapping.keys()}
    for candidate in candidates:
        key = lower_to_key.get(candidate.lower())
        if key is not None:
            return key
    return None


def robust_mad_rms(image: np.ndarray) -> float:
    """Estimate robust rms using MAD."""

    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return float("nan")
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.nanstd(finite))
    return float(rms)


def safe_float(value: Any, default: float = float("nan")) -> float:
    """Convert a value to float without raising."""

    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if math.isfinite(out):
        return out
    return default


def safe_int(value: Any, default: int = -1) -> int:
    """Convert a value to int without raising."""

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def decode_if_bytes(value: Any) -> Any:
    """Decode bytes from H5 attributes/datasets."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.astype(str)
    return value


def normalize_id(value: Any) -> str:
    """Return a stable string identifier."""

    value = decode_if_bytes(value)
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def validate_identifier(value: Any, *, context: str = "identifier") -> str:
    """Return an identifier that is safe to use in output filenames."""

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{context} must be scalar")
        value = value.reshape(-1)[0]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"{context} must be finite")
    text = normalize_id(value).strip()
    if not text or text in {".", ".."}:
        raise ValueError(f"{context} must be non-empty")
    if "/" in text or "\\" in text or any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError(f"{context} contains a path separator or control character")
    if any(char in text for char in "*?[]"):
        raise ValueError(f"{context} contains filename wildcard characters")
    return text


def path_for_import(project_root: str | Path) -> None:
    """Ensure scripts can import the local package when run from the project root."""

    import sys

    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def write_dataframe(df: Any, path: str | Path) -> None:
    """Write a pandas DataFrame, using parquet when available and CSV as fallback."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except Exception as exc:
            try:
                safe_df = parquet_safe_dataframe(df)
                safe_df.to_parquet(path, index=False)
                get_logger().warning("Wrote parquet with string-normalized object columns %s after: %s", path, exc)
                return
            except Exception as safe_exc:
                get_logger().warning("Could not write parquet %s: %s", path, safe_exc)
                # Avoid leaving a stale parquet artifact from an earlier run
                # beside the current CSV fallback.
                if path.exists():
                    path.unlink()
                csv_path = path.with_suffix(".csv")
                df.to_csv(csv_path, index=False)
                return
    df.to_csv(path, index=False)


def parquet_safe_dataframe(df: Any) -> Any:
    """Return a copy whose mixed object columns are stable for pyarrow parquet."""

    try:
        import pandas as pd
    except ImportError:
        return df
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if series.dtype != object:
            continue
        values = [value for value in series.dropna().head(1000).tolist()]
        if not values:
            continue
        types = {_parquet_type_key(value) for value in values}
        has_nested = any(isinstance(value, (dict, list, tuple, set, np.ndarray)) for value in values)
        if len(types) > 1 or has_nested:
            out[col] = series.map(_string_for_parquet_object)
    return out


def _parquet_type_key(value: Any) -> type:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return str
    return type(value)


def _string_for_parquet_object(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    value = decode_if_bytes(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        return json_dumps_safe(value)
    return str(value)


def angular_separation_arcsec(
    ra1_deg: np.ndarray | float,
    dec1_deg: np.ndarray | float,
    ra2_deg: np.ndarray | float,
    dec2_deg: np.ndarray | float,
) -> np.ndarray | float:
    """Small-angle-safe angular separation in arcsec."""

    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(c) * 3600.0


def infer_pixel_scale_arcsec(
    wcs: Any | None,
    default: float | None = None,
    *,
    strict: bool = False,
) -> float:
    """Infer approximate pixel scale in arcsec from an astropy WCS.

    A default is accepted only when a caller supplies an explicit, validated
    value.  The package itself never supplies a survey-specific fallback.
    """

    if wcs is None:
        if default is not None and np.isfinite(default) and default > 0:
            return float(default)
        raise ValueError("pixel scale cannot be inferred because WCS is missing")
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales_deg = proj_plane_pixel_scales(wcs.celestial)
        scale_arcsec = float(np.mean(np.abs(scales_deg)) * 3600.0)
        if np.isfinite(scale_arcsec) and scale_arcsec > 0:
            return scale_arcsec
    except (AttributeError, ImportError, TypeError, ValueError):
        raise ValueError("unable to infer pixel scale from WCS") from None
    raise ValueError("WCS did not provide a positive pixel scale")


def env_int(name: str, default: int) -> int:
    """Read integer environment variables for small operational switches."""

    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
