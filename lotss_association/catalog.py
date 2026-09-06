"""PyBDSF Gaussian catalog helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table

from .utils import angular_separation_arcsec, first_existing_key, get_logger, safe_float

FIELD_ALIASES = {
    "source_id": ["Source_id", "SOURCE_ID", "source_id", "Source_id_1"],
    "island_id": ["Isl_id", "ISL_ID", "Island_id", "island_id"],
    "gaussian_id": ["Gaussian_id", "GAUSSIAN_ID", "Gaus_id", "gaussian_id"],
    "ra": ["RA", "ra", "RA_deg", "ra_deg"],
    "dec": ["DEC", "Dec", "dec", "DEC_deg", "dec_deg"],
    "e_ra": ["E_RA", "e_RA", "ra_err", "E_RA_deg"],
    "e_dec": ["E_DEC", "e_DEC", "dec_err", "E_DEC_deg"],
    "total_flux": ["Total_flux", "TOTAL_FLUX", "total_flux", "Flux", "flux"],
    "peak_flux": ["Peak_flux", "PEAK_FLUX", "peak_flux"],
    "maj": ["Maj", "MAJ", "maj"],
    "min": ["Min", "MIN", "min"],
    "pa": ["PA", "pa"],
    "dc_maj": ["DC_Maj", "DC_MAJ", "dc_maj"],
    "dc_min": ["DC_Min", "DC_MIN", "dc_min"],
    "dc_pa": ["DC_PA", "dc_pa"],
    "s_code": ["S_Code", "S_CODE", "s_code"],
    "x": ["Xposn", "x", "X", "x_pix", "X_IMAGE"],
    "y": ["Yposn", "y", "Y", "y_pix", "Y_IMAGE"],
}

# PyBDSF writes Gaussian angular sizes in degrees.  The normalized columns
# retain their historical names for compatibility, but their contract is
# explicit: ``_maj``, ``_min``, ``_dc_maj`` and ``_dc_min`` are arcseconds.
ANGULAR_SIZE_FIELDS = ("maj", "min", "dc_maj", "dc_min")


@dataclass
class CatalogColumns:
    """Detected source, position, flux, shape, and optional pixel-coordinate columns."""

    # Normalize supported PyBDSF/LoTSS column aliases to internal names.
    source_id: str | None = None
    island_id: str | None = None
    gaussian_id: str | None = None
    ra: str | None = None
    dec: str | None = None
    e_ra: str | None = None
    e_dec: str | None = None
    total_flux: str | None = None
    peak_flux: str | None = None
    maj: str | None = None
    min: str | None = None
    pa: str | None = None
    dc_maj: str | None = None
    dc_min: str | None = None
    dc_pa: str | None = None
    s_code: str | None = None
    x: str | None = None
    y: str | None = None


def read_gaussian_catalog(path: str | Path) -> Table:
    """Read a PyBDSF Gaussian FITS catalog."""

    path = Path(path)
    table = Table.read(path)
    return table


def detect_catalog_columns(table: Table) -> CatalogColumns:
    """Detect common PyBDSF Gaussian catalog fields."""

    mapping = {name: table[name] for name in table.colnames}
    detected = {}
    for canonical, aliases in FIELD_ALIASES.items():
        detected[canonical] = first_existing_key(mapping, aliases)
    return CatalogColumns(**detected)


def warn_missing_columns(columns: CatalogColumns, required: list[str] | None = None) -> None:
    """Warn about missing catalog columns."""

    required = required or ["total_flux", "peak_flux", "maj", "min"]
    missing = [name for name in required if getattr(columns, name) is None]
    has_sky = columns.ra is not None and columns.dec is not None
    has_pixel = columns.x is not None and columns.y is not None
    if not has_sky and not has_pixel:
        missing.append("ra/dec or x/y")
    if missing:
        warnings.warn(f"Gaussian catalog missing expected columns: {missing}", stacklevel=2)


def print_catalog_summary(path: str | Path, n_rows: int = 5) -> None:
    """Print column names, dtypes, and the first rows of a FITS catalog."""

    if int(n_rows) < 0:
        raise ValueError("n_rows must be >= 0")
    table = read_gaussian_catalog(path)
    columns = detect_catalog_columns(table)
    print(f"Catalog: {path}")
    print(f"Rows: {len(table)}")
    print("Columns:")
    for name in table.colnames:
        print(f"  - {name}: {table[name].dtype}")
    print("\nDetected aliases:")
    for field_name in columns.__dataclass_fields__:
        print(f"  {field_name}: {getattr(columns, field_name)}")
    print(f"\nFirst {min(n_rows, len(table))} rows:")
    if len(table) > 0:
        print(table[:n_rows])


def table_to_dataframe(table: Table) -> pd.DataFrame:
    """Convert an astropy Table to pandas with byte strings decoded."""

    df = table.to_pandas()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda value: value.decode("utf-8", errors="replace")
                if isinstance(value, (bytes, np.bytes_))
                else value
            )
    return df


def _angular_size_arcsec(table: Table, column_name: str) -> np.ndarray:
    """Return one catalogue size column in arcseconds.

    Standard PyBDSF tables express ``Maj``/``Min`` and their deconvolved
    counterparts in degrees.  If an Astropy unit is present we honor it and
    convert through Astropy; a unit-less standard PyBDSF column is interpreted
    as degrees.  Coordinates and position angles are intentionally handled
    separately and are not affected by this conversion.
    """

    column = table[column_name]
    unit = getattr(column, "unit", None)
    try:
        quantity = column.quantity if unit is not None else np.asarray(column, dtype=float) * u.deg
        values = quantity.to_value(u.arcsec)
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError(
            f"catalogue angular-size column {column_name!r} must be an angle convertible to arcsec"
        ) from exc
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values, np.nan)
    return np.asarray(values, dtype=float)


def normalized_gaussian_dataframe(table: Table) -> tuple[pd.DataFrame, CatalogColumns]:
    """Return a DataFrame with canonical helper columns added where possible."""

    # Preserve input columns while adding normalized underscore-prefixed fields.
    columns = detect_catalog_columns(table)
    warn_missing_columns(columns)
    df = table_to_dataframe(table)
    n = len(df)

    def copy_or_default(canonical: str, default: Any) -> None:
        original = getattr(columns, canonical)
        out_name = f"_{canonical}"
        if original is not None and original in df:
            df[out_name] = df[original]
        else:
            df[out_name] = default() if callable(default) else default

    copy_or_default("source_id", lambda: np.arange(n))
    copy_or_default("island_id", lambda: np.full(n, -1))
    copy_or_default("gaussian_id", lambda: np.arange(n))
    copy_or_default("ra", lambda: np.full(n, np.nan))
    copy_or_default("dec", lambda: np.full(n, np.nan))
    copy_or_default("total_flux", lambda: np.full(n, np.nan))
    copy_or_default("peak_flux", lambda: np.full(n, np.nan))
    for canonical in ANGULAR_SIZE_FIELDS:
        original = getattr(columns, canonical)
        out_name = f"_{canonical}"
        if original is not None and original in table.colnames:
            df[out_name] = _angular_size_arcsec(table, original)
        else:
            df[out_name] = np.full(n, np.nan)
    copy_or_default("pa", lambda: np.full(n, np.nan))
    copy_or_default("dc_pa", lambda: np.full(n, np.nan))
    copy_or_default("s_code", lambda: np.full(n, ""))
    copy_or_default("x", lambda: np.full(n, np.nan))
    copy_or_default("y", lambda: np.full(n, np.nan))

    for numeric in [
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
        "_x",
        "_y",
    ]:
        df[numeric] = pd.to_numeric(df[numeric], errors="coerce")

    return df, columns


def validate_gaussian_coordinates(
    frame: pd.DataFrame,
    *,
    context: str = "Gaussian catalogue",
    require_pixel: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate row-level sky or explicit pixel coordinates."""

    sky_available = {"_ra", "_dec"}.issubset(frame.columns)
    pixel_available = {"_x", "_y"}.issubset(frame.columns)
    if not sky_available and not pixel_available:
        raise ValueError(f"{context} must provide RA/DEC or pixel x/y columns")

    sky_values = (
        frame[["_ra", "_dec"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if sky_available
        else np.full((len(frame), 2), np.nan)
    )
    pixel_values = (
        frame[["_x", "_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if pixel_available
        else np.full((len(frame), 2), np.nan)
    )
    sky_valid = np.isfinite(sky_values).all(axis=1)
    pixel_valid = np.isfinite(pixel_values).all(axis=1)

    if np.any(sky_valid & ((sky_values[:, 0] < 0.0) | (sky_values[:, 0] > 360.0))):
        raise ValueError(f"{context} contains RA values outside [0, 360] degrees")
    if np.any(sky_valid & ((sky_values[:, 1] < -90.0) | (sky_values[:, 1] > 90.0))):
        raise ValueError(f"{context} contains DEC values outside [-90, 90] degrees")

    invalid = ~pixel_valid if require_pixel else ~(sky_valid | pixel_valid)
    if invalid.any():
        count = int(invalid.sum())
        if "_gaussian_id" in frame:
            references = frame.loc[invalid, "_gaussian_id"].astype(str).tolist()
            reference_label = "Gaussian IDs"
        else:
            references = [str(value) for value in frame.index[invalid].tolist()]
            reference_label = "row indices"
        preview = ", ".join(references[:10])
        suffix = " ..." if len(references) > 10 else ""
        requirement = "finite pixel x/y coordinates" if require_pixel else "finite RA/DEC or pixel x/y coordinates"
        row_label = "row" if count == 1 else "rows"
        raise ValueError(
            f"{context} contains {count} Gaussian {row_label} without {requirement}; "
            f"{reference_label}: {preview}{suffix}"
        )

    return sky_valid, pixel_valid


def validate_normalized_gaussian_dataframe(
    frame: pd.DataFrame,
    *,
    context: str = "Gaussian catalogue",
) -> None:
    """Validate the fields required by the production association path.

    Identifiers are optional because stable row indices are assigned when a
    catalogue omits them. Position, flux, and observed shape fields are
    required. Invalid flux values remain unavailable evidence, while invalid
    shapes and coordinates fail.
    """

    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{context} must be a pandas DataFrame")
    required = {"_total_flux", "_peak_flux", "_maj", "_min"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required normalized column(s): {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{context} is empty")

    if "_gaussian_id" in frame and frame["_gaussian_id"].duplicated(keep=False).any():
        duplicates = sorted({str(value) for value in frame.loc[frame["_gaussian_id"].duplicated(keep=False), "_gaussian_id"]})
        preview = ", ".join(duplicates[:10])
        suffix = " ..." if len(duplicates) > 10 else ""
        raise ValueError(f"{context} contains duplicate Gaussian IDs: {preview}{suffix}")

    validate_gaussian_coordinates(frame, context=context)

    shape = frame[["_maj", "_min"]].apply(pd.to_numeric, errors="coerce")
    shape_values = shape.to_numpy(float)
    if not np.isfinite(shape_values).all(axis=1).all() or (shape_values <= 0).any():
        raise ValueError(f"{context} contains rows without positive observed major/minor axes")


def select_catalog_region(
    df: pd.DataFrame,
    center_ra: float | None,
    center_dec: float | None,
    radius_arcsec: float,
) -> pd.DataFrame:
    """Preselect Gaussian rows near a cutout center."""

    radius = safe_float(radius_arcsec, float("nan"))
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius_arcsec must be finite and non-negative")
    if center_ra is None or center_dec is None:
        return df
    if "_ra" not in df or "_dec" not in df:
        return df
    dist = angular_separation_arcsec(
        center_ra,
        center_dec,
        df["_ra"].to_numpy(float),
        df["_dec"].to_numpy(float),
    )
    keep = np.isfinite(dist) & (dist <= radius)
    return df.loc[keep].copy()


def gaussian_row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a normalized Gaussian row to a compact dict."""

    return {
        "source_id": row.get("_source_id"),
        "island_id": row.get("_island_id"),
        "gaussian_id": row.get("_gaussian_id"),
        "ra": safe_float(row.get("_ra")),
        "dec": safe_float(row.get("_dec")),
        "total_flux": safe_float(row.get("_total_flux")),
        "peak_flux": safe_float(row.get("_peak_flux")),
        "maj": safe_float(row.get("_maj")),
        "min": safe_float(row.get("_min")),
        "pa": safe_float(row.get("_pa")),
        "dc_maj": safe_float(row.get("_dc_maj")),
        "dc_min": safe_float(row.get("_dc_min")),
        "dc_pa": safe_float(row.get("_dc_pa")),
        "s_code": row.get("_s_code"),
        "x": safe_float(row.get("_x")),
        "y": safe_float(row.get("_y")),
    }


def log_catalog_detection(columns: CatalogColumns) -> None:
    """Log detected catalog column mapping."""

    logger = get_logger()
    logger.info("Detected Gaussian catalog columns:")
    for field_name in columns.__dataclass_fields__:
        logger.info("  %s -> %s", field_name, getattr(columns, field_name))
