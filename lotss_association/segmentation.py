"""S/N-map construction and SExtractor-style segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .config import FORMAL_SNR_LEVELS
from .utils import robust_mad_rms

SEGMENTATION_QA_THRESHOLDS = {
    # QA-only warnings; these values never alter masks or association decisions.
    "mask_fraction_warning": 0.10,
    "largest_label_fraction_warning": 0.05,
}


@dataclass
class SegmentationResult:
    """Multi-threshold S/N segmentation products for one cutout."""

    # Each threshold plane stores connected-component labels for pair lookup.
    snr_map: np.ndarray
    thresholds: np.ndarray
    masks: np.ndarray
    labels_by_threshold: np.ndarray
    n_labels: np.ndarray
    n_small_objects_removed: np.ndarray | None = None


def estimate_mean(
    image: np.ndarray,
    mean: np.ndarray | float | None,
    mode: str = "median",
    *,
    strict: bool = False,
) -> np.ndarray | float:
    """Estimate the image mean/background for S/N construction."""

    mode = str(mode).lower()
    if mode not in {"median", "mean", "zero"}:
        raise ValueError("mean mode must be one of: median, mean, zero")
    if mean is not None:
        value = np.asarray(mean)
        if value.ndim > 0 and value.shape != image.shape:
            raise ValueError(f"mean map shape {value.shape} does not match image shape {image.shape}")
        if strict and not np.isfinite(value).all():
            raise ValueError("mean map contains non-finite values")
        return mean
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        if strict:
            raise ValueError("cannot estimate image mean: image has no finite pixels")
        return 0.0
    if mode == "zero":
        return 0.0
    if mode == "mean":
        return float(np.nanmean(finite))
    return float(np.nanmedian(finite))


def estimate_rms(
    image: np.ndarray,
    rms: np.ndarray | float | None,
    mode: str = "mad",
    *,
    strict: bool = False,
) -> np.ndarray | float:
    """Estimate the rms/noise for S/N construction."""

    mode = str(mode).lower()
    if mode not in {"mad", "std"}:
        raise ValueError("rms mode must be one of: mad, std")
    if rms is None and strict:
        raise ValueError("RMS metadata is required in strict mode")
    if rms is not None:
        arr = np.asarray(rms)
        if arr.ndim == 0:
            value = float(arr)
            if np.isfinite(value) and value > 0:
                return value
            if strict:
                raise ValueError("provided scalar rms must be finite and positive")
        if arr.ndim > 0:
            if arr.shape != image.shape:
                raise ValueError(f"rms map shape {arr.shape} does not match image shape {image.shape}")
            clean = arr.astype(float, copy=True)
            invalid = ~np.isfinite(clean) | (clean <= 0)
            # Supplied RMS maps are input metadata; strict mode treats an
            # invalid pixel as an error rather than repairing it.
            if strict and np.any(invalid):
                raise ValueError("rms map contains non-positive or non-finite values")
            fallback = robust_mad_rms(image)
            if strict and (not np.isfinite(fallback) or fallback <= 0):
                raise ValueError("provided rms map contains invalid values and image rms is unavailable")
            clean[invalid] = fallback
            if strict and (not np.isfinite(clean).all() or (clean <= 0).any()):
                raise ValueError("rms map contains non-positive or non-finite values")
            return clean
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        if strict:
            raise ValueError("cannot estimate rms: image has no finite pixels")
        return 1.0
    if mode == "std":
        value = float(np.nanstd(finite))
    else:
        value = robust_mad_rms(image)
    if not np.isfinite(value) or value <= 0:
        value = float(np.nanstd(finite))
    if not np.isfinite(value) or value <= 0:
        if strict:
            raise ValueError("unable to determine a valid positive RMS")
        value = 1.0
    return value


def build_snr_map(
    image: np.ndarray,
    rms: np.ndarray | float | None = None,
    mean: np.ndarray | float | None = None,
    mean_mode: str = "median",
    rms_mode: str = "mad",
    smooth_before_segmentation: bool = True,
    gaussian_smooth_sigma_pix: float = 1.0,
    strict_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray | float, np.ndarray | float]:
    """Build an S/N map from an image and optional mean/rms maps."""

    # The S/N map is shared by connectivity, bridge, ridge, and artifact terms.
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"image must be a 2-D array, got shape {image.shape}")
    if image.size == 0:
        raise ValueError("image must contain at least one pixel")
    if not np.isfinite(image).any() and strict_metadata:
        raise ValueError("image contains no finite pixels")
    if not np.isfinite(gaussian_smooth_sigma_pix) or gaussian_smooth_sigma_pix < 0:
        raise ValueError("gaussian_smooth_sigma_pix must be finite and >= 0")
    mean_value = estimate_mean(image, mean, mode=mean_mode, strict=strict_metadata)
    rms_value = estimate_rms(image, rms, mode=rms_mode, strict=strict_metadata)

    work = image
    if smooth_before_segmentation and gaussian_smooth_sigma_pix > 0:
        filled = np.array(work, copy=True)
        replacement = np.nanmedian(filled[np.isfinite(filled)]) if np.isfinite(filled).any() else 0.0
        filled[~np.isfinite(filled)] = replacement
        work = ndi.gaussian_filter(filled, sigma=gaussian_smooth_sigma_pix)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (work - mean_value) / rms_value
    snr = np.asarray(snr, dtype=np.float32)
    if strict_metadata and not np.isfinite(snr).any():
        raise ValueError("S/N map contains no finite pixels")
    snr[~np.isfinite(snr)] = 0.0
    return snr, mean_value, rms_value


def _remove_small(mask: np.ndarray, min_area: int, structure: np.ndarray) -> tuple[np.ndarray, int]:
    if min_area <= 1:
        return mask, 0
    labels, n_labels = ndi.label(mask, structure=structure)
    if n_labels == 0:
        return mask, 0
    counts = np.bincount(labels.ravel())
    keep = counts >= min_area
    keep[0] = False
    removed = int(np.count_nonzero((counts[1:] > 0) & (counts[1:] < min_area)))
    return keep[labels], removed


def segment_snr_map(
    snr_map: np.ndarray,
    thresholds: list[float],
    min_mask_area_pix: int | None = None,
    connectivity: int = 2,
    binary_opening: bool = False,
    binary_closing: bool = True,
) -> SegmentationResult:
    """Segment an S/N map at multiple thresholds."""

    # High thresholds trace peaks; lower thresholds trace extended structure.
    snr_map = np.asarray(snr_map, dtype=float)
    thresholds_arr = np.asarray(thresholds, dtype=float)
    if thresholds_arr.ndim != 1 or thresholds_arr.size == 0:
        raise ValueError("thresholds must contain at least one value")
    if not np.isfinite(thresholds_arr).all() or np.any(thresholds_arr <= 0):
        raise ValueError("thresholds must be finite and positive")
    if np.unique(thresholds_arr).size != thresholds_arr.size:
        raise ValueError("thresholds must be unique")
    if snr_map.ndim != 2 or snr_map.size == 0:
        raise ValueError("snr_map must be a non-empty 2-D array")
    if int(connectivity) not in {1, 2}:
        raise ValueError("connectivity must be 1 or 2")
    if min_mask_area_pix is None:
        raise ValueError("min_mask_area_pix must be provided by the resolved configuration")
    min_area = float(min_mask_area_pix)
    if not np.isfinite(min_area) or min_area < 1 or int(min_area) != min_area:
        raise ValueError("min_mask_area_pix must be a positive integer")
    min_mask_area_pix = int(min_area)
    structure = ndi.generate_binary_structure(2, 2 if connectivity == 2 else 1)

    masks = []
    labels_all = []
    n_labels = []
    n_small_objects_removed = []

    for threshold in thresholds_arr:
        mask = snr_map > threshold
        mask &= np.isfinite(snr_map)
        if binary_opening:
            mask = ndi.binary_opening(mask, structure=structure)
        if binary_closing:
            mask = ndi.binary_closing(mask, structure=structure)
        mask, removed = _remove_small(mask, min_mask_area_pix, structure=structure)
        labels, count = ndi.label(mask, structure=structure)
        masks.append(mask.astype(np.uint8))
        labels_all.append(labels.astype(np.int32))
        n_labels.append(count)
        n_small_objects_removed.append(removed)

    return SegmentationResult(
        snr_map=snr_map.astype(np.float32),
        thresholds=thresholds_arr.astype(np.float32),
        masks=np.stack(masks, axis=0),
        labels_by_threshold=np.stack(labels_all, axis=0),
        n_labels=np.asarray(n_labels, dtype=np.int32),
        n_small_objects_removed=np.asarray(n_small_objects_removed, dtype=np.int32),
    )


def save_segmentation(path: str | Path, result: SegmentationResult) -> None:
    """Save segmentation products in compressed NPZ format."""

    _validate_segmentation_result(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        snr_map=result.snr_map,
        thresholds=result.thresholds,
        masks=result.masks,
        labels_by_threshold=result.labels_by_threshold,
        n_labels=result.n_labels,
        n_small_objects_removed=(
            result.n_small_objects_removed
            if result.n_small_objects_removed is not None
            else np.zeros_like(result.n_labels)
        ),
    )


def load_segmentation(path: str | Path) -> SegmentationResult:
    """Load a saved segmentation NPZ."""

    with np.load(path, allow_pickle=False) as data:
        required = {"snr_map", "thresholds", "masks", "labels_by_threshold", "n_labels"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"segmentation file is missing required array(s): {', '.join(missing)}")
        result = SegmentationResult(
            snr_map=data["snr_map"],
            thresholds=data["thresholds"],
            masks=data["masks"],
            labels_by_threshold=data["labels_by_threshold"],
            n_labels=data["n_labels"],
            n_small_objects_removed=data["n_small_objects_removed"]
            if "n_small_objects_removed" in data.files
            else np.zeros_like(data["n_labels"]),
        )
    _validate_segmentation_result(result)
    return result


def _validate_segmentation_result(result: SegmentationResult) -> None:
    """Validate saved/in-memory segmentation array dimensions."""

    snr = np.asarray(result.snr_map)
    thresholds = np.asarray(result.thresholds)
    masks = np.asarray(result.masks)
    labels = np.asarray(result.labels_by_threshold)
    n_labels = np.asarray(result.n_labels)
    if snr.ndim != 2 or snr.size == 0:
        raise ValueError("segmentation snr_map must be a non-empty 2-D array")
    if thresholds.ndim != 1 or thresholds.size == 0 or not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError("segmentation thresholds must be a non-empty finite 1-D array of positive values")
    if np.unique(thresholds).size != thresholds.size:
        raise ValueError("segmentation thresholds must be unique")
    expected = (thresholds.size, *snr.shape)
    if masks.shape != expected or labels.shape != expected:
        raise ValueError(f"segmentation masks/labels must have shape {expected}")
    if n_labels.shape != (thresholds.size,):
        raise ValueError("segmentation n_labels must have one value per threshold")


def segmentation_diagnostics(cutout_id: str, result: SegmentationResult) -> list[dict[str, Any]]:
    """Summarize segmentation masks for diagnostics."""

    records: list[dict[str, Any]] = []
    height, width = result.snr_map.shape
    image_area = float(height * width)
    removed = (
        result.n_small_objects_removed
        if result.n_small_objects_removed is not None
        else np.zeros_like(result.n_labels)
    )
    for idx, threshold in enumerate(result.thresholds):
        labels = result.labels_by_threshold[idx]
        counts = np.bincount(labels.ravel())
        largest = int(counts[1:].max()) if len(counts) > 1 else 0
        total = int(result.masks[idx].sum())
        fraction = float(total / image_area) if image_area > 0 else 0.0
        warning = ""
        if fraction > SEGMENTATION_QA_THRESHOLDS["mask_fraction_warning"]:
            warning = "mask_fraction_gt_0.1"
        if largest / image_area > SEGMENTATION_QA_THRESHOLDS["largest_label_fraction_warning"]:
            warning = f"{warning};largest_label_gt_0.05".strip(";")
        records.append(
            {
                "cutout_id": cutout_id,
                "threshold": float(threshold),
                "n_labels": int(result.n_labels[idx]),
                "largest_label_area": largest,
                "total_mask_area": total,
                "mask_fraction": fraction,
                "n_small_objects_removed": int(removed[idx]),
                "warning": warning,
            }
        )
    return records


def labels_at_points(
    labels_by_threshold: np.ndarray,
    thresholds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return component labels at pixel positions for every threshold."""

    labels_by_threshold = np.asarray(labels_by_threshold)
    height, width = labels_by_threshold.shape[-2:]
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    valid = np.isfinite(x) & np.isfinite(y) & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    output: dict[str, np.ndarray] = {}
    for idx, threshold in enumerate(thresholds):
        labels = np.zeros(len(xi), dtype=np.int32)
        labels[valid] = labels_by_threshold[idx, yi[valid], xi[valid]]
        key = threshold_key(float(threshold))
        output[f"label_at_{key}"] = labels
    return output


def threshold_key(threshold: float) -> str:
    """Return a stable column-name suffix for a threshold."""

    text = f"{threshold:g}".replace(".", "p").replace("-", "m")
    return f"{text}sigma"


def connected_at_threshold(
    row_i: Any,
    row_j: Any,
    threshold: float,
) -> bool:
    """Check if two component rows share a non-zero label at a threshold."""

    col = f"label_at_{threshold_key(threshold)}"
    if col not in row_i or col not in row_j:
        return False
    left = int(row_i[col])
    right = int(row_j[col])
    return left > 0 and left == right


def component_support_mask(
    labels_by_threshold: np.ndarray,
    thresholds: np.ndarray,
    component_rows: Any,
    threshold: float = FORMAL_SNR_LEVELS["weak"],
) -> np.ndarray:
    """Build a union mask from the threshold labels touched by components."""

    idx = int(np.argmin(np.abs(np.asarray(thresholds, dtype=float) - threshold)))
    col = f"label_at_{threshold_key(float(thresholds[idx]))}"
    labels = set(int(value) for value in component_rows.get(col, []) if int(value) > 0)
    if not labels:
        return np.zeros(labels_by_threshold.shape[-2:], dtype=bool)
    label_map = labels_by_threshold[idx]
    return np.isin(label_map, list(labels))
