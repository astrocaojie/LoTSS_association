#!/usr/bin/env python
"""Build segmentation maps for cutouts in an H5 file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lotss_association.io import H5CutoutReader
from lotss_association.segmentation import build_snr_map, save_segmentation, segment_snr_map
from lotss_association.utils import (
    ensure_dir,
    load_yaml,
    packaged_config_uri,
    segmentation_config_kwargs,
    setup_logging,
    snr_map_config_kwargs,
    validate_config,
    validate_cutout_physical_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--config", default=packaged_config_uri("default.yaml"))
    parser.add_argument("--output-dir", default=str(Path.cwd() / "outputs"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(debug=args.debug)
    config = validate_config(load_yaml(args.config))
    reader = H5CutoutReader(args.h5_path, config_h5=config.get("h5", {}))
    out_dir = ensure_dir(Path(args.output_dir) / "segmentation")
    indices = reader.iter_indices(args.start_index, args.end_index, args.limit)
    if not indices:
        raise SystemExit("No cutouts selected; check --start-index/--end-index/--limit and the H5 input")
    for index in indices:
        cutout = reader.read(index)
        validate_cutout_physical_metadata(cutout, config)
        snr, _, _ = build_snr_map(
            cutout.image,
            rms=cutout.rms,
            mean=cutout.mean,
            **snr_map_config_kwargs(config),
        )
        seg = segment_snr_map(snr, **segmentation_config_kwargs(config))
        path = out_dir / f"{cutout.cutout_id}_seg.npz"
        save_segmentation(path, seg)
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
