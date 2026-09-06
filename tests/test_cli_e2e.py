from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy.table import Table

from lotss_association import cli
from lotss_association.utils import load_yaml


def test_cli_runs_tiny_h5_and_fits_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Exercise the installed-equivalent entry point through final catalogues."""

    h5_path = tmp_path / "tiny_cutouts.h5"
    image = np.zeros((1, 40, 80), dtype=float)
    image[0, 18:23, 8:68] = 4.0
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("image", data=image)
        handle.attrs["rms"] = 1.0
        handle.attrs["pixel_scale_arcsec"] = 1.0
        handle.attrs["beam_major_arcsec"] = 6.0
        handle.attrs["beam_minor_arcsec"] = 6.0
        handle.attrs["beam_pa_deg"] = 0.0

    catalog_path = tmp_path / "tiny_gaussians.fits"
    Table(
        {
            "Gaussian_id": ["g0", "g1", "g2"],
            "Isl_id": [1, 1, 1],
            "Total_flux": [10.0, 9.0, 8.0],
            "Peak_flux": [8.0, 7.0, 6.0],
            "Maj": [6.0 / 3600.0, 6.0 / 3600.0, 6.0 / 3600.0],
            "Min": [6.0 / 3600.0, 6.0 / 3600.0, 6.0 / 3600.0],
            "PA": [0.0, 0.0, 0.0],
            "RA": [1.0, 1.0, 1.0],
            "DEC": [1.0, 1.0, 1.0],
            "x": [10.0, 35.0, 60.0],
            "y": [20.0, 20.0, 20.0],
        }
    ).write(catalog_path, overwrite=True)

    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lotss-association",
            "--h5-path",
            str(h5_path),
            "--gaus-catalog",
            str(catalog_path),
            "--config",
            "package://data/default.yaml",
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
        ],
    )
    cli.main()

    groups = pd.read_csv(output_dir / "catalogs" / "radio_association_groups.csv")
    edges = pd.read_csv(output_dir / "catalogs" / "radio_association_edges.csv")
    effective = load_yaml(output_dir / "run_metadata" / "effective_config.yaml")
    assert len(groups) == 1
    assert set(groups.loc[0, "gaussian_ids"].split(",")) == {"g0", "g1", "g2"}
    assert len(edges) == 3
    assert int((edges["edge_type"] == "strong").sum()) == 3
    assert effective["runtime"]["effective_metadata"]["software_version"]
