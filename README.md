# LoTSS Association

LoTSS Association is a rule-based Python package for associating PyBDSF
Gaussian components in LoTSS radio images. It groups local radio emission,
measures interpretable association evidence, and proposes parent links for
large double-lobe or extended systems.

The pipeline is designed for reproducible survey processing. It
does not require a machine-learning model: decisions are based on beam-aware
geometry, multi-threshold radio-contour support, ridge and bridge continuity,
artifact penalties, and optional WISE/CatWISE host evidence.

The input graph starts from PyBDSF Gaussian components, not from the PyBDSF
source catalogue. Existing PyBDSF source identifiers can be preserved for
diagnostics, but the association decisions are rebuilt from component geometry,
image support, and configured decision rules in this repository.

## Method Overview

1. Read LoTSS cutouts or FITS-derived cutouts and match PyBDSF Gaussian
   components into each image.
2. Build beam-normalized pairwise evidence between Gaussian components,
   including distance, morphology, multi-threshold contour connectivity,
   bridge/ridge support, and artifact penalties.
3. Form conservative local association groups from strong edges, with weak
   edges used only as limited attachments, then refine possible local
   overmerges before measuring the final Stage 1 groups.
4. Propose parent-scale candidates for large separated systems and record the
   evidence needed for diagnostics and quality control.

## Repository Layout and Entry Points

- `lotss_association/`: reusable package modules for IO, segmentation,
  association, parent-linking, plotting, and utilities.
- `scripts/run_pipeline.py`: single H5 cutout-file entry point.
- `scripts/run_parent_linking.py`: Stage 2 parent-candidate entry point for
  an existing Stage 1 output.
- `lotss_association.association`: authoritative Stage 1 Gaussian-component
  association implementation.
- `lotss_association.graph_merge`: compatibility/diagnostic graph path; its
  clustering is used only when `--no-association-mode` is requested. Stage 1
  association decisions do not use graph-merge scoring.
- `scripts/visualize_results.py`: regenerate diagnostic figures from outputs.
- `configs/default.yaml`: release configuration (strict metadata
  validation enabled).
- `docs/`: method notes and release notes.
- `docs/release_checklist.md`: objective pre-publication verification checklist.
- `tests/`: unit tests for the reusable pipeline components.

The most useful documentation files are:

- `docs/software_package.md`: public package structure, stable modules, and
  reuse boundaries.
- `docs/script_reference.md`: command-line tools grouped by user-facing role.
- `docs/association_strategy.md`: local Gaussian-component association logic.
- `docs/parent_association.md`: parent-linking candidate stage and outputs.
- `docs/algorithm_notes.md`: background on PyBDSF components, segmentation,
  and graph-based grouping.
- `docs/design.md`: input handling, catalogue fields, and visualization
  conventions.

## Installation

Python 3.10–3.12 is supported. Create the environment with one of those
interpreters; Python 3.9 and older are not supported.

```bash
git clone https://github.com/astrocaojie/LoTSS_association.git LoTSS_association
cd LoTSS_association
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a non-editable installation use `python -m pip install .`.  The installed
commands are `lotss-association` (Stage 1), `lotss-parent-link` (Stage 2), and
`lotss-visualize` (diagnostic plots); each accepts `--help`.

`PyBDSF` is optional at runtime. The package consumes an existing PyBDSF
Gaussian catalogue and does not include survey-scale catalogue generation or
data-staging jobs.

After installation, the reusable Python package is imported as:

```python
import lotss_association
```

The files in `scripts/` remain backwards-compatible wrappers and are also
included in the distribution for reproducible legacy invocations.  Installed
entry points are preferred for new runs.

## Required Inputs

For a normal association run, provide:

- Cutouts in H5 format with image data and physical metadata.
- A PyBDSF Gaussian catalogue with sky coordinates (or explicit pixel x/y)
  plus flux and shape columns.
- Optional WISE/CatWISE internet access for host-query support.

Large FITS/H5/catalogue products are intentionally not committed. Store local
data under `data/` or pass paths explicitly on the command line.

For reproducible runs, keep the command line, YAML configuration, image
manifest, and Gaussian-catalogue manifest next to the output catalogues. The
recommended configuration enables `runtime.strict_metadata`: invalid or
missing pixel-scale or RMS information, or invalid configured restoring-beam
axes, aborts a scientific run instead of silently applying LoTSS defaults.
When cutout beam metadata are present, both axes are required and must agree
with the configured beam. Production wrappers also write
`run_metadata/effective_config.yaml`.

Standard PyBDSF Gaussian tables store `Maj`, `Min`, `DC_Maj`, and `DC_Min` in
degrees.  The catalogue reader converts these four fields once to normalized
`_maj`, `_min`, `_dc_maj`, and `_dc_min` columns in arcseconds; all downstream
beam and morphology calculations use those arcsecond values.  Right ascension
and declination remain degrees, and position angles retain their catalogue
definition.  Pixel-only catalogues may use the standard PyBDSF `Xposn` and
`Yposn` columns (or the documented x/y aliases) when RA/DEC are unavailable.

### Restoring-beam and position-angle convention

`beam.major_arcsec`, `beam.minor_arcsec`, and `beam.pa_deg` define the
elliptical restoring beam used to normalize all distances. `beam.pa_deg` is
the sky position angle of the beam major axis in degrees east of north. The
configured beam must agree with the per-cutout beam metadata when the cutouts
provide one.

Sky position angles are converted to image-plane angles as
`pixel_angle = atan2(dec_axis_sign * cos(PA), ra_axis_sign * sin(PA))`. The
defaults `ra_axis_sign: 1.0` and `dec_axis_sign: 1.0` describe cutouts whose
pixel x-axis points toward increasing RA (east) and whose pixel y-axis points
toward increasing Dec (north). For standard FITS-orientation images, where RA
decreases with increasing pixel x, set `beam.ra_axis_sign: -1.0`. To decide
the sign, check whether RA increases or decreases along the pixel x-axis in
your cutout header (`CDELT1 > 0` means `ra_axis_sign: 1.0`, `CDELT1 < 0` means
`ra_axis_sign: -1.0`). Component-to-component PA alignment is insensitive to
a consistent global sign, but comparisons between catalogue PAs and
pixel-plane directions (the line-to-PA alignment and beam-axis geometry
terms) require the correct sign. `beam.pixel_pa_deg` overrides the sky-PA
conversion with a direct image-plane angle when the local WCS parity is known
explicitly.

## Quick Run

```bash
lotss-association \
  --h5-path data/example/lotss_cutouts.h5 \
  --gaus-catalog data/example/pybdsf_gaussians.fits \
  --config package://data/default.yaml \
  --output-dir outputs/example_association \
  --limit 20 \
  --make-figures \
  --overwrite \
  --association-mode
```

When running directly from a source checkout, the equivalent command is
`python scripts/run_pipeline.py` with the same options.  The `scripts/`
modules are compatibility wrappers; installed console commands are the
recommended interface for new runs.

## Minimal API example

The spatial preselection can be exercised without survey data:

```python
import pandas as pd
from lotss_association.association import candidate_pairs
from lotss_association.utils import load_yaml, packaged_config_uri, validate_config

config = validate_config(load_yaml(packaged_config_uri("default.yaml")))
components = pd.DataFrame([
    {"component_index": 0, "x": 10.0, "y": 10.0, "pixel_scale_arcsec": 1.0},
    {"component_index": 1, "x": 12.0, "y": 10.0, "pixel_scale_arcsec": 1.0},
])
print(candidate_pairs(components, config))
```

This prints `[(0, 1)]` for the supplied configuration and demonstrates the
deterministic Stage 1 candidate contract.  A complete association run also
requires the image/SNR map and Gaussian flux/shape fields described below.

Main outputs are written under `outputs/example_association/catalogs/`:

- `radio_association_groups.csv` and `.parquet`
- `radio_association_edges.csv` and `.parquet`
- `radio_association_components.csv` and `.parquet`
- `lotss_association_merged_sources.csv` and `.parquet`

`radio_association_edges` is the main diagnostic table: it records the positive
and negative evidence for each tested component pair. `radio_association_groups`
is the recommended catalogue for science use after validation and visual
quality control.

## Parent-Linking

Parent-linking is a candidate stage for large separated radio systems. It does
not rewrite the local Gaussian groups. The main outputs are:

- `parent_candidates.csv`
- `parent_edges_debug.csv` and `.parquet`
- `source_morph_table.csv`
- `host_candidates.csv` and `.parquet`
- `host_query_log.csv`
- `parent_link_diagnostics.csv`
- `needs_visual_check.csv`

WISE/CatWISE host matches are recorded as supporting diagnostics. They should
be interpreted together with the radio morphology, bridge/ridge evidence,
artifact flags, and visual review products.

## Testing

```bash
pytest
```

The tests avoid large survey data and cover the core scoring, clustering,
metadata, host-cache, and configuration behavior.

For a release check, also run:

```bash
ruff check lotss_association scripts tests
python -m compileall -q lotss_association scripts tests
python -m build
python -m pip install --force-reinstall dist/*.whl
cd /tmp
python -c "import lotss_association; print(lotss_association.__version__)"
```

The final import is intentionally run outside the source tree so package
discovery errors cannot be hidden by `pytest`'s repository path. When a release
environment has no network access, build and installation should be performed
from a pre-populated wheelhouse.

## AI-assisted Development

[Anthropic Claude Code](https://claude.com/claude-code) and
[OpenAI Codex](https://openai.com/codex/) were used as supporting tools during
the development and revision of this repository. Their assistance included
code review, debugging, refactoring suggestions, test development,
visualization and plotting scripts, and software documentation.

All AI-assisted changes were reviewed, edited where necessary, and tested by
the authors. The scientific methodology, association criteria, parameter
choices, validation procedures, interpretation of results, and final software
decisions were determined and verified by the authors, who assume full
responsibility for the contents of this repository.

## Citation

If you use this code, cite the repository and the associated paper or data
release. `CITATION.cff` contains the repository metadata and should be updated
with the final author list, DOI, and paper title before publication.

## License

This package is released under the MIT license.
