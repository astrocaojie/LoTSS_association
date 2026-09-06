# Script Reference

The scripts are grouped by use case. After `python -m pip install .`, prefer
the installed commands below; direct `scripts/*.py` invocation remains
available for backwards compatibility.

## Core User Commands

- `lotss-association` (or `scripts/run_pipeline.py`): run local Gaussian association for one H5 cutout
  file and one PyBDSF Gaussian catalogue.
- `lotss-parent-link` (or `scripts/run_parent_linking.py`): add large-scale parent-link candidates to
  an existing local-association output directory.
- `lotss-visualize` (or `scripts/visualize_results.py`): regenerate overview and zoom figures from
  existing output catalogues.

The installed commands use the packaged default YAML files. Their default
output directories are relative to the current working directory; pass
`--config` and `--output-dir` explicitly for a fully pinned production run.

## Inspection and Small Utilities

- `scripts/inspect_h5.py`: print H5 groups, datasets, shapes, and detected
  cutout keys.
- `scripts/print_gaus_catalog_columns.py`: inspect Gaussian catalogue columns
  and detected aliases.
- `scripts/build_segmentation_maps.py`: build reusable S/N segmentation maps.
- `scripts/match_gaussians_to_cutouts.py`: test catalogue-to-cutout matching.
- `scripts/build_component_graph.py`: build a diagnostic component graph for a
  small sample.
- `scripts/export_extended_sources.py`: lightweight compatibility wrapper for
  running the pipeline.

The public script set covers configuration inspection, association
production, and diagnostic visualization.
