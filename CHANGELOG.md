# Changelog

## 0.1.1 - Unreleased

- Normalized standard PyBDSF Gaussian and deconvolved sizes from degrees to
  arcseconds, and supported `Xposn`/`Yposn` pixel-only Gaussian catalogues.
- Added strict production validation for pixel scale, RMS, and restoring-beam
  metadata, and wrote per-run `run_metadata/effective_config.yaml` provenance.
- Consolidated all configurable thresholds into validated YAML namespaces,
  rejected duplicate and unknown YAML keys, and unified Stage 1 pair-distance
  defaults across the Python API, CLI, and packaged configurations.
- Recomputed Stage 1 group measurements after local overmerge refinement, and
  gave split edges one unambiguous final state in the edge catalogue.
- Rejected duplicate Gaussian/component identifiers, self-loop edges, and
  duplicate pair edges instead of producing ambiguous catalogues.
- Made Stage 2 noise reasons stable symbolic labels, applied
  `host_support.min_host_quality_for_default_candidate` to midpoint host
  support, and resolved competing parent links with a canonical tie-break.
- Shipped the packaged default configuration for installed CLI use, defaulted
  output paths to the working directory, and added an H5/FITS end-to-end smoke
  test with Python 3.10-3.12 wheel-import CI coverage.
- Reduced the public tree to the Gaussian-component association pipeline and
  shipped one release configuration, `configs/default.yaml`.

## 0.1.0 - 2026-08-22

- Prepared the public `lotss-association` package structure.
- Exposed the reusable Python package as `lotss_association`.
- Documented core scripts and reuse boundaries.
- Added release metadata, citation metadata, CI, and unit tests.
- Kept large survey data, generated catalogues, figures, logs, and local paths
  out of version control.
