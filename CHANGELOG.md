# Changelog

## 0.1.1 - Unreleased

- Centralized configurable morphology, parent-seed, endpoint, rescue, and graph scoring thresholds.
- Added strict production metadata validation and per-run `effective_config.yaml` provenance.
- Removed duplicate legacy blocks from the release configuration and renamed graph weights to `weights_graph`.
- Added duplicate-YAML-key protection, configuration contract tests, and a Python 3.10–3.12 wheel-import CI check.
- Unified production configuration readers on the strict YAML loader.
- Restored the JSONL host-query cache fallback and added a regression test.
- Cleared the package/test Ruff backlog and modernized package license metadata.
- Reduced the public tree to Gaussian-component association code, production entry points, documentation, and core tests.
- Enforced strict RMS and pixel-scale validation and removed experiment-specific production-script hooks.
- Added explicit rejection of duplicate Gaussian/component identifiers and
  duplicate or self-loop association edges instead of producing ambiguous
  scientific assignments.
- Replaced threshold-specific Stage 2 noise reason labels with stable symbolic
  reasons and recorded the effective thresholds in the morphology output.
- Moved the catalogue-to-cutout preselection margin into the validated
  `matching.preselect_margin_arcsec` configuration namespace.
- Packaged the default YAML configurations for installed CLI use and changed
  default output paths to the caller's working directory.
- Added row-wise mixed sky/pixel matching, strict integer component-index
  validation, and a canonical parent-conflict tie-break.
- Unified the programmatic Stage 1 pair-distance default with the packaged
  YAML method, improved installed-CLI documentation, and added a tiny
  H5/FITS end-to-end smoke test.
- Made source-revision provenance accept a validated build-environment commit
  for wheel-based runs, and tightened release hygiene checks.
- Unified the release configuration: `configs/real_lotss_conservative.yaml`
  was byte-identical to `configs/default.yaml`, so the repository now ships a
  single release configuration and all entry points and documentation
  reference it.
- Stage 2 now consumes `host_support.min_host_quality_for_default_candidate`
  for midpoint host support, and the unreachable `run_host_support` entry
  point was removed.
- `--association-mode` now fails fast when `association.enabled` or
  `local_association.enabled` is false instead of silently running the legacy
  graph-merge path.
- Reconciled the merged-source support-mask preference with the group
  catalogue (2.5 sigma support is preferred before 2 sigma in both).
- Split local subgroups reproject their own centroid through the cutout WCS
  instead of inheriting the pre-split sky position, and the local-sanity
  tables report the real only-2sigma edge count.
- Parent-support label lookup no longer reads the image corner for a group
  with a non-finite centroid.
- Consolidated duplicated pixel-plane and dataframe helpers into
  `lotss_association._shared` and named the remaining frozen measurement and
  artifact-score gates in `config.py`.
- Removed the unused `beam.normalization` schema key and documented the beam
  axis-sign convention in the README.

- Normalize standard PyBDSF Gaussian and deconvolved sizes from degrees to the
  arcsecond units used by association and morphology calculations.
- Support standard PyBDSF `Xposn`/`Yposn` pixel coordinates and clarify that
  masked image sums retain the input image pixel-value units.
- Recompute Stage 1 group measurements after local overmerge refinement and
  make split-edge decisions canonical in the final edge catalogue.
- Add end-to-end split regression coverage and document local sanity as part
  of the Stage 1 processing path.

## 0.1.0 - 2026-08-22

- Prepared the public `lotss-association` package structure.
- Exposed the reusable Python package as `lotss_association`.
- Documented core scripts and reuse boundaries.
- Added release metadata, citation metadata, CI, and unit tests.
- Kept large survey data, generated catalogues, figures, logs, and local paths
  out of version control.
