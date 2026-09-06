# Software Package Overview

This repository is organized as a reusable source-association package plus a
small set of input, association, parent-candidate, and visualization scripts.
The reusable Python package is `lotss_association`.

## Public Package Layers

The package is split into four practical layers:

1. **Input and normalization**
   - `lotss_association.io`: inspect H5 files and read radio cutouts.
   - `lotss_association.catalog`: read and normalize PyBDSF Gaussian catalogues.
   - `lotss_association.matching`: match catalogue components into each cutout.
2. **Local Gaussian association**
   - `lotss_association.segmentation`: build S/N maps and multi-threshold
     connected-component labels.
   - `lotss_association.beam`: compute beam-aware distances and projected
     beam widths.
   - `lotss_association.morphology`: classify compact, resolved, and lobe-like
     Gaussian components.
   - `lotss_association.association`: score candidate Gaussian pairs and form
     conservative local association groups.
   - `lotss_association.local_sanity`: refine possible local overmerges before
     the final Stage 1 catalogues are measured.
3. **Parent-candidate stage**
   - `lotss_association.parent_seed`: identify local groups that can serve as
     large-scale parent endpoints.
   - `lotss_association.parent_links`: generate and score parent-link
     candidates for large separated systems.
   - `lotss_association.host_query` and `lotss_association.host_support`: query
     and score WISE/CatWISE host evidence as diagnostics.
4. **Diagnostics and reporting**
   - `lotss_association.visualize`: overview and zoom diagnostic plots.
   - `lotss_association.graph_merge`: compatibility/diagnostic graph path for
     callers that explicitly select `--no-association-mode`; it is not an
     implicit dependency of the authoritative Stage 1 association path.
   - `lotss_association.measurements`: measure source-level catalogue fields.

## Stable and Excluded Areas

The recommended public path is:

```text
catalog/io -> segmentation -> association -> local_sanity -> parent_links -> visualization
```

The public tree focuses on the association pipeline itself; survey-scale
data staging and job orchestration are out of scope for this repository.
After installation, users can start with `lotss-association` and
`lotss-parent-link`; the `scripts/*.py` forms remain backwards-compatible
wrappers.

The installed commands read their default YAML configurations from package
resources, so they do not depend on the source checkout. When no output
directory is supplied, output is created below the caller's current working
directory rather than inside `site-packages`.

## Reproducibility and metadata contract

The release configuration in `configs/default.yaml` enables
`runtime.strict_metadata`. Production runs therefore require a positive pixel
scale (from the cutout metadata/WCS or an explicit input column), valid image
noise, and a positive configured restoring beam. Missing values fail loudly
instead of silently assuming LoTSS defaults. Example and unit-test configs may
leave strict mode disabled for backwards compatibility.

Every production wrapper writes
`run_metadata/effective_config.yaml`, containing the exact configuration used
for the run and its resolved metadata. The graph-merge weights live under the
canonical `weights_graph` key; Stage-1 association weights remain under
`weights_association`.

The recorded `git_commit` is taken from the build environment when
`LOTSS_ASSOCIATION_SOURCE_COMMIT` (or another standard CI source-revision
variable) contains a valid commit identifier, and otherwise from the local
Git checkout.  A wheel used outside a checkout therefore records
`unknown` when no source revision was supplied rather than inventing one;
`software_version` remains available in every run.

### Threshold governance

Values that define an input contract or a survey operating point are exposed
in the YAML files and recorded in the effective configuration. Formula
coefficients and fixed decision gates are named in `config.py` and are tied to
the recorded software version and source revision; changing them constitutes a
method revision. Segmentation mask-fraction limits are diagnostic QA warnings
only and do not alter association decisions.

## Reuse Boundaries

The core scoring functions are survey-aware but not hard-coded to a single
local filesystem. Reuse on another radio survey normally requires:

- a Gaussian/component catalogue with sky positions, fluxes, and size columns;
- image cutouts or FITS tiles with a reliable beam and WCS;
- a YAML configuration tuned to the survey resolution, sensitivity, and
  artefact environment;
- a validation reference or review protocol for calibrating thresholds.

Large survey products, generated catalogues, figures, and logs are kept out
of version control (see `.gitignore`).
