# Parent-linking Large-Scale Parent Candidate Association

Parent-linking keeps the target as radio component association. It does not train a model
and does not solve large-scale misses by increasing the weight of 2 sigma
connectivity.

## Scope

local remains the main catalog:

```text
Gaussian components -> local association group
```

Parent-linking is only a supplemental candidate stage for large separated radio-lobe
systems:

```text
local groups -> large-scale parent association candidates
```

The output candidates do not replace `radio_association_groups.csv` and do not
rewrite the local groups. Parent candidate scoring never directly connects
individual Gaussian components.

## Why Candidate-Only

local already handles compact multi-Gaussian sources, lobe-internal Gaussians, and
continuous ridges or tails. The missing case is a large source whose separated
lobes have no reliable 2.5 sigma or 3 sigma bridge. Using stronger 2 sigma
connectivity would also connect unrelated emission in crowded or diffuse
regions, so Parent-linking keeps local association conservative and handles only
large-scale separated candidates.

## Local Sanity

`lotss_association/local_sanity.py` runs in the production Stage 1 path. It checks
whether each local group looks like a continuous local radio structure and can
split a high-risk over-merge when the internal edge evidence is clear. Important fields
include:

- number of Gaussians and LAS in beams;
- weak and only-2sigma edge fractions;
- ridge gap fraction;
- saddle-to-peak ratio;
- multi-peak separation;
- weak-chain and large-mask flags;
- local overmerge risk.

Suspicious local overmerges can be split when the weak internal edge structure
is clear. If the split is unstable, the original local group is kept and marked
for visual review.

## Parent Candidate Gates

`lotss_association/parent_links.py` evaluates pairs of Stage 1 local groups.
Each endpoint is first classified using the thresholds under
`parent_linking.endpoint_thresholds`. Point-like, compact, isolated, low-S/N,
or artifact-vetoed groups do not enter normal double-lobe scoring. A narrowly
defined near-boundary rescue path can admit an extended fragment only when the
radio-support and geometry requirements under
`parent_linking.rescue_thresholds` and
`parent_linking.support_thresholds` are satisfied.

Candidate generation is bounded by `max_box_gap_beam`,
`max_center_distance_beam`, and the per-group/per-cutout limits. The Stage 2
score uses only the explicitly configured `symmetry_weights` and
`score_weights`.

## Conservative Geometry

Candidate scoring uses large-scale geometry and midpoint evidence:

- axis alignment;
- facing score, defined by the inward endpoint extent toward its counterpart;
- midpoint symmetry, defined by the distance balance around a compact midpoint core
  (neutral when no such core is available);
- flux and size ratio;
- lobe-like local morphology;
- compact/core candidate near the midpoint;
- weak bridge as low-weight support only.

Normal candidates must pass the configured axis-alignment, facing, flux-ratio,
size-ratio, and symmetry gates. Near-boundary rescue uses its own explicit
limits and additionally requires common low-threshold emission, bridge/ridge
support, multiple Gaussians, or sufficient 3-sigma area. Host evidence can
support or downgrade a radio candidate but does not replace the radio geometry
gate.

Host use is controlled by two related settings. `host_support.enabled` controls
whether host queries and host scoring run. When
`host_support.require_host_for_parent_link` is true, a final high-confidence
parent link also requires the configured host requirement. When it is false,
the host stage remains optional: radio evidence may produce a final link when
there is no host and no host contradiction. A disabled host stage must be used
with `require_host_for_parent_link: false`; that combination follows the same
radio-only path.

## Quality Levels

### Seed-stage proposal quality

The score thresholds under `parent_seed_selection.thresholds` rank the
geometry-only pair proposals that Stage 2 computes while building candidate
pairs, using the shared seed-scoring rules in `lotss_association.parent_seed`:

- `high`: score >= 4.0;
- `medium`: score >= 3.2;
- `low`: score >= 2.5, debug only;
- `suspicious`: blocked or conflicting large-scale evidence.

These values rank proposals internally; they are not the final acceptance
thresholds. The `parent_candidate_quality` column published in
`parent_edges_debug.csv` is the final link quality produced by the endpoint,
geometry, radio-evidence, host, and conflict-resolution chain described below.
The programmatic entry point `run_parent_seed` exposes the same proposal
scoring for callers that want the seed stage on its own.

### Final parent-link quality

`run_parent_links` applies endpoint eligibility, geometry, radio evidence, host
requirements, contradiction checks, and conflict resolution. Its
`parent_candidate_quality` column is the final link quality after that decision
chain. Only candidates classified as `accepted_high_confidence_parent` with
final quality `high` are written to `parent_candidates.csv`. Medium, low,
geometry-only, and rejected rows remain in `parent_edges_debug.csv` (and its
Parquet companion).

CatWISE2020 is queried first for host support, with AllWISE as the fallback.
Host evidence can support or downgrade a radio candidate, but it cannot replace
the independent radio-evidence gate.

The values above describe `configs/default.yaml`. The YAML
configuration and the run's `run_metadata/effective_config.yaml` are the
authoritative records for a particular run.

## Outputs

Main parent-candidate outputs:

- `parent_candidates.csv`;
- `parent_edges_debug.csv` and `parent_edges_debug.parquet`;
- `source_morph_table.csv`;
- `host_candidates.csv` and `host_candidates.parquet`;
- `host_query_log.csv`.

Diagnostics:

- `parent_link_diagnostics.csv`;
- `needs_visual_check.csv`.
- `local_sanity_diagnostics.csv`;
- `local_needs_visual_check.csv`.

Figures:

- `figures/overview/`: local groups plus high/medium parent candidates;
- `figures/parent_zoom/`: large-scale parent candidates only.

Every Parent-linking parent candidate should be reviewed by a human before being treated
as a parent radio source association.
