# Beam-Aware Radio Component Association Strategy

## Purpose

This stage performs PyBDSF Gaussian component association.

A PyBDSF Gaussian component is treated as a local mathematical description of
radio emission, not as a final physical source by itself. The pipeline builds
pairwise evidence between Gaussian components, then groups associated
components into radio structure groups.

Core objects:

- `Gaussian component`: a PyBDSF-fitted local emission part
- `edge`: evidence that two Gaussian components may belong to the same radio
  emission structure
- `association group`: a set of related Gaussian components

No statistical classifier is trained in this stage; the output is produced by
explicit evidence rules and diagnostic thresholds.

## Why SExtractor Cannot Be Copied Directly

SExtractor was primarily designed for optical and infrared imaging, where
thresholded connected pixels and deblending trees are often a natural first
description of sources.

Radio interferometric images are different:

- the image is convolved by the synthesized beam
- nearby pixels have correlated noise
- sidelobes, negative bowls, and calibration artifacts can mimic diffuse
  structure
- a low-threshold connected mask can join unrelated emission through correlated
  noise

Therefore, 2 sigma connectivity cannot be interpreted as physical connectivity.
In this pipeline, 2 sigma connected labels are weak flags only. They are not a
strong association decision.

## Why Segmentation Still Helps

Multi-threshold segmentation remains useful as morphology evidence:

- it can reveal low-surface-brightness emission around multiple components
- it can support possible bridge or diffuse emission
- 3 sigma and 2.5 sigma connectivity are more credible than 2 sigma
- it provides masks for diagnostic measurements and visual review

The rule is conservative: segmentation can support an association, but low-S/N
connectivity is not used as a sufficient condition by itself.

## Evidence System

The pair score combines positive association evidence and artifact penalties.

Positive evidence:

- beam-aware distance
- Gaussian ellipse overlap or small beam-normalized gap
- component PA alignment
- line-to-PA alignment
- 3 sigma connected segmentation
- 2.5 sigma connected segmentation
- beam-width bridge support
- residual bridge support, measured after subtracting approximate endpoint
  beam models
- ridge continuity along the component-to-component path
- flux continuity
- flow alignment

Weak or diagnostic evidence:

- 2 sigma connectivity
- only-2sigma connected labels

Negative evidence:

- deep S/N valley between components
- only-2sigma penalty
- negative bowl penalty
- sidelobe or artifact risk
- excessive beam-normalized distance
- large low-threshold mask swallow risk

The release config sets:

```yaml
weights_association:
  conn_2sigma: 0.0
```

This means 2 sigma connectivity does not add positive score by default.

## Association Score

The score is:

```text
association_score =
    w_closeness       * closeness_score
  + w_overlap         * ellipse_overlap_score
  + w_pa_alignment    * pa_alignment_score
  + w_conn_3sigma     * connected_at_3sigma
  + w_conn_2p5sigma   * connected_at_2p5sigma
  + w_conn_2sigma     * connected_at_2sigma
  + w_bridge          * bridge_score
  + 0.8 * w_bridge    * residual_bridge_score
  + w_ridge           * ridge_continuity_score
  + w_flux_continuity * flux_continuity_score
  + w_flow_alignment  * flow_alignment_score
  - w_valley          * deep_valley_penalty
  - w_only_2sigma     * only_2sigma_penalty
  - w_negative_bowl   * negative_bowl_penalty
  - w_sidelobe        * sidelobe_risk_penalty
  - w_too_far         * too_far_penalty
  - w_large_mask      * large_mask_swallow_penalty
```

If an edge is connected only at 2 sigma and lacks independent 2.5 sigma, 3
sigma, bridge, residual-bridge, ridge, or overlap support, its score is capped by
`association.max_only_2sigma_score`. The residual-bridge term reuses the bridge
weight scaled by a fixed factor of 0.8.

Long-distance associations must have multiple supporting signals such as bridge,
ridge, and alignment. Negative bowls and sidelobe risk lower the group quality.

## Graph Strategy

Edges are classified as:

- `strong`: `association_score >= threshold_strong`
- `weak`: `threshold_weak <= association_score < threshold_strong`
- `rejected`: below threshold or blocked by rejection logic

The clustering policy is:

1. Build connected components from strong edges only.
2. Treat those as core association groups.
3. Use weak edges only as attachments.
4. A weak edge may attach a singleton component to an existing core group.
5. A weak edge may not merge two existing core groups.
6. A weak edge may not form a long chain of singleton attachments.

Each edge stores:

- `edge_type`
- `association_decision`
- `rejection_reason`

## Local Overmerge Refinement

The initial graph groups pass through a Stage 1.5 local sanity check before
the Stage 1 catalogues are written. Under the default configuration, groups
with at least six Gaussians or a largest angular size of at least eight beam
widths receive additional risk weight, as do groups with weak chains, low
saddles, ridge gaps, multiple separated peaks, or an unusually large
low-threshold mask.

When the internal evidence supports a split, weak or poorly supported edges
are removed and the connected components of the refined graph become the
final local groups. A removed edge has one unambiguous final state:

```text
association_decision = false
edge_type = rejected
rejection_reason = local_sanity_split
local_edge_type = cut
```

The `local_edge_*` fields preserve the refinement decision for diagnostics.
All final group measurements, including centroid, LAS, morphology, quality,
edge counts, and artifact flags, are then recomputed from the refined group
membership and retained edges. `original_association_group_id` records the
initial graph group for provenance.

## Output Catalogs

Recommended outputs:

- `radio_association_groups.csv`
- `radio_association_groups.parquet`
- `radio_association_edges.csv`
- `radio_association_edges.parquet`
- `radio_association_components.csv`
- `radio_association_components.parquet`

Legacy merged-source catalogs are still written for compatibility, but new
analysis should use `radio_association_groups`.

Group catalog fields include:

- `association_group_id`
- `association_type`
- `association_quality`
- `artifact_risk_flags`
- `LAS_arcsec`
- `LAS_beam`
- `association_score_mean`
- `n_strong_edges`
- `n_weak_edges`
- `n_only_2sigma_edges`

Refinement diagnostics are written to `local_sanity_diagnostics.csv` and
`local_needs_visual_check.csv`.

## Association Types

The pipeline does not use source-class labels as association types.

Supported types:

- `compact_multi_gaussian`: several Gaussians inside a small beam-scale
  footprint
- `continuous_extended`: credible 3 sigma or 2.5 sigma connectivity, bridge, or
  ridge support
- `diffuse_extended`: larger low-surface-brightness group with weaker S/N and
  artifact caveats
- `linear_or_tail_like`: high axis ratio or ridge-like component layout
- `complex_association`: many associated Gaussians
- `weak_association`: low-score or weakly supported group
- `artifact_risk`: group dominated by negative bowl, sidelobe, or large-mask
  risk

## Association Quality

Quality is categorical, not a probability:

- `high`: strong edges dominate, score is high, only-2sigma evidence is rare,
  and at least one strong morphology signal is present
- `medium`: geometry and segmentation/bridge/ridge evidence are reasonably
  consistent without severe counter-evidence
- `low`: score is low or evidence is mostly distance/geometry
- `suspicious`: many only-2sigma edges, large mask/crowding concerns, or high
  score dispersion
- `artifact_risk`: negative bowl, sidelobe risk, or large-mask swallow risk is
  prominent

## Visualization Semantics

Overview titles use:

```text
cutout_id | gauss=... assoc_edges=... groups=... multi=... max_group=...
labels shown: ... | edges shown: ...
```

Zoom titles use:

```text
cutout_id group_id | n=... | quality=...
```

Zoom panels draw the configured contours, Gaussian component positions with
optional IDs, and accepted internal edges with their association scores when
`zoom.draw_edge_scores` is enabled.
