# Release checklist

Use this checklist before publishing a package version or paper archive.

## Source and configuration

- [ ] Changes are consolidated on `main`.
- [ ] YAML files load with duplicate-key and unknown-key validation.
- [ ] Beam, pixel scale, RMS, coordinate, and catalogue requirements are
  documented and validated.
- [ ] Effective configuration and its hash are saved with each production run.
- [ ] No local absolute paths, credentials, generated products, or private
  survey data are included in the source archive.

## Scientific checks

- [ ] Stage 1 candidate generation, graph grouping, and deterministic ordering
  are covered by tests.
- [ ] Stage 2 parent-link candidate uniqueness and conflict resolution are
  covered by tests.
- [ ] Synthetic fixtures cover compact, extended, unrelated, bridge, artifact,
  image-edge, and coordinate-wrap cases relevant to the configured method.
- [ ] Missing physical metadata fails loudly in strict production mode.

## Packaging and documentation

- [ ] `python -m build` produces an sdist and wheel.
- [ ] A fresh wheel installation imports `lotss_association` outside the source
  tree.
- [ ] Installed commands respond to `--help`: `lotss-association`,
  `lotss-parent-link`, and `lotss-visualize`.
- [ ] README documents inputs, outputs, configuration, and reproducibility.
- [ ] `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md`, and
  `lotss_association.__version__` agree.

## Verification commands

```text
pytest
ruff check lotss_association scripts tests
python -m compileall -q lotss_association scripts tests
python -m build
git diff --check
```

Record the actual command results in the release issue or archival record.
