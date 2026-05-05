# GitHub Actions Workflows

## Pipeline overview

```text
push/PR
  └─ PR Checks ─────── fmt, clippy, Rust tests, Python tests

push to main
  └─ Crates.io Publish ── publish crates to crates.io
       │                    create GitHub Release
       ├─ PyPI Publish ──── Python wheels → PyPI
       └─ CLI Binaries ──── CLI binaries → GitHub Release
```

## Workflows

| Name | File | Trigger | What it does |
| --- | --- | --- | --- |
| **PR Checks** | `CI.yaml` | `push: main` (path-filtered), `pull_request` | Rust fmt/clippy/tests/docs + Python maturin build + pytest |
| **Crates.io Publish** | `release-plz.yaml` | `push: main` | Opens version-bump PRs (`release-pr` job); on merge: publishes to crates.io, creates git tag + GitHub Release (`release` job) |
| **PyPI Publish** | `release.yaml` | `release: published`, `workflow_dispatch` | Builds wheels for 12 targets (3 OS × 4 platforms) via maturin, publishes to PyPI via `uv publish` |
| **CLI Binaries** | `release-binaries.yaml` | `release: published`, `workflow_dispatch` | Cross-compiles `is-it-slop` CLI for 7 targets, attaches .tar.gz/.zip to the GitHub Release |

## How releases work

1. **Push to main** → `PR Checks` runs (skip if only non-code changes)
2. **Crates.io Publish** opens a version-bump PR (e.g. `release-plz-2026-...`)
3. **Merge the PR** → `PR Checks` runs again → `Crates.io Publish` publishes the crate to crates.io and creates a GitHub Release
4. **GitHub Release created** → triggers both `PyPI Publish` and `CLI Binaries` (parallel)
5. **Result**: new version on crates.io, PyPI, and GitHub Releases

## Manual fallback

Both `PyPI Publish` and `CLI Binaries` accept a `workflow_dispatch` with a `tag` input — useful if a `release: published` event was missed (e.g. PAT cascading failure).

## Feature flag matrix

| Crate | `python` | `cli` |
| --- | --- | --- |
| `is-it-slop-preprocessing` | `bincode, numpy, progress-bars, pyo3, pyo3-log, rkyv, serde` | — |
| `is-it-slop` | `pyo3` | `clap, serde, anyhow` |
