# Next release

Working notes on where the model is heading and what has to change around it. Not a spec — a
record of what we decided and what is still open, to be reviewed and edited as the answers land.

The shape of the plan: **converge the model config with a few more experiments, freeze the artifact
contract, then align the repo to it.** Everything below sits in one of those three phases.

## Where we are

Four arms measured through one harness on the full 118,459-document test split, thresholds fitted on
validation and only then applied to test.

| arm | acc | F1 | MCC | AUC | artifact | ms/doc |
|---|---|---|---|---|---|---|
| production (shipped 3.0.0) | 0.9562 | 0.9579 | 0.9123 | 0.9889 | 11.41 MB | 2.74 |
| magika flat window | 0.9681 | 0.9695 | 0.9360 | 0.9948 | 16.76 MB | 0.82 |
| hierarchical v32768-d128 | **0.9714** | **0.9725** | **0.9427** | **0.9960** | 16.84 MB | 0.40 |
| hierarchical v16384-d64 | 0.9677 | 0.9690 | 0.9353 | 0.9951 | **5.36 MB** | **0.25** |

The small hierarchical model beats what ships today on **every axis at once**: +1.16 pp accuracy,
+2.32 pp MCC, 2.1x smaller, ~11x faster. That is the candidate.

**Update (2026-08-31):** the hyperparameter campaign (EXPERIMENTS.md) moved the small model to
**0.9751 acc / 0.9501 MCC / 0.9968 AUC at 5.23 MB** with `group=2` + AdamW/cosine — past the large
model on every metric. Candidate configs on the size axis: 5.23 / 4.20 / 2.70 MB at
0.9751 / 0.9741 / 0.9732 accuracy.

What the experiment established, for the record:

- Magika's transferable parts were the **trunk** and **fixed padded input**. Its begin/middle/end
  sampling is file-format-specific and was correctly rejected — every flat-window arm lost to
  production on long documents purely through truncation.
- **Chunking was worth keeping.** It is what recovers long-document accuracy the flat window loses.
- The production model's weakness is **short text**: 0.8902 on single-chunk documents against
  0.9753 on multi-chunk, and 22.6% of the corpus is single-chunk. Fixed padded input is the fix.
- The published "AUC 0.9563" in `model_metadata.json` was never an AUC — `train.py` passes hard 0/1
  predictions to `roc_auc_score` at lines 428, 561, 650 and 689. Real value is 0.9889.

## Phase 1 — experiments before freezing the config

Ranked by expected value. Each is one run of the small model unless noted.

### Training setup is the least-tuned part and it demonstrably overfits

```
v32768-d128: best val_auc 0.9953 @ epoch 14; train loss 0.0316 -> 0.0236 after that
v16384-d64 : best val_auc 0.9947 @ epoch  8; train loss 0.0826 -> 0.0596 after that
```

The larger model drove training loss down 12x (0.2815 -> 0.0236) while validation AUC moved 0.008
in total and was flat from epoch 4. Early stopping is the only thing resisting it.

The optimizer is plain **Adam, constant lr 1e-3, no weight decay, no schedule**.

The 2026-08-30 campaign (EXPERIMENTS.md, "Hyperparameter campaign") answered most of this phase:

- [x] **AdamW + cosine decay with short warmup, `weight_decay≈0.01`.** Clear win (+0.34 pp acc
      alone), and cosine arms reliably train past the 10-24-epoch spread constant-LR showed across
      seeds. Flags exist now.

### Constants inherited from Magika, never questioned

- [x] **`group=1` or `2`.** `group=2` was the campaign's single largest win (+0.5 pp acc, outside
      noise). `group=1` confirms the direction but loses to 2 on test at 2.8x train cost — some
      grouping helps, four BPE tokens was too many. Adopted: 2.
- [x] **Chunk-position embedding.** Neutral alone and on top of the combo (best val AUC of the
      campaign, tied test). Rejected on simplicity.
- [x] **Mean+max chunk pooling.** Top of the noise band alone, clearly worse on top of the combo.
      Rejected.
- [x] **`--chunk-loss-weight 0` ablation.** Lands at the bottom of the noise band; the chunk head
      earns its keep. Keep 0.3.

Also untested at Magika defaults: `dropout`/`spatial_dropout` (0.1), `hidden` (256), `min_fill`
(0.10) — still untested, but with everything else converged these are low-value.

### Measurement gaps

- [ ] **Markdown-stripped control.** `dataset_metadata.json` records markdown in 33.66% of AI
      samples against 5.97% of human — a 5.6x ratio, headings 0.991 vs 0.0003. `cleaner.rs:888`
      asserts markdown survives cleaning, deliberately. So we still do not know how much of *any*
      of these numbers is formatting detection rather than style. Needs a control arm building.
      **Still open — the one Phase 1 item the campaign did not touch.**
- [x] **Seed repeats.** Noise floor measured: 0.27 pp accuracy / 0.0008 val AUC across three
      baseline seeds (the combo tightens to 0.09 pp). The CutMix variants (0.9677 / 0.9699 /
      0.9683) are confirmed indistinguishable from noise.

## Phase 2 — correctness gaps, not tuning

- [ ] **Cleaning is missing entirely from the new pipeline.** The parquets were cleaned during
      curation with `CleaningMode.TRAINING`; the training scripts apply **no cleaner**; Rust
      inference applies universal cleaning. Three different transformations across curation,
      training and inference. Production gets this right by calling the same cleaner from both
      sides. This is live skew, not future skew.
- [ ] **Nothing is calibrated.** Production wrapped its models in `CalibratedClassifierCV`. A raw
      sigmoid from an overfitting BCE-trained net is systematically overconfident, and the CLI
      prints "Model 95.1%" as a confidence figure. Measure a reliability curve; add temperature
      scaling fitted on validation if it is off. One parameter.

## Phase 3 — contract decisions to make before 1.0.0

These change the artifact, so deciding them after release means a breaking change.

- [ ] **Does the graph emit chunk-level output?** It currently emits only the document probability —
      `DocumentOnly` wraps the chunk head away. Exposing "this passage looks generated" as a product
      feature requires exporting both tensors. Decide now.
- [ ] **Is `encoder_config.json` sufficient for Rust to reconstruct encoding exactly?** It currently
      carries preprocessing version, tokenizer, chunk size, min_fill, overlap, vocab size, oov and
      pad ids. Anything missing is a silent skew source.
- [ ] **Does the Rust loader refuse a bundle whose `preprocessing_version` it does not match?**
      Failing loudly beats loading fine and being quietly worse.

## Phase 4 — aligning the repo

### Goes

| what | why |
|---|---|
| `pre_processor/vectorizer/` (tfidf, count, params) | no TF-IDF path in the new model — this is the 4.4 MB artifact |
| `pre_processor/ngrams.rs` | exists only to feed TF-IDF |
| `pre_processor/features/statistical.rs` | experimental, flag-gated, unused |
| sklearn ensemble + `skl2onnx` in `train.py` | replaced |
| aggregation in `pipeline/prediction.rs` | now inside the ONNX graph |
| `chunk_classification_threshold.txt` | no chunk threshold at document level |

Deleting the Rust aggregation is a real simplification, not just less code: production maintains a
weighted mean in `train.py` **and** an equivalent in Rust that must agree forever. The graph emitting
one probability removes that whole class of drift.

### Stays

`cleaner.rs` and `tokenizer.rs` unchanged and essential. `chunker.rs` survives as a file but its
algorithm is replaced — the even-distribution solver goes, constant-size + pad + drop-runt replaces
it. `ort` inference, CLI, PyO3/JNI bindings and packaging are unaffected.

The even-distribution chunker is worth removing on its own merits: it emits a redundant trailing
chunk on **31% of multi-chunk documents** (always a strict subset of its predecessor, so the tail is
double-counted by the weighted mean), and produces 164 distinct chunk sizes between 115 and 164
tokens despite being configured for 150.

### New

A `DocumentEncoder` in the preprocessing crate — clean, tokenize, constant chunk, apply the
vocabulary, pad, emit `[chunks, chunk_size]` — constructed by `from_bundle()` so chunk size and
vocabulary come from the artifact rather than being assumed at either end. Exposed through PyO3
alongside the existing classes so **training calls the same code inference does**. Plus safetensors
reading for `token_vocabulary.safetensors`.

Worth adding either way: a **golden-vector test**, a handful of documents with expected encodings and
probabilities checked in, asserted from both the Rust and Python sides, so drift fails CI rather than
quietly degrading accuracy. `python/is-it-slop-preprocessing/tests/test_training_integration.py` is
the existing pattern.

## Phase 5 — consolidating `notebooks/`

Two problems, both worth fixing while the pipeline is being rewritten anyway.

### Flip the source of truth to `.py`

Today `.ipynb` is authoritative: `just sync-notebooks` runs `notebooks-to-scripts`, which regenerates
every `.py` from its notebook via `nbconvert`. Both `dataset-curation` and `training-pipeline` depend
on it, so **editing a `.py` and running the pipeline silently discards the edit**. The whitespace
churn currently sitting unstaged in four `notebooks/*.py` files is exactly that mechanism firing.

Wanted: scripts are authoritative and directly runnable, notebooks generated on demand for
exploration (`jupytext --to notebook`) and never committed as the source. The four scripts added
during this work — `magika_style_train.py`, `hierarchical_train.py`, `compare_models.py`,
`analyse_comparison.py` — already follow that pattern: argparse entry points, no paired notebook, run
directly.

### Untangle `notebooks/__init__.py`

It does far too much at import time:

- `RETRAIN_VECTORIZER = "--force-retrain-vectorizer" in sys.argv or True` — the trailing `or True`
  makes the flag dead; it is always on.
- Parses `sys.argv` **at module import**, so behaviour depends on the argv of whatever imports it.
- Four `mkdir()` calls as import side effects. Importing the module creates empty version
  directories — this is why stray `data/3.1.0/` and `data/4.0.0/` appear.
- Four module-level `pl.scan_parquet()` calls: I/O at import.
- The version bump on import means `DATA_DIR` points at a version that does not exist on disk, which
  is why the new scripts pin `data/3.0.0` explicitly rather than importing it.
- `MODEL_VERSION = "3.0.0"` duplicates `crates/is-it-slop/build.rs:17`. The abandoned
  `# from is_it_slop import MODEL_VERSION` above it shows the original intent.

Wanted: paths and constants as plain values or a small function, no argv parsing, no I/O and no
directory creation at import, and one source of truth for the model version shared with `build.rs`.

## Burn training port

The training pipeline is ported to Rust (`crates/is-it-slop-training`, Burn 0.21) and **passed the
parity gate**: one full frozen-recipe run landed at val AUC 0.9960 / test AUC 0.9967 against
Python's 0.9961 / 0.9968 — inside seed noise (EXPERIMENTS.md, "Burn training port"). The Rust
implementation is the reference now; encoder and vocabulary behaviour is frozen by contract tests
(`tests/encoder_golden.rs`, `tests/vocab_parity.rs`) because trained artifacts depend on it, and
the PyTorch weight-parity scaffolding is retired. The binary
(`cargo run -p is-it-slop-training --features metal`) trains with the TUI dashboard and prints the
full-split AUCs; backends are additive features (metal/cuda/rocm/vulkan/wgpu/tch/remote/flex,
ndarray default) with `f16`/`flex32` element options.

Open items, in order:

- [ ] **Throughput: Burn-Metal trains ~2.3x slower than torch-MPS** (7.3 vs 3.2 min/epoch). Root
      cause unproven. The investigation ladder, cheapest first, all runnable via existing flags:
      1. `--features tch-gpu` A/B — same libtorch-MPS kernels as Python, so it cleanly splits
         "Burn framework overhead" from "CubeCL-vs-MPSGraph kernel gap".
      2. `--features metal,f16` and `metal,flex32` — Apple silicon's f16 throughput is roughly
         double f32; needs a parity re-check since the recipe was tuned at f32.
      3. `burn.toml` autotune `level = "full"` and cubecl profiling loggers to see where time goes.
      4. ~~Fusion hygiene in `model.rs::forward`~~ — done: masks are now derived up front so the
         trunk runs as one uninterrupted compute region. Measured no throughput change (114.8 vs
         115.2 s/1000 iterations, within noise), so the fusion boundaries were never the
         bottleneck; kept for readability. Burn's fusion (fuse-on-read/write) is the framework's
         `torch.compile` analogue and is already enabled — there is no whole-graph AOT compile to
         switch on.
      5. Custom CubeCL kernels only if profiling shows a specific hot fused-region CubeCL misses —
         the matmuls are autotuned and not worth hand-writing; the candidate would be the masked
         group-pool + norm chain. Last resort, real maintenance cost.
- [ ] **Evaluation harness**: the binary reports full-split AUC only. Threshold fitting
      (best-F1, precision>=0.99 on validation), per-dataset and per-length breakdowns still live in
      Python (`compare_models.py`); port or keep, decide with the artifact contract.
- [ ] **Artifact contract**: Burn cannot export ONNX, so shipping the Burn-trained model means
      Burn-native inference (record/Burnpack + the existing `token_vocabulary.safetensors`) and
      retiring `ort` — the Phase 3/4 decision this port was building toward. The trained bundle
      currently saves `model.mpk` + vocabulary + configs.
- [ ] **CutMix** remains unported pending the chunk-output decision (its deltas are inside noise).

## Versioning

Model artifacts **3.0.0 -> 4.0.0** — breaking: different files, different input shape, no vectorizer.
`build.rs`, the `model_metadata.json` consumers and the `justfile` release targets all move together.

Crate/package **1.0.0** once the contract above is frozen. The hesitation about the label is not the
model — it is that 1.0.0 implies a stable contract while the three Phase 3 questions are open.

## Open questions

- ~~Small model or large?~~ **Answered by the campaign**: the tuned small model (5.23 MB) beats the
  large one (0.9751 vs 0.9714) — the large/small gap was never architecture, it was training. Three
  candidate sizes now exist (5.23 / 4.20 / 2.70 MB at 0.9751 / 0.9741 / 0.9732); which ships is a
  product call, not a modelling one.
- Keep CutMix? Its document-level deltas are now confirmed to be inside the measured noise floor.
  Its real justification is chunk-level supervision, which is unmeasurable on this corpus — no real
  mixed-authorship documents exist in it. Decide alongside the chunk-output question.
- ~~`--max-chunks` for the shipped model.~~ **Answered**: k=16 ties k=8 on headline and buys only
  the 2000+-token tail (0.9984 -> 0.9992 on 3.2% of docs) at 2.7x train cost. k=32 has nothing left
  to win. Ship k=8 unless the long tail matters to the product.
