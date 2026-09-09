# Next release

Working notes on where the model is heading and what has to change around it. Not a spec — a
record of what we decided and what is still open, to be reviewed and edited as the answers land.

The shape of the plan: **converge the model config with a few more experiments, freeze the artifact
contract, then align the repo to it.** Everything below sits in one of those three phases.

## Where we are

Four arms measured through one harness on the full 118,459-document test split, thresholds fitted on
validation and only then applied to test.

| arm                        | acc        | F1         | MCC        | AUC        | artifact    | ms/doc   |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ----------- | -------- |
| production (shipped 3.0.0) | 0.9562     | 0.9579     | 0.9123     | 0.9889     | 11.41 MB    | 2.74     |
| magika flat window         | 0.9681     | 0.9695     | 0.9360     | 0.9948     | 16.76 MB    | 0.82     |
| hierarchical v32768-d128   | **0.9714** | **0.9725** | **0.9427** | **0.9960** | 16.84 MB    | 0.40     |
| hierarchical v16384-d64    | 0.9677     | 0.9690     | 0.9353     | 0.9951     | **5.36 MB** | **0.25** |

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
      asserts markdown survives cleaning, deliberately. So we still do not know how much of _any_
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
- [x] **Calibration.** Done: the training run fits a temperature on validation and ships it as
      `calibration_temperature.txt`; inference computes `sigmoid(logit / T)` and the threshold is
      fit in that calibrated space (see "Artifact contract" below).

## Phase 3 — contract decisions to make before 1.0.0

These change the artifact, so deciding them after release means a breaking change.

- [ ] **Are chunk-level outputs part of the contract?** The model's forward returns chunk logits,
      the library exposes them as uncalibrated per-chunk probabilities, and the CLI's `--chunks`
      view shows them. Undecided: whether that is a stable, documented output (which pins the chunk
      head and its calibration story) or an inspection aid that may change.
- [ ] **Is `encoder_config.json` sufficient for Rust to reconstruct encoding exactly?** It currently
      carries preprocessing version, tokenizer, chunk size, min_fill, overlap, vocab size, oov and
      pad ids. Anything missing is a silent skew source.
- [x] **Does the Rust loader refuse a bundle whose `preprocessing_version` it does not match?**
      Done: `DocumentEncoder::from_bundle` refuses a bundle whose version is not semver-compatible
      with the preprocessing crate (same major; same minor while the major is 0), so a bundle
      survives patch releases and fails loudly across a breaking one.

## Phase 4 — aligning the repo

### Goes

| what                                               | why                                                           |
| -------------------------------------------------- | ------------------------------------------------------------- |
| `pre_processor/vectorizer/` (tfidf, count, params) | no TF-IDF path in the new model — this is the 4.4 MB artifact |
| `pre_processor/ngrams.rs`                          | exists only to feed TF-IDF                                    |
| `pre_processor/features/statistical.rs`            | experimental, flag-gated, unused                              |
| sklearn ensemble + `skl2onnx` in `train.py`        | replaced                                                      |
| aggregation in `pipeline/prediction.rs`            | now inside the ONNX graph                                     |
| `chunk_classification_threshold.txt`               | no chunk threshold at document level                          |

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

### Port dataset curation to Rust

`notebooks/dataset_curation.py` is **currently broken**: it imported `TextCleaner`/`tokenize` from
the preprocessing Python bindings, which were removed with the rest of the Python training env
(the training-time cleaning and tokenization it needs still exist in Rust — only the bindings are
gone). It is the last Python step in the pipeline: download the source datasets (kagglehub,
HuggingFace), clean, dedupe, stratify, and write the released `data/<version>` parquet splits.
Port it as a `dataset-curation` subcommand or crate alongside `is-it-slop-training` — polars is
already a workspace dependency, and the cleaner/tokenizer are one crate away. Until then, curating
a new dataset version means either resurrecting the bindings locally (`git checkout 9725e29 --
crates/is-it-slop-preprocessing python/is-it-slop-preprocessing` and `maturin develop`) or calling
tiktoken directly from Python. The existing `data/3.0.0` splits are unaffected — training runs
read the released parquets and never re-curate.

## Burn training port

The training pipeline is ported to Rust (`crates/is-it-slop-training`, Burn 0.21) and **passed the
parity gate**: one full frozen-recipe run landed at val AUC 0.9960 / test AUC 0.9967 against
Python's 0.9961 / 0.9968 — inside seed noise (EXPERIMENTS.md, "Burn training port"). The Rust
implementation is the reference now; encoder and vocabulary behaviour is frozen by contract tests
(`tests/encoder_golden.rs`, `tests/vocab_parity.rs`) because trained artifacts depend on it, and
the PyTorch weight-parity scaffolding is retired. The binary
(`cargo run -p is-it-slop-training --features metal`) trains with the TUI dashboard and prints the
full-split AUCs; backends are additive features (metal/cuda/rocm/vulkan/wgpu/tch/remote/flex,
ndarray default) with a `flex32` element option (pure f16 was removed after it NaN'd).

Open items, in order:

- [x] **Throughput: understood.** Burn trains ~2.4x slower than torch-MPS and the decomposition
      in EXPERIMENTS.md ("Training throughput") pins the whole gap on the autodiff-generated
      backward pass (~97 vs ~31 ms/iter): forward is at kernel parity (12.2 vs 11.4 ms), and
      harness, dataloader, optimizer, metric sync, and transfers are all noise. It reproduces on
      burn-tch, so it is Burn's backward decomposition, not CubeCL — fixing it means upstream
      Burn work, not changes here. Accepted for now: training is a per-release one-off.
      **A pure-f16 training element does not work and the feature was removed**: a full-data
      attempt (2026-09-01) produced NaN loss from epoch 1 — no loss scaling means this recipe
      overflows (BCE exp/log, attention softmax). Reviving it means loss scaling, i.e. real work.
      `--features metal,flex32` (f32 semantics, backend free to relax intermediate precision) is
      the untested remaining candidate. Rerun the benches with `examples/step_bench.rs` +
      `notebooks/step_bench.py`. Known sharp edge: a forward-only loop on the `Autodiff` backend
      leaks and gets SIGKILLed — validation belongs on the inner backend.
- [ ] **Evaluation harness**: threshold fitting (best-F1, precision>=0.99) and an uncertainty
      block (temperature calibration plus three error-ranking confidence signals: calibrated
      margin, chunk dispersion, doc-vs-chunk agreement) are ported and land in `results.json`
      each run. Still Python-only: per-dataset and per-length breakdowns (`compare_models.py`) —
      the per-length one is also how the long-document-splitting arm gets judged. Plan: a light
      script over the inference Python bindings once those run the Burn model; the notebooks
      retire then.
- [ ] **Artifact contract**: weight formats decided and emitted — `model.safetensors` is the
      canonical shipped artifact (inspectable, consistent with the vocabulary; embeds via
      `from_bytes` with one ~4.6MB memcpy at startup, irrelevant at this size);
      `model.f16.safetensors` is the half-size candidate; `model.bpk` is **local-only, never
      uploaded** (zero-copy embedding option, checkpoint format once the Burn version allows —
      until then checkpoints stay MessagePack because the `Learner` checkpointer takes a
      `Recorder`, which burn-store does not implement). The decision contract is _calibrated_:
      inference computes `sigmoid(logit / T)` with `calibration_temperature.txt` and classifies
      at `classification_threshold.txt`, both fit on validation in calibrated probability space.
      Uncertainty contract: calibrated distance-from-0.5 is the confidence number (strongest
      error-ranker); chunk-probability dispersion is the secondary "mixed content" indicator,
      paired with per-chunk highlighting; the doc-vs-chunk-agreement signal was measured weakest
      and dropped. The `evaluate` subcommand (renamed from `check`) re-scores full/f16/Q8 on validation and folds
      `artifact_checks` into `results.json` — run it after every training run; adopting f16 or
      Q8 as _the_ shipped form is that eval-gated decision. **The inference swap is done**:
      `is-it-slop` embeds `model.f16.safetensors` (build.rs `MODEL_VERSION` 4.0.0) and runs it
      on Burn's Flex CPU backend — ort, TF-IDF, sprs, and `AggregationMethod` are gone; the
      network shares one definition via the `is-it-slop-model` crate and one text→grid path via
      `is-it-slop-preprocessing::encoder`. Long documents are no longer truncated at inference
      (dynamic chunk axis, capped at 256 chunks ≈ 49k tokens, each document padded only to its own
      last chunk — see "Inference performance") — a deliberate behavior change to
      quantify on the full run's long-doc band, since T and the threshold were fit at the 8-chunk
      eval encoding. Still optional: shrinking the vocabulary LUT from i32 to u16
      (~800KB -> ~400KB, breaking).
- [ ] **Resume a training run.** A killed or crashed run currently restarts from scratch even
      though the `Learner` writes full-precision checkpoints every epoch — wire up
      checkpoint resume (a `--resume`/`--from-checkpoint` flag on `train`) so a multi-hour full
      run survives interruption. Care points: the LR schedule and early-stopping state must
      resume from the same step, and the encoded-split/vocabulary must be rebuilt identically
      (seeded, so they are).
- [ ] **Improve the training CLI.** The `train` subcommand exposes a fraction of what the Python
      trainer took (the richer clap surface was explicitly deferred): batch size, epochs,
      patience, LR, chunk-loss weight, dropout are all recompile-only today. Bring the
      `TrainingConfig` knobs up as flags with the frozen recipe as defaults, and make `evaluate`
      accept a specific weights file rather than assuming the artifact-dir layout.
- [ ] **Regression benchmarks** (late-game want): pin today's numbers so upgrades (Burn 0.22,
      toolchains) can't silently regress. Inference outputs are already pinned (the golden parity
      test) and inference latency/throughput is measured (`just bench-inference`, results under
      "Inference performance"); what is missing is a recorded baseline for that bench and for
      training throughput (`examples/step_bench.rs`), plus end-model quality via a fixed
      `--limit` run with a tolerance band.
- [ ] **CutMix**: port after the splitting re-baseline lands, as its own arm. The design, settled
      in review: chunk isolation makes spliced grids _clean_ supervision for the chunk head (it
      sees only its own chunk, so per-chunk labels from different source documents are exact);
      the distribution risk lands on the document head, bounded the way Python bounded it — only
      ~15% of batches spliced, and the spliced document's target becomes _soft_ (the fraction of
      AI chunks), which is the honest label for a mixed document. Porting needs `Batch` to carry
      chunk labels (and a soft document label) instead of deriving them, plus the splice in the
      batcher. Judge it by chunk-level quality — the dispersion signal and mixed-content behavior
      — not document AUC, where its deltas were already measured inside noise.
- [ ] **Training cost is now the binding constraint**, and it reshapes the model-size strategy.
      Burn trains ~2.4x slower than the old torch path, and quantization being measured as nearly
      free (f16 and Q8-matrices cost ~1e-6 and ~1e-4 validation AUC on the smoke; confirm at full
      scale via `evaluate`) means the balanced/minimal picks lose their reason to exist: train the
      accuracy model once, ship it at f16 (~2.3MB) or Q8 (~1.3MB of matrices) instead of training
      smaller models. It also opens the larger-embedding arm the campaign priced out — train big,
      quantize down. Practical cadence: `--limit` runs for iteration, full/larger runs scheduled
      long (overnight or weekend). One honest caveat: on Burn 0.21 quantization shrinks the
      _artifact only_ — there are no integer compute kernels, tensors dequantize for every op —
      so expect no inference speedup from Q8 today (if anything a small slowdown); the win is
      size and memory.

## Inference performance

Measured with `just bench-inference` (`crates/is-it-slop/benches/inference.rs`) on a fixed 128-document
sample of the test split (`benches/corpus.jsonl`), Flex CPU backend, Apple Silicon with 15 logical cores.
Outputs are pinned by the golden parity test (`crates/is-it-slop/tests/inference_golden.rs`: 12 real
documents from 1 to 33 chunks at tolerance 1e-5, regenerated only with `IS_IT_SLOP_UPDATE_GOLDEN=1`
when the model artifacts change), so every change below reproduced the previous outputs exactly.

**What the profile showed.** A forward's cost is linear in chunk slots, and one Flex forward runs on a
single thread. So (a) padding rows are paid for in full even though the model masks them, and (b) the
batch dimension buys no parallelism: batching documents into one forward only adds padding to the
longest document in the batch, while independent forwards on several threads scale with cores.

**Padding to the real chunk count** (`DocumentEncoder::encode_batch_chunks`). Inference previously
reused the training tiling: whole 8-chunk windows, each padded to 8 rows, so a 1-chunk document ran
as 8 slots. Now only the last chunk is padded to 192 tokens. Training is unchanged; the tiling is a
training-batch requirement, not a property of the weights.

| Real chunks | Whole windows | Real chunks only |
| ----------- | ------------- | ---------------- |
| 1           | 6.1 ms        | 0.77 ms          |
| 8           | 7.7 ms        | 7.45 ms          |
| 12          | 14.1 ms       | 11.2 ms          |

**One document per forward, documents concurrent** (on CPU backends `pipeline::predict_each` runs
one forward per document on rayon's pool, order preserved; library-level, so Python and Kotlin callers
get it too). Documents per second on the real corpus, by how many documents `predict_each` is handed
at once:

| Documents per call | Original (padded batch) | Real chunks only | Plus concurrency |
| ------------------ | ----------------------- | ---------------- | ---------------- |
| 1                  | 147                     | 399              | 372              |
| 32                 | 79                      | 117              | 2273             |
| 128                | 81                      | 105              | 3416             |

**Backends and the scheduling split.** The backend is a cargo feature on `is-it-slop` (Flex by
default; `ndarray`, `accelerate` (ndarray + Apple's BLAS, macOS only), `apple-amx` and `cpu`
(burn-cpu) as other CPU backends; `metal`, `wgpu`, `vulkan`, `webgpu`, `cuda`, `rocm` as
accelerators, which set `cfg(accelerated)`). `pipeline::Strategy { docs_per_forward, concurrent }` is
the scheduling policy, `predict_each_with` takes one explicitly, and the compiled backend's default
lives in `pipeline/cpu.rs` or `pipeline/accelerated.rs`. Golden parity holds at 1e-5 on every backend
run. Only Flex runs in CI. Run locally on an Apple M-series machine (15 logical cores) over 512 real
documents: flex, flex + AMX, ndarray, ndarray + Accelerate, wgpu, metal, webgpu. `cuda` and `rocm`
compile only; `vulkan` needs the Vulkan SDK to build on macOS; `cpu` (burn-cpu) aborts with a stack
overflow on its first forward.

The `strategy` bench group sweeps documents per pass against concurrent passes; documents per second:

| Docs per pass | Flex seq | Flex par | ndarray par | Accelerate par | wgpu seq | wgpu par | Metal seq | Metal par | webgpu seq |
| ------------- | -------- | -------- | ----------- | -------------- | -------- | -------- | --------- | --------- | ---------- |
| 1             | 368      | 3424     | 4107        | 5102           | 145      | 194      | 172       | 215       | 129        |
| 4             | 366      | 2667     | 3285        | 4223           | 613      | 671      | 720       | 740       | 467        |
| 16            | 342      | 1308     | 1676        | 2215           | 2419     | 2120     | 2465      | 2315      | 2111       |
| 32            | 310      | 815      | 1035        | 1364           | 3569     | 2256     | 4294      | 2574      | 3262       |
| 64            | 250      | 441      | 560         | 752            | 3932     | 2508     | 4882      | 3533      | 3790       |
| 128           | 175      | 230      | 297         | 402            | 3320     | 3067     | 4639      | 4255      | 3343       |
| 256           | 107      | 118      | 153         | 211            | 2212     | 2209     | 3125      | 1626      | 2282       |
| 512           | 60       | 58       | 72          | 109            | 1329     | 1331     | 1723      | 1881      | 1345       |

The two families want opposite policies, conclusively. Every CPU backend is best at one document per
pass with passes concurrent, and batching degrades monotonically. Every GPU backend is best at 64
documents per pass run sequentially: a pass costs 6-8 ms of launch and sync whether it holds 1 chunk or
16, passes serialise on the device queue so concurrency costs 25-40% at 32-64 per pass, and past 64
the padding to the longest document in a pass outweighs the saved launches. Those are the two
defaults. AMX made no difference to Flex; Accelerate's BLAS made ndarray 49% faster than Flex.

The `input_size` group is the latency a caller with N documents in hand sees, in ms, each backend
on its own default policy:

| N   | Flex  | ndarray | Accelerate | wgpu  | Metal | webgpu |
| --- | ----- | ------- | ---------- | ----- | ----- | ------ |
| 1   | 3.6   | 2.7     | 2.0        | 7.4   | 6.1   | 7.9    |
| 8   | 9.0   | 7.1     | 5.6        | 12.2  | 8.9   | 14.2   |
| 32  | 16.0  | 12.7    | 10.2       | 18.1  | 14.4  | 19.9   |
| 64  | 26.2  | 22.1    | 16.8       | 30.9  | 22.9  | 30.0   |
| 128 | 52.3  | 42.9    | 34.7       | 62.2  | 45.6  | 61.6   |
| 256 | 85.2  | 67.3    | 55.9       | 84.0  | 65.3  | 86.5   |
| 512 | 165.7 | 130.1   | 106.8      | 131.9 | 103.7 | 135.2  |

On this machine the GPU is not clearly the faster device: ndarray + Accelerate beats Metal up to 256
documents per call and ties it at 512; Flex beats Metal only below 8 documents, ties at 8, and loses
from 16 up (Metal is 25% faster at 512). A single document costs 6 ms on Metal against 2-4 ms on the
CPU backends. Metal's CLI window sweep kept climbing (3949 docs/s at 256, 5004 at 512) because length
sorting works better with more documents per window.

**Hybrid routing (done).** An accelerated build keeps the compiled CPU backend's model
(`model::CPU_MODEL`) and `pipeline::accelerated::infer_each` sends calls below `CPU_UP_TO` to it under
the CPU policy; `predict_each_with` bypasses the routing. The crossover is the measured tie point of the
compiled CPU backend against Metal: 8 for Flex, 256 for ndarray, 512 for ndarray + Accelerate. With
routing, a plain `metal` build scores one document at 279 docs/s instead of 164, and calls from 8
documents up are unchanged. The tie points are from an Apple M-series machine; a discrete GPU would
cross lower.

**Shipping policy.** The default build (`default = ["cli"]`) is Flex only: no accelerator crate is in
its dependency graph and every accelerated code path is behind `cfg(accelerated)`, so the shipped
binaries and the Python/JNI packages carry none of it. Flex stays the CPU default (ndarray's lead here
is Apple-specific and unmeasured on x86); `wgpu` is the portable accelerator, and `metal`, `vulkan`,
`cuda` and `rocm` override it when enabled. An accelerated build panics at its first large call if the
machine has no usable GPU device, so a runtime probe that demotes the build to the CPU path is required
before any accelerated binary is shipped as a general download.

**Decided.** The CLI's `BATCH_SIZE` is 256 (window sweep on Flex: 2143 / 2664 / 3021 / 3375 / 3464
docs/s at 32 / 64 / 128 / 256 / 512, so 256 is 97% of the plateau at ~85 ms to first output, and holds
four of the GPU policy's 64-document passes).

**Next.** (1) Runtime GPU probe with CPU fallback before any accelerated binary ships. (2) An x86
Linux measurement of Flex vs ndarray, and of a discrete GPU's crossover, before revisiting any default.
(3) A GPU build may want a larger CLI window than 256. (4) burn 0.22.0-pre.3 is not a bump: the
pre-release removes the backend type parameter from tensors and modules (`Tensor<const D, K>`,
`Linear` with no `<B>`) and dispatches backends at runtime, so the model, training and inference
crates all need porting (21 errors in `is-it-slop-model` alone). Worth doing when 0.22 releases, not
least because runtime dispatch is the natural home for item (1); until then the numbers above stand.

## Versioning

Model artifacts **3.0.0 -> 4.0.0** — breaking: different files, different input shape, no vectorizer.
`build.rs` and the `justfile` release targets moved together (done); `model_metadata.json` is no
longer produced or required. The root README still describes the v3 TF-IDF/ONNX architecture and
quotes v3 numbers — rewrite it alongside the v4 release once the full run's numbers land.

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

- ~~Proper error types in the library interface.~~ **Done**: thiserror enums with structured variants
  in the model, preprocessing and inference crates; anyhow with context in the training binary.
- ~~Speed benchmarks on inference.~~ **Done**: see "Inference performance", including the CPU vs GPU
  scheduling sweep across seven backends and the hybrid CPU/GPU routing; `BATCH_SIZE` is decided
  (256).
- text without markdown formatting is often passed as human but text with markdown formatting is often passed as AI. How much of the model's accuracy is detecting formatting rather than style? A control arm stripping markdown from all text wouldanswer that.
- I have noticed one of the most pervasive tells of LLM-isms is that they love to make inanimate objects the subject of verbs. For example an llm says 'Your conversation has been moved to the background - enter opens it - esc returns to it - ctrl+c twice quits', where a human would have written 'enter to open, esc to return, ctrl+c twice to quit'. Is this something we could try and explicitly train on? You could call something like this the personification of abstract nouns, but not always strictly abstract. Often very frequent with technical nouns 'holiday has the rest'. Is this something we could do in our training?
