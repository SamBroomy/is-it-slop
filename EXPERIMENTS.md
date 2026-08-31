# Experiment results

Reference for the Magika-inspired model work. Numbers are transcribed from the saved artifacts, not
from memory; the source file is named against each table.

Every arm is scored on the **same released `data/3.0.0` splits** — 473,848 train / 118,459 test /
95,009 validation — and, unless stated otherwise, through **one harness** (`compare_models.py`) on a
single CPU thread, with the decision threshold fitted on validation and only then applied to test.

## Headline

Superseded on the tuning axes by the [hyperparameter campaign](#hyperparameter-campaign-2026-08-30)
below, which improved the small model past every arm in this table.

Source: `model_artifacts/magika_style/comparison-tokens.json`

| arm | rule | thr | acc | prec | rec | F1 | MCC | AUC |
|---|---|---|---|---|---|---|---|---|
| production | shipped | 0.3993 | 0.9562 | 0.9607 | 0.9551 | 0.9579 | 0.9123 | 0.9889 |
| production | best_f1 | 0.3954 | 0.9561 | 0.9601 | 0.9555 | 0.9578 | 0.9121 | 0.9889 |
| production | precision≥0.99 | 0.6878 | 0.9441 | 0.9793 | 0.9120 | 0.9445 | 0.8907 | 0.9889 |
| magika flat | best_f1 | 0.2054 | 0.9681 | 0.9671 | 0.9718 | 0.9695 | 0.9360 | 0.9948 |
| magika flat | precision≥0.99 | 0.6900 | 0.9638 | 0.9851 | 0.9449 | 0.9646 | 0.9285 | 0.9948 |
| **hier v32768-d128** | best_f1 | 0.4458 | **0.9714** | 0.9755 | 0.9695 | **0.9725** | **0.9427** | **0.9960** |
| **hier v32768-d128** | precision≥0.99 | 0.9432 | 0.9674 | 0.9875 | 0.9494 | 0.9681 | 0.9355 | 0.9960 |
| **hier v16384-d64** | best_f1 | 0.4783 | 0.9677 | 0.9709 | 0.9670 | 0.9690 | 0.9353 | 0.9951 |
| **hier v16384-d64** | precision≥0.99 | 0.8470 | 0.9632 | 0.9873 | 0.9416 | 0.9639 | 0.9276 | 0.9951 |

At the ≥99%-precision operating point — the one that bounds how often a human is wrongly accused —
hier-large recovers **+3.74 pp recall** over production (0.9494 vs 0.9120), hier-small **+2.96 pp**.

## Cost

Source: same file. 300 documents, one CPU thread, idle machine.

| arm | ms/doc | vs production | artifact | notes |
|---|---|---|---|---|
| production | 2.74 | 1.0x | 11.41 MB | preprocess 0.19 + model 2.55 |
| magika flat | 0.82 | 3.4x | 16.76 MB | fixed 1,536-token window |
| hier v32768-d128 | 0.40 | 6.8x | 16.84 MB | 2.62 chunks/doc, real chunks only |
| hier v16384-d64 | **0.25** | **10.9x** | **5.36 MB** | 2.62 chunks/doc, real chunks only |

Production's cost is the *model*, not preprocessing: the ensemble consumes a dense 105,190-float
vector (420 KB) per chunk at 4.36 chunks/doc, ~1.8 MB fed per document. The hierarchical arms feed
~480 tokens because the exported chunk axis is dynamic, so cost tracks document length.

Sizes count the whole bundle. `4.59 MB` quoted earlier for hier-small was the graph alone; the
vocabulary (0.77 MB) is required at inference, so the honest figure is **5.36 MB**. Production is
likewise counted as ONNX + vectorizer.

Latency is sensitive to machine load — an earlier run with training in the background measured
production at 6.00 ms. Ratios held (3.9x vs 3.4x), absolutes did not. Only idle-machine numbers are
recorded here.

## Accuracy by document length

Source: `model_artifacts/magika_style/comparison-per-length-tokens.csv`

| tokens | n | production | magika flat | hier v32768-d128 | hier v16384-d64 |
|---|---|---|---|---|---|
| 0–50 | 9,685 | 0.8293 | 0.8981 | **0.9113** | 0.9012 |
| 50–100 | 7,013 | 0.9092 | **0.9485** | 0.9434 | 0.9410 |
| 100–150 | 7,309 | 0.9325 | 0.9495 | **0.9517** | 0.9509 |
| 150–200 | 9,570 | 0.9485 | **0.9623** | 0.9601 | 0.9529 |
| 200–300 | 22,179 | 0.9558 | 0.9637 | **0.9685** | 0.9642 |
| 300–500 | 28,897 | 0.9766 | 0.9796 | **0.9826** | 0.9789 |
| 500–1000 | 19,962 | 0.9883 | 0.9879 | **0.9919** | 0.9899 |
| 1000–2000 | 10,007 | 0.9969 | 0.9949 | 0.9968 | **0.9975** |
| 2000+ | 3,837 | **0.9992** | 0.9958 | 0.9984 | 0.9966 |

**This table is the main finding of the whole exercise.** Every flat-window arm has a crossover —
better than production on short text, worse on long. The hierarchical model is the first that is
level or ahead essentially everywhere, because it keeps chunking (full coverage) *and* fixed padded
input (works on short text). The only band where it trails is 2000+, by 0.08 pp on 3.2% of
documents, caused by truncation at `max_chunks=8` (1,536 tokens).

The byte arm's crossover sits at ~150 tokens against the token arm's ~500 — the ratio predicted by
4.68 bytes per token, i.e. its window covers ~328 tokens of text rather than 1,536.

## Where the difference lives

Source: `model_artifacts/magika_style/comparison-per-dataset-tokens.csv`, sorted by
hier-large minus production.

Production loses badly on **short human text**:

| dataset | n | production | magika flat | hier-large | hier-small | Δ |
|---|---|---|---|---|---|---|
| english_quotes | 501 | 0.6866 | 0.9641 | 0.9800 | 0.9800 | **+29.3** |
| rotten_tomatoes | 1,500 | 0.7607 | 0.9753 | 0.9907 | 0.9793 | **+23.0** |
| ag_news | 4,000 | 0.8763 | 0.9408 | 0.9578 | 0.9618 | +8.2 |
| General-Knowledge | 2,000 | 0.8335 | 0.9245 | 0.9035 | 0.8995 | +7.0 |
| human_vs_machine | 5,978 | 0.8904 | 0.9346 | 0.9344 | 0.9428 | +4.4 |
| Llama-3.2-1B | 1,000 | 0.9100 | 0.9460 | 0.9520 | 0.9470 | +4.2 |
| AI-and-Human-Generated-Text | 5,729 | 0.9297 | 0.9501 | 0.9571 | 0.9564 | +2.7 |
| raid | 30,000 | 0.9569 | 0.9686 | 0.9694 | 0.9626 | +1.3 |

Production keeps small edges on long-form prose, all under 1 pp:

| dataset | n | production | hier-large | Δ |
|---|---|---|---|---|
| Llama-3.1-8B | 2,000 | 0.9440 | 0.9340 | −1.0 |
| human_vs_ai_sentences | 3,000 | 0.8477 | 0.8393 | −0.8 |
| imdb | 2,000 | 0.9925 | 0.9890 | −0.4 |
| ivy panda essays | 6,093 | 0.9975 | 0.9954 | −0.2 |
| newswire | 10,000 | 0.9931 | 0.9922 | −0.1 |

Note `ai-vs-human-HuggingFaceTB-SmolLM2-360M` is the hardest source for every arm (0.7558 production,
0.8258 magika flat, 0.7818 hier-large) and `human_vs_ai_sentences` the second (0.82–0.85 across all).
Both are worth looking at directly.

## Per-run training records

Source: `results-*.json` in each artifact directory.

| run | params | train | epochs | acc | F1 | MCC | AUC |
|---|---|---|---|---|---|---|---|
| magika bytes w512 | 231,554 | 61 min | 27 | 0.9538 | 0.9558 | 0.9074 | 0.9914 |
| magika tokens w512 | 4,393,090 | 35 min | 11 | 0.9681 | 0.9695 | 0.9360 | 0.9948 |
| hier v32768-d128 | 4,394,755 | 49 min | 18 | 0.9714 | 0.9725 | 0.9427 | 0.9960 |
| hier v16384-d64 | 1,183,363 | 25 min | 12 | 0.9677 | 0.9690 | 0.9353 | 0.9951 |
| hier v16384-d64 + CutMix | 1,183,363 | 26 min | 12 | 0.9683 | 0.9695 | 0.9365 | 0.9947 |

All on MPS (Apple Silicon). Early stopping on validation AUC, patience 4.

### Overfitting

Every run drives training loss well past the point validation stops improving:

```
v32768-d128: best val_auc 0.9953 @ epoch 14; train loss 0.0316 -> 0.0236 after that
v16384-d64 : best val_auc 0.9947 @ epoch  8; train loss 0.0826 -> 0.0596 after that
```

The larger model fell 0.2815 → 0.0236 (12x) while validation AUC gained 0.008 total and was flat
from epoch 4. Optimizer is plain Adam, constant lr 1e-3, no weight decay, no schedule.

## Ablations

### Byte vs token encoding — confounded, and the confound is the finding

At the same 512-symbol window the two encodings do **not** see comparable amounts of text:

| arm | docs truncated | mean fraction of doc seen | docs fully seen |
|---|---|---|---|
| bytes | **47.8%** (56,573) | 0.791 | 52.2% |
| tokens | 5.5% (6,503) | 0.983 | 94.5% |

At 4.68 bytes per token the byte window reaches ~328 tokens against the token window's 1,536. So
`--encoding bytes` vs `tokens` measures *context length* as much as representation. The controlled
version — `--encoding tokens --window 328` — was never run.

Byte encoding is nonetheless the smallest artifact by a wide margin: **231,554 params / 0.89 MB**,
at 0.9538 accuracy, i.e. within 0.24 pp of production at 12.8x smaller.

### CutMix

Three variants of the small model, single runs each:

| variant | acc | F1 | MCC | AUC |
|---|---|---|---|---|
| no CutMix | 0.9677 | 0.9690 | 0.9353 | 0.9951 |
| CutMix, random pairing | **0.9699** | **0.9711** | **0.9398** | 0.9951 |
| CutMix, source-aware pairing | 0.9683 | 0.9695 | 0.9365 | 0.9947 |

Random pairing beat source-aware pairing, against expectation. The likely reading is that the gain
was **augmentation diversity** rather than chunk supervision — welding unrelated sources produces
more varied inputs, and constraining partners to the same source removes that.

**These three are not distinguishable.** Single runs, spread 0.22 pp, no measured noise floor.
`--seed` exists now; two repeats of one config would settle it.

Only the source-aware numbers survive on disk — the random-pairing run wrote to the same tag and was
overwritten. Its figures above come from the session transcript.

Source-aware pairing prefers an opposite-label partner from the same corpus. It can only reach so
far: **48.9% of training rows come from single-label sources** (newswire all human, arena-preference
all AI, and so on), which have no same-source opposite-label partner by construction and still fall
back cross-source.

## Corpus properties measured along the way

Facts about `data/3.0.0` that bear on any future model.

**Chunking, under the production chunker (150/15/30):**

- **164 distinct chunk sizes** between 1 and 164 tokens, despite a configured 150. Chunk size and
  overlap are both solved per document, so they vary with length.
- A **redundant trailing chunk on 31% of multi-chunk documents** (4,833 of 15,576 sampled) — always a
  strict subset of its predecessor, so the weighted mean double-counts the tail. It appears exactly
  when the solver saturates its `overlap ≤ 2×overlap` ceiling.
- Length-controlled, that redundancy costs production 0.1–0.7 pp per band. Real, consistent in
  direction, small.
- 22.6% of the test split is single-chunk (≤150 tokens), and production scores 0.8902 there against
  0.9753 on multi-chunk — an **8.5 pp** gap, and where its headline number is spent.

**Document length:** mean 508 tokens, p25 180, p50 316, p75 563, p95 1,613, max 37,118.

**Constant chunking, no overlap** (candidate scheme), by chunk size:

| chunk_size | chunks/doc | 1-chunk docs | final chunk <30% full | padding waste |
|---|---|---|---|---|
| 128 | 4.49 | 17.3% | 32.5% | 21.6% |
| 150 | 3.91 | 20.4% | 32.4% | 23.8% |
| **192** | **3.18** | 27.1% | 35.1% | 27.5% |
| 256 | 2.52 | 39.2% | 35.1% | 31.8% |
| 512 | 1.54 | 72.4% | 31.9% | 45.1% |

Roughly a third of documents end in a mostly-empty chunk at *every* chunk size — a consequence of the
length distribution, not a tunable. Chunk-count caps at 192 tokens: 8 covers 94.5% of documents
whole, 16 covers 98.8%, 32 covers 99.7%.

**Vocabulary**, o200k_base over the training split — 171,571 distinct ids:

| top-K by frequency | coverage | dim 64 | dim 128 |
|---|---|---|---|
| 4,096 | 74.5% | 1.0 MB | 2.0 MB |
| 8,192 | 81.9% | 2.0 MB | 4.0 MB |
| 16,384 | **88.7%** | 4.0 MB | 8.0 MB |
| 32,768 | 94.4% | 8.0 MB | 16.0 MB |

The embedding is 89–95% of the artifact; the trunk is only ~200k parameters. Size is almost entirely
a vocabulary and width decision.

**Markdown bias** (from `dataset_metadata.json`): markdown appears in **33.66% of AI samples against
5.97% of human** — 5.6x. Headings 0.991 vs 0.0003, bold-asterisk 4.22 vs 0.0026, code blocks 0.33 vs
0.0. `cleaner.rs:888` deliberately preserves markdown. How much of every number in this document is
formatting detection rather than style is **still unmeasured**.

## Methodology notes

- The published `AUC 0.9563` in `model_artifacts/3.0.0/model_metadata.json` is not an AUC.
  `train.py` passes hard 0/1 predictions to `roc_auc_score` at lines 428, 561, 650 and 689, making it
  balanced accuracy. Production's real AUC is **0.9889**.
- `train.py:410` fits the decision threshold on the test set it then reports. Every number here fits
  thresholds on validation instead. Production's shipped threshold is also reported for reference,
  and reproduces its published metrics to 4 decimal places (0.9562 / 0.9579 / 0.9123 against
  published 0.9562 / 0.9579 / 0.9124), which is what validates the reimplementation.
- ONNX exports are asserted against the source model on real documents, not merely printed:
  max drift 3.6e-07 to 5.4e-07 across runs.

## Hyperparameter campaign (2026-08-30)

One sweep over the small hierarchical model (`c192-k8-v16384-d64`), run through
`hierarchical_train.py` with the new experiment flags, full splits, thresholds fitted on validation.
Sources: each arm's `results.json` under `model_artifacts/hierarchical/<tag>/`. Arms were ranked on
validation AUC and the noise floor below; test numbers were read only for the finalists.

### Noise floor first

Three seeds of the unchanged baseline:

| seed | epochs | train s | val AUC | test acc | test MCC | test AUC |
|---|---|---|---|---|---|---|
| 42 | ~12 | 1511 | 0.9947 | 0.9677 | 0.9353 | 0.9951 |
| 43 | 24 | 2876 | 0.9951 | 0.9703 | 0.9404 | 0.9959 |
| 44 | 10 | 1186 | 0.9943 | 0.9676 | 0.9351 | 0.9950 |

Spread: **0.27 pp accuracy / 0.0008 val AUC**. Single-run deltas inside that band are not results.
The best seed is the one early stopping let train 2x longer — training duration, not init, is the
live variable, which the optimizer arm then confirmed. This band also retroactively classifies the
earlier CutMix deltas (0.9677 / 0.9699 / 0.9683) as noise.

### Wave 1 — one change at a time

| arm | epochs | train s | val AUC | test acc | test F1 | test MCC | test AUC |
|---|---|---|---|---|---|---|---|
| baseline (seed mean) | 10-24 | ~1900 | 0.9947 | 0.9685 | 0.9697 | 0.9369 | 0.9953 |
| AdamW wd 0.01 + cosine, patience 6 | 30 (full) | 3541 | 0.9951 | 0.9719 | 0.9731 | 0.9437 | 0.9961 |
| **group 2** | 16 | 3103 | **0.9959** | **0.9734** | **0.9745** | **0.9468** | **0.9966** |
| group 1 | 22 | 8653 | 0.9961 | 0.9725 | 0.9736 | 0.9450 | 0.9964 |
| mean+max chunk pool | 24 | 3365 | 0.9952 | 0.9712 | 0.9723 | 0.9422 | 0.9960 |
| chunk-position embedding | 19 | 2262 | 0.9949 | 0.9694 | 0.9707 | 0.9386 | 0.9957 |
| chunk-loss-weight 0 | 12 | 1449 | 0.9943 | 0.9661 | 0.9677 | 0.9322 | 0.9948 |

- **`group=2` is the single largest win** (+0.5 pp accuracy, outside noise) — NEXT_RELEASE's "most
  suspicious inherited constant" was the right suspect. `group=1` confirms the direction but loses
  to 2 on test at 2.8x the training cost: some grouping still helps, four BPE tokens was too many.
  Epoch cost rises 118 s -> 194 s (g2) -> 390 s (g1).
- **AdamW + cosine** is a clear win. The wave-1 cosine arm used its full 30-epoch budget (several
  combo arms still early-stopped under patience 6), and the schedule reliably trains past the
  10-24-epoch spread the constant-LR baseline showed across seeds.
- Mean+max pooling and the position embedding sit at the top of the noise band alone, and neither
  survived on top of the combo (below). Dropping the chunk loss lands at the bottom of the band:
  the chunk head earns its keep.

### Wave 2 — combining and shrinking

All arms carry `--group 2 --weight-decay 0.01 --schedule cosine --patience 6`:

| arm | ship MB | epochs | train s | val AUC | test acc | test F1 | test MCC | test AUC | rec@p>=.99 |
|---|---|---|---|---|---|---|---|---|---|
| **combo** (v16384-d64-g2-wd-cos) | 5.23 | 30 (full) | 5859 | 0.9961 | **0.9751** | **0.9760** | **0.9501** | 0.9968 | 0.9562 |
| combo, seed 43 | 5.23 | 29 | 5695 | 0.9960 | 0.9742 | 0.9753 | 0.9484 | 0.9968 | 0.9575 |
| combo + meanmax | 5.25 | 20 | 4632 | 0.9960 | 0.9732 | 0.9743 | 0.9464 | 0.9966 | 0.9575 |
| combo + pos-embed | 5.24 | 27 | 5343 | 0.9963 | 0.9749 | 0.9758 | 0.9496 | 0.9969 | 0.9588 |
| combo, max_chunks 16 | 5.23 | 30 (full) | 15549 | 0.9962 | 0.9749 | 0.9759 | 0.9497 | **0.9969** | 0.9584 |
| combo, d48 | 4.20 | 28 | 5345 | 0.9960 | 0.9741 | 0.9751 | 0.9481 | 0.9967 | 0.9560 |
| combo, v8192 | 3.23 | 30 (full) | 5929 | 0.9960 | 0.9730 | 0.9741 | 0.9460 | 0.9966 | 0.9543 |
| combo, v8192 + d48 | **2.70** | 29 | 5550 | 0.9960 | 0.9732 | 0.9743 | 0.9462 | 0.9965 | 0.9515 |

Ship MB = `slop-classifier.onnx` + `token_vocabulary.safetensors`, the files inference needs.
rec@p>=.99 = test recall at the >=99%-precision operating point (production: 0.9120).

- The two wave-1 wins **stack**: combo seed spread is 0.09 pp (0.9742/0.9751), comfortably above
  everything else. Combo, combo+pos and k16 are statistically indistinguishable — combo is the pick
  on simplicity, not metrics.
- **meanmax and pos-embed add nothing on top of the combo.** Rejected.
- **max_chunks 16 buys only the 2000+-token tail** (0.9984 -> 0.9992 on 3.2% of documents) at 2.7x
  the training cost; headline is a tie. k=32 was skipped — that band has nothing left to give.
- **Size**: d48 is nearly free (-0.10 pp, inside the combo band's edge); vocabulary cuts cost more
  than width cuts. The stacked cut lands at 2.70 MB for -0.19 pp.

### Accuracy by document length, finalists

| tokens | n | production | combo | combo k16 | combo d48 | combo v8192+d48 |
|---|---|---|---|---|---|---|
| 0-50 | 9,685 | 0.8293 | 0.9183 | 0.9173 | 0.9154 | 0.9137 |
| 50-100 | 7,013 | 0.9092 | 0.9561 | 0.9577 | 0.9542 | 0.9549 |
| 100-150 | 7,309 | 0.9325 | 0.9590 | 0.9611 | 0.9572 | 0.9583 |
| 200-300 | 22,179 | — | 0.9732 | 0.9717 | 0.9729 | 0.9696 |
| 500-1000 | 19,962 | — | 0.9923 | 0.9926 | 0.9914 | 0.9922 |
| 2000-inf | 3,837 | — | 0.9984 | 0.9992 | 0.9979 | 0.9990 |

The short-text weakness that motivated the redesign is +8.9 pp over production at 0-50 tokens.

### Recommended configurations

The training recipe is a tuple, not independent knobs — with a cosine schedule, `epochs` shapes the
LR curve rather than merely capping it. Recipe for every pick:
`--group 2 --weight-decay 0.01 --schedule cosine --patience 6 --epochs 30 --lr 1e-3 --batch-size 128`
(one epoch linear warmup is built into the schedule).

| axis | config | ship MB | test acc | test MCC |
|---|---|---|---|---|
| accuracy | `c192-k8-v16384-d64` + recipe | 5.23 | 0.9751 | 0.9501 |
| balanced | `c192-k8-v16384-d48` + recipe | 4.20 | 0.9741 | 0.9481 |
| minimal | `c192-k8-v8192-d48` + recipe | 2.70 | 0.9732 | 0.9462 |

Versus shipped production (11.41 MB, 0.9562 acc, 0.9120 recall@p99): the minimal model is **4.2x
smaller and +1.7 pp more accurate**; the accuracy pick is 2.2x smaller and +1.9 pp.

Follow-up arms, run after the tables above were first drafted:

- **d48 seed repeats**: 0.9741 / 0.9745 / 0.9749 across seeds 42/43/44 — mean 0.9745, spread
  0.08 pp, statistically identical to the d64 combo. **d48 gives up nothing**; the balanced pick is
  a legitimate default.
- **45-epoch cosine horizon** (`--epochs 45 --patience 8`): early-stopped at 33, test 0.9741 acc /
  0.9482 MCC — inside the combo band. The longer schedule does not beat 30. **Freeze epochs=30.**
- **Single-thread CPU timing**, 300 test documents, real chunks only, idle machine (same
  methodology as the Cost table above):

| arm | ms/doc | vs production (2.74) |
|---|---|---|
| baseline d64 (group 4) | 0.244 | 11.2x |
| combo (group 2) | 0.427 | 6.4x |
| combo d48 | 0.422 | 6.5x |
| combo v8192+d48 | 0.421 | 6.5x |

`group=2` costs 1.75x inference over the old group-4 graph — the one config axis where speed moved.
The size cuts are speed-neutral (the trunk dominates). All picks remain >6x faster than production.

### Still open after this campaign

Everything here inherits the known caveats: the live cleaning skew (curation cleaned, training raw,
inference universal), the unmeasured markdown-formatting share of the signal, and uncalibrated
output probabilities. CutMix remains undecided pending the chunk-output contract question — its
document-level deltas are now known to be inside the noise floor.

## Burn training port: parity run (2026-08-31)

The Python trainer was ported to Rust on Burn 0.21 (`crates/is-it-slop-training`, branch
`sbroomfield/agent/hparam-sweep/burn-training-crate`). Three golden-vector oracles pin the port to
the Python pipeline before any training judgement:

| oracle | result |
|---|---|
| model forward (trained v8192-d48 weights, 16 real docs) | logits within 1e-3, probabilities within 1e-5 of PyTorch |
| document encoding (trim/tokenize/vocab/chunk) | bit-identical grids to the Python cache |
| vocabulary build (full 473k-doc train split) | byte-identical lookup table to the Python bundle |

Plus unit oracles: the warmup-cosine schedule matches `LambdaLR` to 1e-15, the masked loss matches
torch to 1e-6, and the AUC implementation matches sklearn including ties.

### The parity gate

One full run of the frozen recipe (accuracy config v16384-d64, seed 42) on the Metal backend,
`model_artifacts/burn/parity-run.log`:

| | val AUC (full split) | test AUC (full split) | epochs |
|---|---|---|---|
| Python combo, seed 42 | 0.9961 | 0.9968 | 30 (full) |
| Python combo, seed 43 | 0.9960 | 0.9968 | 29 |
| **Burn, seed 42** | **0.9960** | **0.9967** | 23 (early stop) |

Both deltas are ~0.0001 — an order of magnitude inside the seed-noise band. The port's three
documented loop deviations (early stopping on mean per-batch AUROC rather than full-split AUC,
final rather than best epoch weights, no `drop_last`) measurably do not matter. The per-epoch
validation trajectories also track each other closely (0.993 at epoch 4 to 0.996 by epoch 16 on
both sides).

### Training throughput, the one regression

~7.3 min/epoch on Burn-Metal against 3.2 min/epoch on torch-MPS (~2.3x), same machine, autotune
cache warm, fusion enabled. Diagnosis (not yet profiled): torch dispatches to Apple's hand-tuned
MPSGraph kernels, while Burn JIT-compiles its own via CubeCL — a gap that bites hardest on this
model's shape, many small ops around thin matmuls under eager autodiff. Investigation plan lives
in NEXT_RELEASE.md; training cost is a per-release one-off and no shipping axis is affected.

Ruled out: fusion hygiene. Restructuring `encode_chunks` so every mask is derived before the trunk
(one uninterrupted compute region) changed nothing — 114.8 vs 115.2 s/1000 iterations, within
noise. The fusion boundaries were not where the 2.3x lives.
