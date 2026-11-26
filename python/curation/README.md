# curation

Builds the human/AI text dataset the model is trained and evaluated on, from the public sources
listed in [`src/curation/sources.toml`](src/curation/sources.toml).

```sh
just curation::build 5.1.0       # data/5.1.0/{train,test,validation,holdout}.parquet + dataset_metadata.json
just curation::sources           # every source and whether it can be read (credentials)
just curation::test              # offline tests
```

`just curation::build 5.0.0 --only raid --limit-per-source 5000` is a quick trial run. Output goes to
`<data-dir>/<version>` (`--data-dir`, default `./data`).

## Credentials

Read from the environment or a `.env` file in the working directory (git-ignored):

| Variable | Needed for |
| --- | --- |
| `HF_TOKEN` | `human_vs_machine-*` (a gated Hugging Face dataset) |
| `KAGGLE_API_TOKEN` | `daigt-v2-train-dataset` |

A source whose credentials are missing is skipped, and the build ends by saying the dataset is
incomplete and which variable unlocks which source; `--strict` fails instead.

## How it works

1. Every source is plain files (parquet, csv, ndjson, JSON arrays, gzipped ndjson) read lazily into one schema:
   `text`, `label` (0 human, 1 AI), `group`, `generator`, `domain`, `attack`. A source's repairs
   (below) are applied here, before anything is hashed.
2. Rows are picked *before* any text is held in memory: a first pass hashes each row and ranks
   rows within each label. `take` says how many rows of the source to use; `[settings]` cuts them into
   train / test / validation (70 / 15 / 15) **in the same proportions for every source**, so each
   split has the same mix of sources, domains and generators. Rows that share a `group` (variants of one
   source document, the answers to one question) rank together and land in the same split.
3. Only the picked rows are materialised and cleaned with the Rust training cleaner
   (`is_it_slop_preprocessing`).
4. Exact duplicates, label conflicts and normalised or shared-prefix overlap between splits are
   removed with joins; `dataset_metadata.json` records what each guard removed.

### Splits and holdout

Every output file has a `split` column, so the four files can be concatenated into one table and cut again.
`train`, `test` and `validation` are 70/15/15 of every source and label. The `holdout` is never trained on or
tuned against, and comes from two places:

- a source with `holdout = true` goes there whole (small benchmark sets with named current models);
- a source with `holdout_each = n` reserves `n` rows of each label (at most 10% of the source) before the
  rest is split, so the holdout has a small, varied slice of every domain and generator.

The overlap guards treat the holdout as the last split, so a holdout row that repeats or closely resembles
any training row is dropped.

### Balancing formatting

`--balance RATIO` (or `balance = RATIO` under `[settings]`) compares human and AI rows inside every
combination ("cell") of length bin, markdown, paragraph breaks, non-ASCII and em dashes. It keeps the
minority class whole and cuts the majority class to `RATIO` times as many rows. The dropped rows are
listed per source in `dataset_metadata.json`. The holdout is never balanced.

### Repairs

Sources are repaired for *how they were collected*, never for what their authors wrote:

| Option | Fixes |
| --- | --- |
| `unwrap_lines` | hard-wrapped text (arXiv abstracts are wrapped at about 80 characters) |
| `detokenise` | pre-tokenised text (`it 's`, `word .`, `` `` '' `` quotes) |
| `normalise_breaks` | one class marks paragraphs with a single newline and the other with two |
| `ascii_punctuation` | curly quotes, en/em dashes and ellipses kept by only one side of a source |
| `remove_pattern` | boilerplate characters (applied before `detokenise`: removing one can leave a gap) |
| `trim_fragments` | text that begins or ends mid-sentence (also re-applied after the Rust cleaner) |
| `flatten_paragraphs` | a source whose human text has no paragraph breaks while its AI text does, only because of collection |
| `strip_think` | a leading `<think>...</think>` reasoning trace |
| `drop_pattern` | contamination: rows matching a regex are dropped |
| `min_chars` | fragments |

### The format-leak report

Features that say "AI" without being about the writing (paragraph breaks, markdown, em dashes, curly
quotes, non-ASCII, spaced punctuation, a lowercase start, the final punctuation) are measured on every
build: `format_leaks` in `dataset_metadata.json` gives each feature's rate per class and its AUC as a
one-feature classifier, and lists every source in which human and AI text differ by 35 points or more.
Inside one source the writing should be the only difference, so a flagged feature is a collection
artefact the model could learn instead. The cleaner (`is_it_slop_preprocessing`) already removes some
formatting; what it leaves is what this report shows.

## Adding a source

1. Human text must be *certainly* human: before late 2022 (ChatGPT), or human by construction. Text from
   later needs a filter or a reason and its own source, so it can be dropped on its own. Be more suspicious
   of newer datasets: most of them are AI.
2. Prefer raw single-class sources (all human, or all AI) over mixed ones: you know where they come from.
   If a mixed source repeats one of them, the raw one wins.
3. Run `just curation::build x --only <source> --limit-per-source 5000`, then look at the
   `format_leaks` section of the metadata for artefacts and fix them with a repair or drop the source.

Selection is deterministic for a given `sources.toml`, seed and polars version. polars is pinned exactly because
`hash()`, which ranks the rows, is only guaranteed stable within one version: upgrading it can pick different rows,
so `dataset_metadata.json` records the version used.
