# is-it-slop-curation

Builds the human/AI text dataset the model is trained and evaluated on, from the public sources
listed in [`src/curation/sources.toml`](src/curation/sources.toml).

```sh
just curation::build 4.0.0       # data/4.0.0/{train,test,validation}.parquet + dataset_metadata.json
just curation::sources           # every source and whether it can be read (credentials)
just curation::test              # offline tests
```

`just curation::build --only raid --limit-per-source 5000` is a quick trial run. Output goes to
`<data-dir>/<version>` (`--data-dir`, default `./data`).

## Credentials

Read from the environment or a `.env` file in the working directory (git-ignored):

| Variable | Needed for |
| --- | --- |
| `HF_TOKEN` | `human_vs_machine` (a gated Hugging Face dataset) |
| `KAGGLE_API_TOKEN` | `daigt-v2-train-dataset` |

A source whose credentials are missing is skipped with a message; `--strict` fails instead.

## How it works

1. Every source is plain files (parquet, csv, ndjson, JSON arrays) read lazily into one schema:
   `text`, `label` (0 human, 1 AI), `group`, `generator`, `domain`, `attack`.
2. Rows are picked *before* any text is held in memory: a first pass hashes each row and ranks
   rows within each label, and the quotas in `sources.toml` slice that ranking into train / test /
   validation. Rows that share a `group` (variants of one source document) rank together and so
   land in the same split.
3. Only the picked rows are materialised and cleaned with the Rust training cleaner
   (`is_it_slop.preprocessing`).
4. Exact duplicates, label conflicts and normalised or shared-prefix overlap between splits are
   removed with joins; `dataset_metadata.json` records what each guard removed.

Selection is deterministic for a given `sources.toml`, seed and polars version. polars is pinned exactly because
`hash()`, which ranks the rows, is only guaranteed stable within one version: upgrading it can pick different rows,
so `dataset_metadata.json` records the version used.
