#!/usr/bin/env python
# coding: utf-8

# # Text Curation
#
# Here we are trying to document and curate a really good collection of texts that represent human and AI generated texts.
#
# ## Labels
# 0 - Human generated texts
# 1 - AI generated texts

# In[ ]:


import json
from datetime import UTC, datetime

import kagglehub
import numpy as np
import polars as pl
import polars.selectors as cs
from __init__ import DATA_DIR, DATA_PATH, RETRAINED_MODEL_VERSION, SEED, TEST_PATH, TRAIN_PATH, VALIDATION_PATH
from datasets import load_dataset
from is_it_slop_preprocessing import CleaningMode, TextCleaner, __version__, tokenize
from kagglehub import KaggleDatasetAdapter
from loguru import logger

print(f"Bindings version: {__version__}")
print(f"Pipeline model version output: {RETRAINED_MODEL_VERSION}")

cleaner = TextCleaner(mode=CleaningMode.TRAINING)


def clean_text_inner(series: pl.Series) -> pl.Series:
    return pl.Series(series.name, cleaner.clean_batch(series.to_list()))


def clean_text(df: pl.LazyFrame, text_col: str = "text") -> pl.LazyFrame:

    return (
        df
        .with_columns(pl.col(text_col).map_batches(clean_text_inner, return_dtype=pl.Utf8))
        .filter(pl.col(text_col).is_not_null())
        .filter(pl.col(text_col).str.len_chars() > 0)
    )


def load_normal(
    dataset_name: str,
    rename: dict[str, str] | None = None,
    *,
    subset_name: str | None = None,
    clean: bool = True,
    drop_nulls: bool = True,
    file_name: str | None = None,
) -> pl.LazyFrame:

    if file_name is None:
        ds = load_dataset(dataset_name, name=subset_name)
        lf = pl.concat(
            ds[split].to_polars()  # type: ignore[attr-defined]
            for split in ds
        )
    else:
        lf: pl.LazyFrame = kagglehub.dataset_load(KaggleDatasetAdapter.POLARS, dataset_name, file_name).lazy()
    dataset_name = dataset_name.rsplit("/", maxsplit=1)[-1] + (f"/{subset_name}" if subset_name else "")
    if rename:
        lf = lf.rename(rename)
    if drop_nulls:
        lf = lf.drop_nulls(subset="text")
    if clean:
        lf = lf.pipe(clean_text)
    return lf.with_columns(cs.by_dtype(pl.Utf8).str.strip_chars(), dataset=pl.lit(dataset_name)).lazy()


def get_value_counts(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(pl.col("label").value_counts()).unnest("label").sort("label")


def output_value_counts(df: pl.LazyFrame) -> list[dict]:
    return df.pipe(get_value_counts).collect().to_dicts()


# # [English Quotes dataset](https://huggingface.co/datasets/Abirate/english_quotes)

# In[ ]:


english_quotes = load_normal("Abirate/english_quotes", {"quote": "text"}).select(
    pl.col("text"), pl.col("dataset"), label=pl.lit(0, dtype=pl.Int8)
)
logger.info("Loaded English quotes")
print(output_value_counts(english_quotes))


# # [Newswire dataset](https://huggingface.co/datasets/dell-research-harvard/newswire)
#
# Number of rows: 1,440,010
# Likes: 85
# Downloads last month: 8,719
#
#
# >Assertion: Using newswire articles would provide a rich source of human-generated text.

# In[ ]:


newswire = (
    load_normal("dell-research-harvard/newswire", {"cleaned_article": "text"})
    .select("text", "dataset")
    .with_columns(label=pl.lit(0, dtype=pl.Int8))
)
# newswire.head(5).collect()
logger.info("Loaded newswire articles")
print(output_value_counts(newswire))


# # [rotten_tomatoes dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
#
# Number of rows 10,662
# Likes: 93
# Downloads last month: 60,583
#
# > Assertion: Using movie reviews would provide a rich source of human-generated text (may be help with shorter texts)
# > Assumption: Movie reviews are more likely to be human generated than AI generated.
#

# In[ ]:


rt = load_normal("cornell-movie-review-data/rotten_tomatoes").with_columns(label=pl.lit(0, dtype=pl.Int8))
# rt.head(5).collect()
logger.info("Loaded Rotten Tomatoes reviews")
print(output_value_counts(rt))


# # [ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)
#
# Number of rows: 127,600
# Likes: 177
# Downloads last month: 84,165

# In[ ]:


ag = load_normal("fancyzhx/ag_news").with_columns(label=pl.lit(0, dtype=pl.Int8))
logger.info("Loaded AG News articles")
print(output_value_counts(ag))


# # [Imdb dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
#
# Number of rows: 100,000
# Likes: 352
# Downloads last month: 171,036

# In[ ]:


imdb = load_normal("stanfordnlp/imdb").with_columns(label=pl.lit(0, dtype=pl.Int8))
logger.info("Loaded IMDB reviews")
print(output_value_counts(imdb))


# # [AI-human-text](https://huggingface.co/datasets/andythetechnerd03/AI-human-text)
#
# Number of rows: 487,235
# Likes: 8
# Downloads last month: 365

# In[ ]:


ai_human = load_normal("andythetechnerd03/AI-human-text").rename({"generated": "label"})
logger.info("Loaded AI vs Human text")
print(output_value_counts(ai_human))


# # [Human vs Machine](https://huggingface.co/datasets/NicolaiSivesind/human-vs-machine)
#
# Number of rows: 320,000
# Likes: 19
# Downloads last month: 188

# In[ ]:


from huggingface_hub import hf_hub_download

wiki_path = hf_hub_download(
    repo_id="NicolaiSivesind/human-vs-machine", filename="wiki-labeled.csv", repo_type="dataset"
)
abstracts_path = hf_hub_download(
    repo_id="NicolaiSivesind/human-vs-machine", filename="research-abstracts-labeled.csv", repo_type="dataset"
)


human_vs_machine = (
    pl
    .concat([pl.scan_csv(wiki_path), pl.scan_csv(abstracts_path)])
    .with_columns(cs.by_dtype(pl.Utf8).str.strip_chars(), dataset=pl.lit("human_vs_machine"))
    .drop(["title", "word_count"])
    .pipe(clean_text)
    .cast({"label": pl.Int8})
)
logger.info("Loaded Human vs Machine text")
print(output_value_counts(human_vs_machine))


# # [AI-and-Human-Generated-Text](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text)
#
# Number of rows: 28,662
# Likes: 19
# Downloads last month: 486

# In[ ]:


ai_and_human = (
    load_normal("Ateeqq/AI-and-Human-Generated-Text", {"abstract": "text"}).cast({"label": pl.Int8}).drop("title")
)
logger.info("Loaded AI vs Human text from Ateeqq/AI-and-Human-Generated-Text")
print(output_value_counts(ai_and_human))


# # [AI generated movie reviews](https://huggingface.co/datasets/Milkyway-islander/AI_Human_generated_movie_reviews)
#
# Number of rows: 10,460
# Likes: 3
# Downloads last month: 29
#
# There are a good verity of AI models used to generate these texts.

# In[ ]:


ai_movie_reviews = (
    load_normal("Milkyway-islander/AI_Human_generated_movie_reviews")
    .rename({"labels": "label"})
    .cast({"label": pl.Int8})
    .drop("__index_level_0__")
)
logger.info("Loaded AI vs Human movie reviews")
print(output_value_counts(ai_movie_reviews))


# # [Human vs AI Sentences](https://huggingface.co/datasets/shahxeebhassan/human_vs_ai_sentences)
#
# Number of rows: 105,000
# Likes: 9
# Downloads last month: 151

# In[ ]:


human_vs_ai_sentences = load_normal("shahxeebhassan/human_vs_ai_sentences").cast({"label": pl.Int8})
logger.info("Loaded Human vs AI sentences")
print(output_value_counts(human_vs_ai_sentences))


# # [Human Raid](https://huggingface.co/datasets/charisgao/human-raid)
#
# Number of rows: 948,371
# Likes: 1
# Downloads last month: 10
#
# Unsure about this one as it seems to be data taken from diffrent sources `reddit`, `recipes`, `reviews` which could quite easily be AI generated

# In[ ]:


# %%script true
# human_raid = (
#     load_normal("charisgao/human-raid")
#     .rename({"generation": "text"})
#     .with_columns(label=pl.lit(0, dtype=pl.Int8))
#     .select(["domain", "text", "label"])
# )
# human_raid.head(5).collect()


# # [AI-vs-human collection](https://huggingface.co/collections/zcamz/ai-vs-human)
#
# Number of rows: 5,000 (but  its ai-human cols so its 10,000 rows when expanded)
# Likes: 1
# Downloads last month: 5

# In[ ]:


def load_ai_vs_human_collection(dataset_name: str) -> pl.LazyFrame:
    ds = load_normal(dataset_name, clean=False, drop_nulls=False)
    dataset_name = dataset_name.rsplit("/", maxsplit=1)[-1]
    return (
        ds
        .rename({"ai": "1", "human": "0"})
        .select(["1", "0"])
        .unpivot()
        .rename({"variable": "label", "value": "text"})
        .with_columns(dataset=pl.lit(dataset_name))
        .cast({"label": pl.Int8})
        .pipe(clean_text)
        .drop_nulls(subset="text")
    )


# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
#
# ## Dataset Description
# This dataset contains pairs of original articles and their AI-generated completions.
#
#

# In[ ]:


ai_vs_human_gpt35t = load_ai_vs_human_collection("ilyasoulk/ai-vs-human")
logger.info("Loaded AI vs Human GPT-3.5-Turbo text")
print(output_value_counts(ai_vs_human_gpt35t))


#
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
#

# In[ ]:


ai_vs_human_smolLM2 = load_ai_vs_human_collection("zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-360M-Instruct")  # noqa: N816
logger.info("Loaded AI vs Human SmolLM2 text")
print(output_value_counts(ai_vs_human_smolLM2))


#
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
#

# In[ ]:


ai_vs_human_smolLM2_1_7B = load_ai_vs_human_collection("zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-1.7B-Instruct")  # noqa: N816
logger.info("Loaded AI vs Human SmolLM2 1.7B text")
print(output_value_counts(ai_vs_human_smolLM2_1_7B))


#
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
#

# In[ ]:


ai_vs_human_qwen = load_ai_vs_human_collection("zcamz/ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct")
logger.info("Loaded AI vs Human Qwen text")
print(output_value_counts(ai_vs_human_qwen))


#
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
#

# In[ ]:


ai_vs_human_gemma = load_ai_vs_human_collection("zcamz/ai-vs-human-google-gemma-2-2b-it")
logger.info("Loaded AI vs Human Gemma text")
print(output_value_counts(ai_vs_human_gemma))


#
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
#

# In[ ]:


ai_vs_human_llama = load_ai_vs_human_collection("zcamz/ai-vs-human-meta-llama-Llama-3.2-1B-Instruct")
logger.info("Loaded AI vs Human Llama text")
print(output_value_counts(ai_vs_human_llama))


#
# # AI vs Human dataset on the [OpenWebTxt](https://huggingface.co/datasets/stas/openwebtext-10k)
#

# In[ ]:


ai_vs_human_llama_8B = load_ai_vs_human_collection("ilyasoulk/ai-vs-human-meta-llama-Llama-3.1-8B-Instruct")  # noqa: N816
logger.info("Loaded AI vs Human Llama 8B text")
print(output_value_counts(ai_vs_human_llama_8B))


# ## [LM Arena Search](https://huggingface.co/datasets/lmarena-ai/search-arena-24k)
#

# In[ ]:


def load_lm_arena(dataset_name: str, *, clean: bool = False) -> pl.LazyFrame:
    ds_name = dataset_name.rsplit("/", maxsplit=1)[-1]

    lf = (
        load_normal(dataset_name, clean=False, drop_nulls=False)
        .rename({"messages_a": "conversation_a", "messages_b": "conversation_b"}, strict=False)
        .select(pl.col("conversation_a"), pl.col("conversation_b"))
        .unpivot()
        .rename({"value": "text"})
        .drop("variable")
        .explode("text")
        .unnest("text")
        .rename({"content": "text"})
        .filter(pl.col("role") == "assistant")
        .select("text")
        .with_columns(label=pl.lit(1, dtype=pl.Int8), dataset=pl.lit(ds_name))
        .drop_nulls(subset="text")
    )
    if clean:
        lf = lf.pipe(clean_text)
    return lf


# In[ ]:


search_arena = load_lm_arena("lmarena-ai/search-arena-24k")
logger.info("Loaded Search Arena text")
print(output_value_counts(search_arena))


# # [Arena Expert 5k](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k)

# In[ ]:


import ast
import contextlib
import re


def _find_matching(s: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    i = start
    in_str = False
    str_char = ""
    esc = False
    while i < len(s):
        ch = s[i]
        if ch == "\\" and not esc:
            esc = True
            i += 1
            continue
        if not in_str and ch in {"'", '"'}:
            in_str = True
            str_char = ch
        elif in_str and ch == str_char and not esc:
            in_str = False
        elif not in_str:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        esc = False
        i += 1
    return -1


def _replace_array_instances(s: str) -> str:
    out = []
    i = 0
    while True:
        m = s.find("array(", i)
        if m == -1:
            out.append(s[i:])
            break
        out.append(s[i:m])
        # find matching closing ')'
        close = _find_matching(s, m, "(", ")")
        if close == -1:
            # cannot find, bail out: append rest and break
            out.append(s[m:])
            break
        inner = s[m + 6 : close]  # content inside array(...)
        # strip trailing dtype=... if present
        inner = re.sub(r"\s*,\s*dtype\s*=\s*[^)\]]+\s*$", "", inner)
        # convert to list if it already uses [ ... ] keep as-is; otherwise wrap
        if inner.strip().startswith("["):
            out.append(inner)
        else:
            out.append("[" + inner + "]")
        i = close + 1
    return "".join(out)


def _extract_top_level_dicts(s: str) -> list[str]:
    objs = []
    i = 0
    while True:
        j = s.find("{", i)
        if j == -1:
            break
        k = _find_matching(s, j, "{", "}")
        if k == -1:
            break
        objs.append(s[j : k + 1])
        i = k + 1
    return objs


def _np_to_py(x: object) -> object:
    if isinstance(x, np.ndarray):
        return _np_to_py(x.tolist())
    if isinstance(x, list):
        return [_np_to_py(v) for v in x]
    if isinstance(x, dict):
        return {k: _np_to_py(v) for k, v in x.items()}
    return x


def _ensure_num_tokens(obj: object) -> object:
    if isinstance(obj, dict):
        obj.setdefault("num_tokens", None)
    elif isinstance(obj, list):
        for el in obj:
            _ensure_num_tokens(el)
    return obj


def _parse_batch(series: pl.Series) -> pl.Series:
    out = []
    for s in series:
        if s is None:
            out.append(None)
            continue
        s0 = s.strip()
        with contextlib.suppress(Exception):
            val = eval(s0, {"np": np})  # noqa: S307
            val = _np_to_py(val)
            out.append(_ensure_num_tokens(val))
            continue

        s_proc = _replace_array_instances(s0)
        dicts = _extract_top_level_dicts(s_proc)
        if dicts:
            parsed = []
            for d in dicts:
                with contextlib.suppress(Exception):
                    parsed.append(ast.literal_eval(d))
                    continue
                try:
                    parsed.append(eval(d, {"np": np}))  # noqa: S307
                except Exception:  # noqa: BLE001
                    parsed.append(None)
            parsed = _np_to_py(parsed)
            out.append(_ensure_num_tokens(parsed))
            continue

        if not s_proc.startswith("["):
            s_proc = "[" + s_proc + "]"
        try:
            val = ast.literal_eval(s_proc)
            out.append(_ensure_num_tokens(_np_to_py(val)))
            continue
        except Exception:  # noqa: BLE001
            try:
                val = eval(re.sub(r"\barray\(", "np.array(", s0), {"np": np})  # noqa: S307
                out.append(_ensure_num_tokens(_np_to_py(val)))
            except Exception:  # noqa: BLE001
                out.append(None)
    # return a plain python list for map_batches
    return pl.Series(out)


expert_arena = (
    load_normal("lmarena-ai/arena-expert-5k", clean=False, drop_nulls=False)
    .rename({"messages_a": "conversation_a", "messages_b": "conversation_b"}, strict=False)
    .select(
        # parse each conversation column separately so map_batches receives one column at a time
        pl.col("conversation_a").map_batches(
            _parse_batch,
            return_dtype=pl.List(
                pl.Struct([
                    pl.Field("role", pl.Utf8),
                    pl.Field(
                        "content",
                        pl.List(
                            pl.Struct([
                                pl.Field("type", pl.Utf8),
                                pl.Field("text", pl.Utf8),
                                pl.Field("image", pl.Utf8),
                                pl.Field("mimeType", pl.Utf8),
                            ])
                        ),
                    ),
                    pl.Field("num_tokens", pl.Float64),  # added optional field
                ])
            ),
        ),
        pl.col("conversation_b").map_batches(
            _parse_batch,
            return_dtype=pl.List(
                pl.Struct([
                    pl.Field("role", pl.Utf8),
                    pl.Field(
                        "content",
                        pl.List(
                            pl.Struct([
                                pl.Field("type", pl.Utf8),
                                pl.Field("text", pl.Utf8),
                                pl.Field("image", pl.Utf8),
                                pl.Field("mimeType", pl.Utf8),
                            ])
                        ),
                    ),
                    pl.Field("num_tokens", pl.Float64),  # added optional field
                ])
            ),
        ),
    )
    .unpivot()
    .rename({"value": "text"})
    .drop("variable")
    .explode("text")
    .unnest("text")
    .filter(pl.col("role") == "assistant")
    .select("content")
    .explode("content")
    .unnest("content")
    .filter(pl.col("type") == "text")
    .select(pl.col("text"))
    .pipe(clean_text)
    .with_columns(label=pl.lit(1, dtype=pl.Int8), dataset=pl.lit("arena-expert-5k"))
    .drop_nulls(subset="text")
)


# ## [LM Arena human prefrence](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
#
#

# In[ ]:


human_preference_140k = (
    load_lm_arena("lmarena-ai/arena-human-preference-140k", clean=False)
    .select(pl.col("text"))
    .explode("text")
    .unnest("text")
    .filter(pl.col("type") == "text")
    .select(pl.col("text"))
    .pipe(clean_text)
    .with_columns(label=pl.lit(1, dtype=pl.Int8), dataset=pl.lit("arena-human-preference-140k"))
)
logger.info("Loaded Human Preference 140k text")
print(output_value_counts(human_preference_140k))


# # [Human Essays](https://huggingface.co/datasets/artfultom/human-essays)

# In[ ]:


human_essays = (
    pl
    .concat([
        load_normal("artfultom/human-essays", subset_name="asap2 essays"),
        load_normal("artfultom/human-essays", subset_name="ivy panda essays"),
        load_normal("artfultom/human-essays", subset_name="persuade essays"),
    ])
    .with_columns(label=pl.lit(0, dtype=pl.Int8))
    .select(["text", "dataset", "label"])
)
logger.info("Loaded human essays")
print(output_value_counts(human_essays))


# # [LLM Generated Essays](https://huggingface.co/datasets/artfultom/llm-generated-essays)
#
# 37,488 records

# In[ ]:


llm_essays = pl.concat([
    load_normal("artfultom/llm-generated-essays", subset_name="one prompt").select("text", "dataset", "model"),
    load_normal("artfultom/llm-generated-essays", subset_name="two prompts").select("text", "dataset", "model"),
]).with_columns(label=pl.lit(1, dtype=pl.Int8))
logger.info("Loaded LLM-generated essays")
print(output_value_counts(llm_essays))


# In[ ]:


general_knowledge = (
    load_normal("MuskumPillerum/General-Knowledge", {"Answer": "text"})
    .with_columns(label=pl.lit(1, dtype=pl.Int8))
    .select(["text", "dataset", "label"])
)
logger.info("Loaded general knowledge questions and answers")
print(output_value_counts(general_knowledge))


# In[ ]:


daigt_v2 = (
    load_normal("thedrcat/daigt-v2-train-dataset", file_name="train_v2_drcat_02.csv", clean=False)
).with_columns(pl.col("label").cast(pl.Int8))
logger.info("Loaded DAIGT v2 dataset")
print(output_value_counts(daigt_v2))


# # [Raid Bench](https://huggingface.co/datasets/liamdugan/raid)

# In[ ]:


from polars_splitters import sample

df: pl.LazyFrame = load_dataset("liamdugan/raid")["train"].to_polars().lazy()  # type: ignore[reportAttributeAccessIssue]

human, ai = (
    df
    .select("model", "attack", "domain", "generation")
    .rename({"generation": "text"})
    .with_columns(pl.when(pl.col("model") == "human").then(0).otherwise(1).alias("label").cast(pl.Int8))
    .filter(pl.col("model").is_in(["human", "gpt4", "cohere-chat", "chatgpt", "gpt3", "llama-chat", "mistral-chat"]))
    .collect()
    .partition_by("label")
)

target_rows = human.height + 1
ai_rows = ai.height
fraction = target_rows / ai_rows

ai = sample(ai, fraction=fraction, stratify_by=["domain", "model", "attack"], seed=SEED).sample(target_rows, seed=SEED)
raid = (
    pl
    .concat([human.lazy(), ai.lazy()])
    .select(["text", "label"])
    .pipe(clean_text)
    .with_columns(dataset=pl.lit("raid"))
    .drop_nulls(subset="text")
)
logger.info("Loaded Raid Bench text")
print(output_value_counts(raid))


# ## AI vs Human
#
# 7 datasets containing 10,000 samples, 5,000 human-written and 5,000 AI-generated.
#
# This equates to a total of 70,000 samples.
#
#
# ## Human Raid
#
# I worry about the quality of this dataset so will ignore.
#
# ## AI-vs-Human Sentences
#
# This contains 105,000 sentences where half are human written and half are AI generated.
#
# ## AI generated movie reviews
#
# 5.23k AI generated movie reviews and 5.23k human written reviews from the Stanford IMDB dataset.
#
# ## AI-and-Human-Generated-Text
#
# 28,662 samples of abstracts and titles, half generated by AI (using GPT-3) and half original.
#
# ## Human vs Machine
#
# Older but contains 320,000 samples of human produced and machine generated text from Wikipedia introductions and scientific research abstracts.
# 50/50 split.
#
# ## AI-human-text
#
# 0	305797
# 1	181438
#
# Imbalanced dataset with around 65% human written and 35% AI generated text.
#
# ## IMDB
#
# 100000 samples all human written movie reviews.
#
# ## AG News
#
# 127600 samples of news articles in 4 topics all human written.
#
# ## Rotten Tomatoes Movie Reviews
#
# 10,662 samples all human written movie reviews.
#
# ## Newswire
#
# Contains 2.7 million unique public domain U.S. news wire articles, written between 1878 and 1977.
#
# ## English Quote
#
# 2,508 rows of human written quotes.
#
#
# ## LM Arena Datasets
#
# ### Search Arena 24k
#
# 24,069 samples (x2 for both columns) of AI generated data. This is recent and contains data from more current models.
#
# ### Arena Expert 5k
#
# 5,128 samples (x2 for both columns) of AI generated data. (outputs from 'expert' level problems in LM Arena).
#
# ### Arena Human Preference 140k
#
# 135,634 samples (x2 for both columns) of AI generated data.
#
# ### Raid Bench Summary
#
# 320_916 (160452 human, 160464 AI) samples of AI vs Human text

# # Dataset curation v2.3.0
#
# Two-dataset approach for better model validation:
#
# ## Dataset 1: Primary Curated (train.parquet + test.parquet)
# Target: 500k samples (250k human / 250k AI)
# - Carefully balanced across datasets and genres
# - No single dataset > 30% of total
# - Train/Test split: 80/20
#
# ### Allocation Strategy:
#
# **RAID Bench:** 75,000 per class (30% of total)
# - Benchmark dataset with adversarial attacks
# - Increased from v2.2.0 (was 85k = 42.8%)
#
# **Essays (NEW):**
# - Human essays: 40,000 (academic writing, avoid topic overfitting)
# - LLM essays: 37,488 (all available)
# - DAIGT v2 student essays: 25,000 human + 17,497 AI (all available)
# - Total essays: ~120k (24% of dataset)
#
# **News:**
# - Newswire: 50,000 (historical, 1878-1977)
# - AG News: 20,000 (modern news)
#
# **Arena datasets (recent models):**
# - Human Preference 140k: 50,000
# - Search Arena 24k: 30,000
# - Expert Arena 5k: ~10,000 (all available)
#
# **AI vs Human Collection (CNN DailyMail):**
# - 7 datasets × ~5-10k each = ~69k total
#
# **Other datasets:**
# - Human vs Machine: 15,000 per class
# - Human vs AI Sentences: 7,500 per class
# - AI Movie Reviews: 5,230 per class (at limit)
# - AI-and-Human-Text: 14,331 per class (at limit)
# - IMDB: 10,000
# - Rotten Tomatoes: 7,500
# - English Quotes: 2,508 (at limit)
# - General Knowledge: 10,000 (NEW, limited to avoid formulaic style)
#
# ## Dataset 2: Secondary Validation (validation.parquet)
# Target: ~96k samples from leftovers
# - Built from remaining samples after primary curation
# - Tests generalization to "more of the same"
# - Less carefully curated, minor imbalance OK
#
# ### Composition:
# - RAID leftovers: 30k per class
# - Human vs Machine leftovers: 12k per class
# - Arena leftovers: 20k (AI only)
# - AI vs Human Collection: ~12k (whatever remains)
#
# **Purpose:** If model performs similarly on both test sets, it generalizes well beyond the curated distribution.

# In[ ]:


def strat_sample(df: pl.LazyFrame, n_per_stratum: int, stratify_by: str = "label") -> pl.LazyFrame:
    sample_h = (
        df
        .filter(pl.col(stratify_by) == 0)
        .unique(maintain_order=True)
        .collect()
        .sample(n=n_per_stratum, seed=SEED, shuffle=True)
        .lazy()
    )
    sample_a = (
        df
        .filter(pl.col(stratify_by) == 1)
        .unique(maintain_order=True)
        .collect()
        .sample(n=n_per_stratum, seed=SEED, shuffle=True)
        .lazy()
    )
    return pl.concat([sample_h, sample_a])


def sample(df: pl.LazyFrame, n: int) -> pl.LazyFrame:
    return df.unique(maintain_order=True).collect().sample(n=n, seed=SEED, shuffle=True).lazy()


# In[ ]:


logger.info("Combining datasets for primary curated dataset...")
df_primary = (
    (
        pl
        .concat(
            [
                # RAID - increased to 75k per class (30% of 500k total)
                strat_sample(raid, n_per_stratum=75_000),
                # Essays - NEW additions
                sample(human_essays, 40_000),  # Human essays (avoid topic overfitting)
                llm_essays,  # All 37,488 LLM essays
                # DAIGT v2 - NEW addition (student essays)
                sample(daigt_v2.filter(pl.col("label") == 0), 25_000),  # Human student essays
                daigt_v2.filter(pl.col("label") == 1),  # All 17,497 AI student essays
                # General Knowledge - NEW (limited to 10k to avoid formulaic style)
                sample(general_knowledge, 10_000),
                # Newswire - increased from 35k to 50k
                sample(newswire, 50_000),
                # AG News - increased from 15k to 20k
                sample(ag, 20_000),
                # IMDB - increased from 7.5k to 10k
                sample(imdb, 10_000),
                # Rotten Tomatoes
                sample(rt, 7_500),
                # Arena datasets - increased significantly
                sample(human_preference_140k, 50_000),  # Up from 20k
                sample(search_arena, 30_000),  # Up from 20k
                expert_arena,  # All ~10k
                # AI vs Human Collection
                ai_vs_human_llama_8B,
                ai_vs_human_gemma,
                strat_sample(ai_vs_human_gpt35t, 4_500),
                ai_vs_human_llama,
                ai_vs_human_qwen,
                ai_vs_human_smolLM2,
                ai_vs_human_smolLM2_1_7B,
                # Other balanced datasets
                strat_sample(human_vs_ai_sentences, 7_500),
                ai_movie_reviews,  # All available (at limit)
                ai_and_human,  # All available (at limit)
                strat_sample(human_vs_machine, n_per_stratum=15_000),
                # Human-only at limits
                english_quotes,  # All 2,508
            ],
            how="diagonal",
        )
        .drop("models", "model", strict=False)
        .unique(["text", "label"], maintain_order=True)
        .cast({"dataset": pl.Categorical})
    )
    .collect()
    .lazy()
)

logger.info("Primary dataset combined.")
logger.info(f"Total samples: {df_primary.select(pl.len()).collect()[0, 0]}")
logger.info("Label distribution:")
print(df_primary.group_by("label").agg(pl.len()).sort("label").collect())


# In[ ]:


logger.info("Saving primary curated dataset to Parquet...")
df_primary.sink_parquet(DATA_PATH)
df_primary = pl.scan_parquet(DATA_PATH)
logger.info("Primary dataset saved.")
logger.info("Dataset summary:")
summary = df_primary.group_by("label").agg(pl.len()).sort("label").collect()
logger.info(f"{summary}")


# In[ ]:


from polars_splitters import split_into_train_eval

logger.info("Splitting primary dataset into train and eval sets...")
df_train, df_test = split_into_train_eval(
    df_primary.collect(), eval_rel_size=0.2, stratify_by=["dataset", "label"], seed=SEED
)

logger.info(f"Train size: {len(df_train)}")
logger.info(f"Test size: {len(df_test)}")


# In[ ]:


logger.info("Saving train set to Parquet...")
df_train.sample(fraction=1, shuffle=True, seed=SEED).write_parquet(TRAIN_PATH)


# In[ ]:


logger.info("Saving eval set to Parquet...")
df_test.sample(fraction=1, shuffle=True, seed=SEED).write_parquet(TEST_PATH)


# In[ ]:


logger.info("Building secondary validation dataset from leftovers...")

# Get primary texts as a set for efficient filtering
primary_texts = set(df_primary.select("text").collect().to_series().to_list())

df_validation = (
    pl
    .concat(
        [
            # RAID leftovers (30k per class)
            raid
            .filter(~pl.col("text").is_in(primary_texts))
            .collect()
            .partition_by("label")[0]
            .sample(30_000, seed=SEED)
            .lazy(),
            raid
            .filter(~pl.col("text").is_in(primary_texts))
            .collect()
            .partition_by("label")[1]
            .sample(30_000, seed=SEED)
            .lazy(),
            # Arena leftovers (AI only)
            human_preference_140k
            .filter(~pl.col("text").is_in(primary_texts))
            .collect()
            .sample(12_000, seed=SEED)
            .lazy(),
            search_arena.filter(~pl.col("text").is_in(primary_texts)).collect().sample(8_000, seed=SEED).lazy(),
            # Human vs Machine leftovers
            human_vs_machine
            .filter(~pl.col("text").is_in(primary_texts))
            .collect()
            .partition_by("label")[0]
            .sample(12_000, seed=SEED)
            .lazy(),
            human_vs_machine
            .filter(~pl.col("text").is_in(primary_texts))
            .collect()
            .partition_by("label")[1]
            .sample(12_000, seed=SEED)
            .lazy(),
            # AI vs Human Collection leftovers (take all remaining)
            *[
                ds.filter(~pl.col("text").is_in(primary_texts))
                for ds in [
                    ai_vs_human_llama_8B,
                    ai_vs_human_gemma,
                    ai_vs_human_gpt35t,
                    ai_vs_human_llama,
                    ai_vs_human_qwen,
                    ai_vs_human_smolLM2,
                    ai_vs_human_smolLM2_1_7B,
                ]
            ],
        ],
        how="diagonal",
    )
    .drop("models", "model", strict=False)
    .unique(["text", "label"], maintain_order=True)
    .cast({"dataset": pl.Categorical})
)

logger.info("Secondary validation dataset built.")
logger.info(f"Total samples: {df_validation.select(pl.len()).collect()[0, 0]}")
logger.info("Label distribution:")
print(df_validation.group_by("label").agg(pl.len()).sort("label").collect())


# In[ ]:


logger.info("Saving validation set to Parquet...")
df_validation.sink_parquet(VALIDATION_PATH)
df_validation = pl.scan_parquet(VALIDATION_PATH)


# In[ ]:


logger.info("Dataset curation complete!")
logger.info("Files created:")
logger.info(f"  - Primary dataset: {DATA_PATH}")
logger.info(f"  - Train set: {TRAIN_PATH}")
logger.info(f"  - Test set: {TEST_PATH}")
logger.info(f"  - Validation set: {VALIDATION_PATH}")


# In[ ]:


# ==============================================================================
# Markdown Bias Analysis (Dataset Property)
# ==============================================================================

logger.info("\n" + "=" * 80)
logger.info("MARKDOWN BIAS ANALYSIS")
logger.info("=" * 80)

# Define markdown patterns
MARKDOWN_PATTERNS = {
    "heading": r"^#+\s+",  # # Heading, ## Subheading
    "bold_asterisk": r"\*\*[^*]+\*\*",  # **bold**
    "italic_asterisk": r"(?<!\*)\*(?!\*)[\w\s]+\*(?!\*)",  # *italic*
    "code_block": r"```",  # ```code```
    "inline_code": r"`[^`]+`",  # `code`
    "list_item": r"^\s*[-*+]\s+",  # - item, * item
    "numbered_list": r"^\s*\d+\.\s+",  # 1. item
    "blockquote": r"^>\s+",  # > quote
}


def has_markdown(text: str) -> tuple[bool, dict[str, int]]:
    """Check if text contains markdown patterns."""
    pattern_counts = {}
    has_any = False

    for pattern_name, pattern in MARKDOWN_PATTERNS.items():
        count = len(re.findall(pattern, text, re.MULTILINE | re.DOTALL))
        pattern_counts[pattern_name] = count
        if count > 0:
            has_any = True

    return has_any, pattern_counts


# Analyze training set
logger.info("Analyzing training set for markdown bias...")
train_markdown_stats: dict[str, dict[str, int | dict[str, int]]] = {}

for text, label in zip(df_train["text"].to_list(), df_train["label"].to_list(), strict=True):
    label_key = "ai" if label == 1 else "human"
    has_md, counts = has_markdown(text)

    if label_key not in train_markdown_stats:
        train_markdown_stats[label_key] = {"total": 0, "with_markdown": 0, "patterns": {}}

    train_markdown_stats[label_key]["total"] += 1  # type: ignore[operator]
    if has_md:
        train_markdown_stats[label_key]["with_markdown"] += 1  # type: ignore[operator]

    for pattern, count in counts.items():
        if pattern not in train_markdown_stats[label_key]["patterns"]:  # type: ignore[operator]
            train_markdown_stats[label_key]["patterns"][pattern] = 0  # type: ignore[index, assignment]
        train_markdown_stats[label_key]["patterns"][pattern] += count  # type: ignore[index, operator]

# Compute percentages
ai_total = train_markdown_stats["ai"]["total"]  # type: ignore[assignment]
human_total = train_markdown_stats["human"]["total"]  # type: ignore[assignment]
ai_with_md = train_markdown_stats["ai"]["with_markdown"]  # type: ignore[assignment]
human_with_md = train_markdown_stats["human"]["with_markdown"]  # type: ignore[assignment]

ai_pct = (ai_with_md / ai_total) * 100  # type: ignore[operator]
human_pct = (human_with_md / human_total) * 100  # type: ignore[operator]

logger.info("\nMarkdown usage in training data:")
logger.info(f"  AI samples with markdown: {ai_with_md}/{ai_total} ({ai_pct:.2f}%)")
logger.info(f"  Human samples with markdown: {human_with_md}/{human_total} ({human_pct:.2f}%)")
logger.info(f"  Bias ratio (AI/Human): {ai_pct / human_pct:.2f}x")

# Pattern-specific breakdown
logger.info("\nPer-pattern breakdown (count per sample):")
pattern_stats = {}
for pattern_name in MARKDOWN_PATTERNS:
    ai_count = train_markdown_stats["ai"]["patterns"].get(pattern_name, 0) / ai_total  # type: ignore[operator]
    human_count = train_markdown_stats["human"]["patterns"].get(pattern_name, 0) / human_total  # type: ignore[operator]
    logger.info(f"  {pattern_name:20s}: AI={ai_count:.3f}, Human={human_count:.3f}")
    pattern_stats[pattern_name] = {"ai": round(ai_count, 4), "human": round(human_count, 4)}

# Save markdown bias for dataset metadata
markdown_bias = {
    "ai_pct": round(ai_pct, 2),
    "human_pct": round(human_pct, 2),
    "ratio": round(ai_pct / human_pct if human_pct > 0 else 0, 2),
    "pattern_breakdown": pattern_stats,
}

logger.info("=" * 80 + "\n")


# In[ ]:


# ==============================================================================
# Export Dataset Metadata
# ==============================================================================


logger.info("\n" + "=" * 80)
logger.info("GENERATING DATASET METADATA")
logger.info("=" * 80)


def get_dataset_composition(df: pl.LazyFrame) -> dict:
    """Extract dataset composition with counts per source dataset and label."""
    composition_df = df.group_by(["dataset", "label"]).agg(pl.len().alias("count")).collect().sort(["dataset", "label"])

    composition = {}
    for row in composition_df.iter_rows(named=True):
        dataset_name = row["dataset"]
        label = row["label"]
        count = row["count"]

        if dataset_name not in composition:
            composition[dataset_name] = {"count": 0, "human": 0, "ai": 0}

        composition[dataset_name]["count"] += count
        if label == 0:
            composition[dataset_name]["human"] += count
        else:
            composition[dataset_name]["ai"] += count

    return composition


def get_class_balance(df: pl.LazyFrame) -> dict:
    """Calculate class balance (proportion of human vs AI)."""
    counts = df.group_by("label").agg(pl.len().alias("count")).collect().sort("label")
    total = sum(row["count"] for row in counts.iter_rows(named=True))

    balance = {}
    for row in counts.iter_rows(named=True):
        label = "human" if row["label"] == 0 else "ai"
        balance[label] = round(row["count"] / total, 4)

    return balance


def compute_text_statistics(texts: list[str]) -> dict:
    """Compute statistical summary for text lengths and token counts."""
    # Text lengths (characters)
    text_lengths = [len(text) for text in texts]

    # Token counts (requires tokenization)
    logger.info(f"Tokenizing {len(texts)} texts for statistics...")
    tokens = tokenize(texts)
    token_counts = [len(t) for t in tokens]

    def stats_dict(values: list[int]) -> dict:
        arr = np.array(values)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "p50": int(np.percentile(arr, 50)),
            "p95": int(np.percentile(arr, 95)),
        }

    return {"text_lengths_chars": stats_dict(text_lengths), "token_counts": stats_dict(token_counts)}


# Collect dataset composition
logger.info("Collecting dataset composition...")
dataset_composition = {
    "train": get_dataset_composition(df_train.lazy()),
    "test": get_dataset_composition(df_test.lazy()),
    "validation": get_dataset_composition(df_validation),
}

# Collect class balance
class_balance = {
    "train": get_class_balance(df_train.lazy()),
    "test": get_class_balance(df_test.lazy()),
    "validation": get_class_balance(df_validation),
}

# Compute text statistics (sampling for validation to avoid long computation)
logger.info("Computing text statistics...")
train_stats = compute_text_statistics(df_train["text"].to_list())
test_stats = compute_text_statistics(df_test["text"].to_list())

validation_count = df_validation.select(pl.len()).collect()[0, 0]
val_sample_size = min(50_000, validation_count)
val_texts = df_validation.collect().sample(val_sample_size, seed=SEED)["text"].to_list()
logger.info(f"Computing validation stats on {val_sample_size} samples...")
val_stats = compute_text_statistics(val_texts)


# Create dataset metadata
dataset_metadata = {
    "dataset_version": str(RETRAINED_MODEL_VERSION),
    "created_timestamp": datetime.now(UTC).isoformat(),
    "preprocessing_version": __version__,
    "seed": SEED,
    "sample_counts": {
        "train": len(df_train),
        "test": len(df_test),
        "validation": validation_count,
        "total": len(df_train) + len(df_test) + validation_count,
    },
    "dataset_composition": dataset_composition,
    "class_balance": class_balance,
    "statistics": {"train": train_stats, "test": test_stats, "validation": val_stats},
    "markdown_bias": markdown_bias,
}

# Save metadata
metadata_path = DATA_DIR / "dataset_metadata.json"
with metadata_path.open("w", encoding="utf-8") as f:
    json.dump(dataset_metadata, f, indent=2)

logger.info(f"Saved dataset metadata to {metadata_path}")
logger.info(f"Dataset version: {RETRAINED_MODEL_VERSION}")
logger.info(
    f"Total datasets: train={len(dataset_composition['train'])}, "
    f"test={len(dataset_composition['test'])}, "
    f"validation={len(dataset_composition['validation'])}"
)
logger.info("=" * 80 + "\n")
