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


import polars as pl
import polars.selectors as cs
from __init__ import DATA_PATH, RETRAINED_MODEL_VERSION, SEED, TEST_PATH, TRAIN_PATH
from datasets import load_dataset
from is_it_slop_preprocessing import __version__
from loguru import logger

print(f"Bindings version: {__version__}")
print(f"Pipeline model version output: {RETRAINED_MODEL_VERSION}")


# logging.basicConfig(level=logging.INFO)
# logging.getLogger("matplotlib").setLevel(logging.WARNING)
# logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def clean_text(df: pl.LazyFrame, text_col: str = "text") -> pl.LazyFrame:
    return (
        df.with_columns(
            pl.col(text_col)
            .str.replace_all(r"â€™", "'")  # Right single quote
            .str.replace_all(r"â€œ", '"')  # Left double quote
            .str.replace_all(r"â€", '"')  # Right double quote
            .str.replace_all(r'â€"', "—")  # Em dash
            .str.replace_all(r'â€"', "–")  # En dash  # noqa: RUF001
        )
        # 2. Decode HTML entities
        .with_columns(
            pl.col(text_col)
            .str.replace_all(r"&#39;", "'")
            .str.replace_all(r"&quot;", '"')
            .str.replace_all(r"&amp;", "&")
            .str.replace_all(r"&lt;", "<")
            .str.replace_all(r"&gt;", ">")
            .str.replace_all(r"&nbsp;", " ")
            .str.replace_all(r"&#(\d+);", "")  # Catch-all numeric entities
            # Clean up partial HTML entities (malformed)
            .str.replace_all(r"#39;?", "'")  # Partial &#39; → '
            .str.replace_all(r"quot;?", '"')  # Partial &quot; → "
        )
        # 3. Remove HTML tags
        .with_columns(
            pl.col(text_col)
            .str.replace_all(r"<br\s*/?>", " ")
            .str.replace_all(r"</?(p|div|span|strong|em|b|i|ul|ol|li|h[1-6])>", "")
        )
        # 4. Remove citation markers (more aggressive)
        .with_columns(
            pl.col(text_col)
            # Remove [1], [2], [123] with optional trailing punctuation
            .str.replace_all(r"\[\d+\][\.,;:\)\]]?", "")  # [1], [2], [1].
            # Remove malformed citations: 1], 2].
            .str.replace_all(r"\b\d+\][\.,;]?", "")
            # Remove [1].\n patterns
            .str.replace_all(r"\[\d+\]\.\s*\n", "\n")
            .str.replace_all(r"\[\d+", "")  # Remove [1, [2, etc.
            .str.replace_all(r"\d+\]", "")  # Remove 1], 2], etc.
            .str.replace_all(r"\[\d+\]", "")  # Remove [1], [2], etc.
        )
        # 5. Remove news wire attributions and datelines
        .with_columns(
            pl.col(text_col)
            # Remove news agencies in parentheses
            .str.replace_all(r"\s*\([A-Z]{2,}\)\s*", " ")  # (AP), (Reuters), (UPI)
            # Remove news agencies followed by dash
            .str.replace_all(r"\b(AP|AFP|Reuters|UPI)\s*[-—]\s*", "")
            # Remove common dateline patterns
            .str.replace_all(r"^(WASHINGTON|NEW YORK|LONDON|PARIS|BEIJING|MOSCOW|TOKYO|BERLIN)[,—]\s*", "")
            .str.replace_all(r"\n(WASHINGTON|NEW YORK|LONDON|PARIS|BEIJING|MOSCOW|TOKYO|BERLIN)[,—]\s*", "\n")
            # Remove "— The" and "— A" patterns (dateline endings)
            .str.replace_all(r"\s*—\s*(The|A)\s+", " ")
            # Clean up remaining em-dash patterns from datelines
            .str.replace_all(r"\)\s*—\s*", ") ")
        )
        # 5. Remove news wire attributions and datelines
        .with_columns(
            pl.col(text_col)
            # News agencies in parentheses
            .str.replace_all(r"\s*\((?:AP|AFP|Reuters|UPI|Bloomberg)\)\s*", " ")
            # News agencies followed by dash
            .str.replace_all(r"\b(?:AP|AFP|Reuters|UPI|Bloomberg)\s*[-—]\s*", "")
            # US state abbreviations in datelines (after comma)
            .str.replace_all(
                r",\s*(?:Ala\.|Ariz\.|Ark\.|Calif\.|Colo\.|Conn\.|Del\.|Fla\.|Ga\.|Ill\.|Ind\.|Kan\.|Ky\.|La\.|Md\.|Mass\.|Mich\.|Minn\.|Miss\.|Mo\.|Mont\.|Neb\.|Nev\.|N\.Y\.|N\.C\.|N\.D\.|Ohio|Okla\.|Ore\.|Pa\.|R\.I\.|S\.C\.|S\.D\.|Tenn\.|Tex\.|Vt\.|Va\.|Wash\.|W\.Va\.|Wis\.|Wyo\.)\b",
                ",",
            )
            # Common dateline cities at start or after newline
            .str.replace_all(
                r"^(?:WASHINGTON|NEW YORK|LONDON|PARIS|BEIJING|MOSCOW|TOKYO|BERLIN|BRUSSELS|GENEVA|ROME|MADRID|SEOUL|SYDNEY|MEXICO CITY|LOS ANGELES|SANFRANCISCO|CHICAGO|BOSTON|MIAMI|ATLANTA|HOUSTON|DALLAS|PHILADELPHIA|PHOENIX|SAN DIEGO|SEATTLE|DETROIT|DENVER)\s*[,—]\s*",  # noqa: E501
                "",
            )
            .str.replace_all(
                r"\n(?:WASHINGTON|NEW YORK|LONDON|PARIS|BEIJING|MOSCOW|TOKYO|BERLIN|BRUSSELS|GENEVA|ROME|MADRID|SEOUL|SYDNEY|MEXICO CITY|LOS ANGELES|SANFRANCISCO|CHICAGO|BOSTON|MIAMI|ATLANTA|HOUSTON|DALLAS|PHILADELPHIA|PHOENIX|SAN DIEGO|SEATTLE|DETROIT|DENVER)\s*[,—]\s*",  # noqa: E501
                "\n",
            )
            # "— The" and "— A" patterns (dateline endings)
            .str.replace_all(r"\s*—\s*(?:The|A)\s+", " ")
            .str.replace_all(r"\)\s*—\s*", ") ")
        )
        # 6. Remove academic/structured document section headers
        .with_columns(
            pl.col(text_col)
            .str.replace_all(
                r"\b(?:ABSTRACT|BACKGROUND|OBJECTIVE|METHODS?|RESULTS?|CONCLUSIONS?|DISCUSSION|INTRODUCTION)[\s:]*", ""
            )
            # Abstract section headers (case-sensitive, with punctuation)
            .str.replace_all(
                r"(?:^|\n)\s*(?:BACKGROUND|OBJECTIVE|METHODS|RESULTS|CONCLUSION|CONCLUSIONS|INTRODUCTION|DISCUSSION|ABSTRACT|SUMMARY):\s*",
                "\n",
            )
            # Wikipedia/biography section headers
            .str.replace_all(
                r"\.\s*(?:History|Biography|Early life|Background|Career|Personal life|Death|Legacy|Education|Awards|References|External links)\s*\n",
                ".\n",
            )
            .str.replace_all(
                r"^\s*(?:History|Biography|Early life|Background|Career|Personal life|Death|Legacy|Education|Awards|References|External links)\s*\n",
                "",
            )
            # Description headers (Wikipedia-style)
            .str.replace_all(r"\.\s*Description\s*\n", ".\n")
            .str.replace_all(r"^\s*Description\s*\n", "")
        )
        # 7. Remove timestamp/timezone markers (metadata artifacts)
        .with_columns(
            pl.col(text_col)
            # Timezone abbreviations with punctuation (EDT, EST, PST, etc.)
            .str.replace_all(r"\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC)[,\.]?\s+", " ")
            # "at HH:MM AM/PM EST" patterns
            .str.replace_all(
                r"\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC)?\s*[,\.]?", " "
            )
        )
        # 8. Remove academic prompt artifacts (sentence start only)
        .with_columns(
            pl.col(text_col)
            .str.replace_all(r"^(?:This (?:paper|study|article|abstract|research))\s+", "")
            .str.replace_all(r"\n(?:This (?:paper|study|article|abstract|research))\s+", "\n")
        )
        .with_columns(
            pl.col(text_col)
            .str.replace_all(r"^\d+\.\s+", "")  # Start of line
            .str.replace_all(r"\n\d+\.\s+", "\n")  # After newline
        )
        # 9. Normalize whitespace
        .with_columns(
            pl.col(text_col)
            .str.replace_all(r"  +", " ")
            .str.replace_all(r"\n\n\n+", "\n\n")
            .str.replace_all(r" \n", "\n")
            .str.replace_all(r"\n ", "\n")
        )
        # 10. Strip quotes/whitespace
        .with_columns(pl.col(text_col).str.strip_chars().str.strip_chars("'\"").str.strip_chars())
        # 10. Filter null/empty (preserve original logic)
        .filter(pl.col(text_col).is_not_null())
        .filter(pl.col(text_col).str.len_chars() > 0)
    )


def load_normal(dataset_name: str, rename: dict[str, str] | None = None, *, clean: bool = True) -> pl.LazyFrame:
    ds = load_dataset(dataset_name)
    dataset_name = dataset_name.rsplit("/", maxsplit=1)[-1]
    lf = pl.concat(
        ds[split].to_polars().lazy()  # type: ignore[attr-defined]
        for split in ds
    )
    if rename:
        lf = lf.rename(rename)
    if clean:
        lf = lf.pipe(clean_text)
    return lf.with_columns(cs.by_dtype(pl.Utf8).str.strip_chars(), dataset=pl.lit(dataset_name))


# # [English Quotes dataset](https://huggingface.co/datasets/Abirate/english_quotes)
# 
# Number of rows: 2,508
# Likes: 101
# Downloads last month: 5,702
# 
# 
# 
# > Assertion: Using quotes would provide a rich source of human-generated text.
# > Assumption: Quotes from real authors are human generated. (Issues may arise with popular quotes that have been AI generated and misattributed to famous authors.)
# 
# english_quotes is a dataset of all the quotes retrieved from goodreads quotes. This dataset can be used for multi-label text classification and text generation. The content of each quote is in English and concerns the domain of datasets for NLP and beyond.
# 
# 
# Data Fields
# 
#     author : The author of the quote.
#     quote : The text of the quote.
#     tags: The tags could be characterized as topics around the quote.
# 

#  Dataset Card for English quotes
# I-Dataset Summary
# 
# english_quotes is a dataset of all the quotes retrieved from goodreads quotes. This dataset can be used for multi-label text classification and text generation. The content of each quote is in English and concerns the domain of datasets for NLP and beyond.
# II-Supported Tasks and Leaderboards
# 
#     Multi-label text classification : The dataset can be used to train a model for text-classification, which consists of classifying quotes by author as well as by topic (using tags). Success on this task is typically measured by achieving a high or low accuracy.
#     Text-generation : The dataset can be used to train a model to generate quotes by fine-tuning an existing pretrained model on the corpus composed of all quotes (or quotes by author).
# 
# III-Languages
# 
# The texts in the dataset are in English (en).
# IV-Dataset Structure
# Data Instances
# 
# A JSON-formatted example of a typical instance in the dataset:
# 
# {'author': 'Ralph Waldo Emerson',
#  'quote': '“To be yourself in a world that is constantly trying to make you something else is the greatest accomplishment.”',
#  'tags': ['accomplishment', 'be-yourself', 'conformity', 'individuality']}
# 
# Data Fields
# 
#     author : The author of the quote.
#     quote : The text of the quote.
#     tags: The tags could be characterized as topics around the quote.
# 
# Data Splits
# 
# I kept the dataset as one block (train), so it can be shuffled and split by users later using methods of the hugging face dataset library like the (.train_test_split()) method.
# V-Dataset Creation
# Curation Rationale
# 
# I want to share my datasets (created by web scraping and additional cleaning treatments) with the HuggingFace community so that they can use them in NLP tasks to advance artificial intelligence.
# Source Data
# 
# The source of Data is goodreads site: from goodreads quotes
# Initial Data Collection and Normalization
# 
# The data collection process is web scraping using BeautifulSoup and Requests libraries. The data is slightly modified after the web scraping: removing all quotes with "None" tags, and the tag "attributed-no-source" is removed from all tags, because it has not added value to the topic of the quote.
# Who are the source Data producers ?
# 
# The data is machine-generated (using web scraping) and subjected to human additional treatment.
# 
# below, I provide the script I created to scrape the data (as well as my additional treatment):
# 
# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import json
# from collections import OrderedDict
# 
# page = requests.get('https://www.goodreads.com/quotes')
# if page.status_code == 200:
#     pageParsed = BeautifulSoup(page.content, 'html5lib')
#     
# # Define a function that retrieves information about each HTML quote code in a dictionary form.
# def extract_data_quote(quote_html):
#         quote = quote_html.find('div',{'class':'quoteText'}).get_text().strip().split('\n')[0]
#         author = quote_html.find('span',{'class':'authorOrTitle'}).get_text().strip()
#         if quote_html.find('div',{'class':'greyText smallText left'}) is not None:
#             tags_list = [tag.get_text() for tag in quote_html.find('div',{'class':'greyText smallText left'}).find_all('a')]
#             tags = list(OrderedDict.fromkeys(tags_list))
#             if 'attributed-no-source' in tags:
#                 tags.remove('attributed-no-source')
#         else:
#             tags = None
#         data = {'quote':quote, 'author':author, 'tags':tags}
#         return data
# 
# # Define a function that retrieves all the quotes on a single page. 
# def get_quotes_data(page_url):
#     page = requests.get(page_url)
#     if page.status_code == 200:
#         pageParsed = BeautifulSoup(page.content, 'html5lib')
#         quotes_html_page = pageParsed.find_all('div',{'class':'quoteDetails'})
#         return [extract_data_quote(quote_html) for quote_html in quotes_html_page]
# 
# # Retrieve data from the first page.
# data = get_quotes_data('https://www.goodreads.com/quotes')
# 
# # Retrieve data from all pages.
# for i in range(2,101):
#     print(i)
#     url = f'https://www.goodreads.com/quotes?page={i}'
#     data_current_page = get_quotes_data(url)
#     if data_current_page is None:
#         continue
#     data = data + data_current_page
# 
# data_df = pd.DataFrame.from_dict(data)
# for i, row in data_df.iterrows():
#     if row['tags'] is None:
#         data_df = data_df.drop(i)
# # Produce the data in a JSON format.
# data_df.to_json('C:/Users/Abir/Desktop/quotes.jsonl',orient="records", lines =True,force_ascii=False)
# # Then I used the familiar process to push it to the Hugging Face hub.
# 
# Annotations
# 
# Annotations are part of the initial data collection (see the script above).
# VI-Additional Informations
# Dataset Curators
# 
# Abir ELTAIEF
# Licensing Information
# 
# This work is licensed under a Creative Commons Attribution 4.0 International License (all software and libraries used for web scraping are made available under this Creative Commons Attribution license).
# Contributions
# 
# Thanks to @Abirate for adding this dataset. 

# In[ ]:


english_quotes = (
    load_normal("Abirate/english_quotes", {"quote": "text"})
    .with_columns(pl.col("text"), label=pl.lit(0, dtype=pl.Int8))
    .drop(["tags", "author"])
)
logger.info("Loaded English quotes")
# english_quotes.head(5).collect()


# # [Newswire dataset](https://huggingface.co/datasets/dell-research-harvard/newswire)
# 
# Number of rows: 1,440,010
# Likes: 85
# Downloads last month: 8,719
# 
# 
# >Assertion: Using newswire articles would provide a rich source of human-generated text.

# ---
# license: cc-by-4.0
# task_categories:
# - text-classification
# - text-generation
# - text-retrieval
# - summarization
# - question-answering
# language:
# - en
# tags:
# - social science
# - economics
# - news
# - newspaper
# - large language modeling
# - nlp
# - lam
# pretty_name: NewsWire
# size_categories:
# - 1M<n<10M
# ---
# # Dataset Card for NewsWire
# 
# ## Dataset Description
# 
# - **Homepage:** [Dell Research homepage](https://dell-research-harvard.github.io/)
# - **Repository:** [Github repository](https://github.com/dell-research-harvard)
# - **Paper:** [arxiv submission](https://arxiv.org/abs/2406.09490)
# - **Point of Contact:** [Melissa Dell](mailto:melissadell@fas.harvard.edu)
# 
# ### Dataset Summary
# 
# NewsWire contains 2.7 million unique public domain U.S. news wire articles, written between 1878 and 1977. Locations in these articles are georeferenced, topics are tagged using customized neural topic classification, named entities are recognized, and individuals are disambiguated to Wikipedia using a novel entity disambiguation model.  
# 
# ### Languages
# 
# English (en)
# 
# ## Dataset Structure
# Each year in the dataset is divided into a distinct file (eg. 1952_data_clean.json)
# 
# ### Data Instances
# An example from the NewsWire dataset looks like:
# 
# ```
# {
#     "year": 1880,
#     "dates": ["Feb-23-1880"], 
#     "article": "SENATE Washington, Feb. 23.--Bayard moved that in respect of the 
#         memory of George Washington the senate adjourn ... ",
#     "byline": "",
#     "newspaper_metadata": [
#         {
#             "lccn": "sn92053943",
#             "newspaper_title": "the rock island argus",
#             "newspaper_city": "rock island",
#             "newspaper_state": " illinois "
#         },
#         ...
#     ],
#     "antitrust": 0,
#     "civil_rights": 0,
#     "crime": 0,
#     "govt_regulation": 1,
#     "labor_movement": 0,
#     "politics": 1,
#     "protests": 0,
#     "ca_topic": "Federal Government Operations",
#     "ner_words": ["SENATE", "Washington", "Feb", "23", "Bayard", "moved", "that", 
#         "in", "respect", "of", "the", "memory", "of", "George", "Washington", 
#         "the", "senate", "adjourn", ... ],
#     "ner_labels": ["B-ORG", "B-LOC", "O", "B-PER", "B-PER", "O", "O", "O", "O", 
#         "O", "O", "O", "O", "B-PER", "I-PER", "O", "B-ORG", "O", ...],
#     "wire_city": "Washington",
#     "wire_state": "district of columbia",
#     "wire_country": "United States",
#     "wire_coordinates": [38.89511, -77.03637],
#     "wire_location_notes": "",
#     "people_mentioned": [
#         {
#             "wikidata_id": "Q23",
#             "person_name": "George Washington",
#             "person_gender": "man",
#             "person_occupation": "politician"
#         },
#         ...
#     ],
#     "cluster_size": 8
# }
# ```
# 
# 
# ### Data Fields
# 
# - `year`: year of article publication.
# 
# - `dates`: list of dates on which this article was published, as strings in the form mmm-DD-YYYY. 
# 
# - `byline`: article byline, if any.
# 
# - `article`: article text. 
# 
# - `newspaper_metadata`: list of newspapers that carried the article. Each newspaper is represented as a list of dictionaries, where `lccn` is the newspaper's Library of Congress identifier, `newspaper_title` is the name of the newspaper, and `newspaper_city` and `newspaper_state` give the location of the newspaper. 
# 
# - `antitrust`: binary variable. 1 if the article was classified as being about antitrust. 
# 
# - `civil_rights`: binary variable. 1 if the article was classified as being about civil rights. 
# 
# - `crime`: binary variable. 1 if the article was classified as being about crime. 
# 
# - `govt_regulation`: binary variable. 1 if the article was classified as being about government regulation. 
# 
# - `labor_movement`: binary variable. 1 if the article was classified as being about the labor movement. 
# 
# - `politics`: binary variable. 1 if the article was classified as being about politics. 
# 
# - `protests`: binary variable. 1 if the article was classified as being about protests. 
# 
# - `ca_topic`: predicted Comparative Agendas topic of article.
# 
# - `wire_city`: City of wire service bureau that wrote the article. 
# 
# - `wire_state`: State of wire service bureau that wrote the article. 
# 
# - `wire_country`: Country of wire service bureau that wrote the article.
# 
# - `wire_coordinates`: Coordinates of city of wire service bureau that wrote the article. 
# 
# - `wire_location_notes`: Contains wire dispatch location if it is not a geographic location. Can be one of ``Pacific Ocean (WWII)'', ``Supreme Headquarters Allied Expeditionary Force (WWII)'', ``North Africa'', ``War Front (WWI)'', ``War Front (WWII)'' or ``Johnson Space Center''.
# 
# - `people_mentioned`: list of disambiguated people mentioned in the article. Each disambiguated person is represented as a dictionary, where `wikidata_id` is their ID in Wikidata, `person_name` is their name on Wikipedia, `person_gender` is their gender from Wikidata and `person_occupation` is the first listed occupation on Wikidata. 
# 
# - `cluster_size`: Number of newspapers that ran the wire article. Equals length of `newspaper_metadata`.
# 
# 
# 
# ### Accessing the Data
# 
# The whole dataset can be easily downloaded using the `datasets` library: 
# 
# ```
# from datasets import load_dataset
# dataset_dict = load_dataset("dell-research-harvard/newswire")
# ```
# 
# Specific files can be downloaded by specifying them:
# 
# ```
# from datasets import load_dataset
# load_dataset(
#     "dell-research-harvard/newswire", 
#     data_files=["1929_data_clean.json", "1969_data_clean.json"]
# )
# ```
# 
# 
# ## Dataset Creation
# 
# ### Curation Rationale
# 
# The dataset was created to provide researchers with a large, high-quality corpus of historical news articles.  
# These texts provide a massive repository of information about historical topics and events - and which newspapers were covering them. 
# The dataset will be useful to a wide variety of researchers including historians, other social scientists, and NLP practitioners.
# 
# ### Source Data
# 
# #### Initial Data Collection and Normalization
# 
# Dataset construction is described in the associated paper. 
# 
# #### Who are the source language producers?
# 
# The source language was produced by people - by newspaper editors, columnists, and other sources.
# 
# ### Annotations
# 
# #### Annotation process
# 
# Not Applicable
# 
# #### Who are the annotators?
# 
# The dataset does not contain any additional annotations.
# 
# ### Personal and Sensitive Information
# 
# The dataset may contain information about individuals, to the extent that this is covered in news stories. However we make no additional information about individuals publicly available.
# 
# ## Considerations for Using the Data
# 
# ### Social Impact of Dataset
# 
#  This dataset provides high-quality data that could be used for pre-training a large language model to achieve better understanding of historical English and historical world knowledge. 
#  The dataset could also be added to the external database of a retrieval-augmented language model to make historical information more widely accessible.
#  
# ### Discussion of Biases
# 
# This dataset contains unfiltered content composed by newspaper editors, columnists, and other sources. 
# In addition to other potentially harmful content, the corpus may contain factual errors and intentional misrepresentations of news events. 
# All content should be viewed as individuals' opinions and not as a purely factual account of events of the day. 
# 
# 
# ## Additional Information
# 
# ### Dataset Curators
# 
# Emily Silcock (Harvard), Abhishek Arora (Harvard), Luca D'Amico-Wong (Harvard), Melissa Dell (Harvard) 
# 
# ### Licensing Information
# 
# The dataset has a CC-BY 4.0 license
# 
# ### Citation Information
# 
# You can cite this dataset using
# 
# ```
# @misc{silcock2024newswirelargescalestructureddatabase,
#       title={Newswire: A Large-Scale Structured Database of a Century of Historical News}, 
#       author={Emily Silcock and Abhishek Arora and Luca D'Amico-Wong and Melissa Dell},
#       year={2024},
#       eprint={2406.09490},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL},
#       url={https://arxiv.org/abs/2406.09490}, 
# }
# ```
# 
# ### Contributions
# 
# Coming Soon

# In[ ]:


newswire = (
    load_normal("dell-research-harvard/newswire", {"cleaned_article": "text"})
    .select("text", "dataset")
    .with_columns(label=pl.lit(0, dtype=pl.Int8))
)
# newswire.head(5).collect()
logger.info("Loaded newswire articles")


# # [rotten_tomatoes dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
# 
# Number of rows 10,662
# Likes: 93
# Downloads last month: 60,583
# 
# > Assertion: Using movie reviews would provide a rich source of human-generated text (may be help with shorter texts)
# > Assumption: Movie reviews are more likely to be human generated than AI generated.
# 

# ---
# annotations_creators:
# - crowdsourced
# language_creators:
# - crowdsourced
# language:
# - en
# license:
# - unknown
# multilinguality:
# - monolingual
# size_categories:
# - 1K<n<10K
# source_datasets:
# - original
# task_categories:
# - text-classification
# task_ids:
# - sentiment-classification
# paperswithcode_id: mr
# pretty_name: RottenTomatoes - MR Movie Review Data
# dataset_info:
#   features:
#   - name: text
#     dtype: string
#   - name: label
#     dtype:
#       class_label:
#         names:
#           '0': neg
#           '1': pos
#   splits:
#   - name: train
#     num_bytes: 1074810
#     num_examples: 8530
#   - name: validation
#     num_bytes: 134679
#     num_examples: 1066
#   - name: test
#     num_bytes: 135972
#     num_examples: 1066
#   download_size: 487770
#   dataset_size: 1345461
# train-eval-index:
# - config: default
#   task: text-classification
#   task_id: binary_classification
#   splits:
#     train_split: train
#     eval_split: test
#   col_mapping:
#     text: text
#     label: target
#   metrics:
#   - type: accuracy
#     name: Accuracy
#   - type: f1
#     name: F1
#     args:
#       average: binary
#   - type: f1
#     name: F1 micro
#     args:
#       average: micro
#   - type: f1
#     name: F1 weighted
#     args:
#       average: weighted
#   - type: precision
#     name: Precision macro
#     args:
#       average: macro
#   - type: precision
#     name: Precision micro
#     args:
#       average: micro
#   - type: precision
#     name: Precision weighted
#     args:
#       average: weighted
#   - type: recall
#     name: Recall macro
#     args:
#       average: macro
#   - type: recall
#     name: Recall micro
#     args:
#       average: micro
#   - type: recall
#     name: Recall weighted
#     args:
#       average: weighted
# ---
# 
# # Dataset Card for "rotten_tomatoes"
# 
# ## Table of Contents
# - [Dataset Description](#dataset-description)
#   - [Dataset Summary](#dataset-summary)
#   - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
#   - [Languages](#languages)
# - [Dataset Structure](#dataset-structure)
#   - [Data Instances](#data-instances)
#   - [Data Fields](#data-fields)
#   - [Data Splits](#data-splits)
# - [Dataset Creation](#dataset-creation)
#   - [Curation Rationale](#curation-rationale)
#   - [Source Data](#source-data)
#   - [Annotations](#annotations)
#   - [Personal and Sensitive Information](#personal-and-sensitive-information)
# - [Considerations for Using the Data](#considerations-for-using-the-data)
#   - [Social Impact of Dataset](#social-impact-of-dataset)
#   - [Discussion of Biases](#discussion-of-biases)
#   - [Other Known Limitations](#other-known-limitations)
# - [Additional Information](#additional-information)
#   - [Dataset Curators](#dataset-curators)
#   - [Licensing Information](#licensing-information)
#   - [Citation Information](#citation-information)
#   - [Contributions](#contributions)
# 
# ## Dataset Description
# 
# - **Homepage:** [http://www.cs.cornell.edu/people/pabo/movie-review-data/](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
# - **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# - **Paper:** [https://arxiv.org/abs/cs/0506075](https://arxiv.org/abs/cs/0506075)
# - **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# - **Size of downloaded dataset files:** 0.49 MB
# - **Size of the generated dataset:** 1.34 MB
# - **Total amount of disk used:** 1.84 MB
# 
# ### Dataset Summary
# 
# Movie Review Dataset.
# This is a dataset of containing 5,331 positive and 5,331 negative processed
# sentences from Rotten Tomatoes movie reviews. This data was first used in Bo
# Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for
# sentiment categorization with respect to rating scales.'', Proceedings of the
# ACL, 2005.
# 
# ### Supported Tasks and Leaderboards
# 
# [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# 
# ### Languages
# 
# [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# 
# ## Dataset Structure
# 
# ### Data Instances
# 
# #### default
# 
# - **Size of downloaded dataset files:** 0.49 MB
# - **Size of the generated dataset:** 1.34 MB
# - **Total amount of disk used:** 1.84 MB
# 
# An example of 'validation' looks as follows.
# ```
# {
#     "label": 1,
#     "text": "Sometimes the days and nights just drag on -- it 's the morning that make me feel alive . And I have one thing to thank for that : pancakes . "
# }
# ```
# 
# ### Data Fields
# 
# The data fields are the same among all splits.
# 
# #### default
# - `text`: a `string` feature.
# - `label`: a classification label, with possible values including `neg` (0), `pos` (1).
# 
# ### Data Splits
# 
# Reads Rotten Tomatoes sentences and splits into 80% train, 10% validation, and 10% test, as is the practice set out in
# 
# Jinfeng Li, ``TEXTBUGGER: Generating Adversarial Text Against Real-world Applications.''
# 
# | name  |train|validation|test|
# |-------|----:|---------:|---:|
# |default| 8530|      1066|1066|
# 
# ### Citation Information
# 
# ```
# @InProceedings{Pang+Lee:05a,
#   author =       {Bo Pang and Lillian Lee},
#   title =        {Seeing stars: Exploiting class relationships for sentiment
#                   categorization with respect to rating scales},
#   booktitle =    {Proceedings of the ACL},
#   year =         2005
# }
# 
# ```
# 
# 
# ### Contributions
# 
# Thanks to [@thomwolf](https://github.com/thomwolf), [@jxmorris12](https://github.com/jxmorris12) for adding this dataset.

# In[ ]:


rt = load_normal("cornell-movie-review-data/rotten_tomatoes").with_columns(label=pl.lit(0, dtype=pl.Int8))
# rt.head(5).collect()
logger.info("Loaded Rotten Tomatoes reviews")


# In[ ]:


# rt.group_by("label").agg(pl.len()).sort("label").collect()


# # [ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)
# 
# Number of rows: 127,600
# Likes: 177
# Downloads last month: 84,165

# ---
# annotations_creators:
# - found
# language_creators:
# - found
# language:
# - en
# license:
# - unknown
# multilinguality:
# - monolingual
# size_categories:
# - 100K<n<1M
# source_datasets:
# - original
# task_categories:
# - text-classification
# task_ids:
# - topic-classification
# paperswithcode_id: ag-news
# pretty_name: AG’s News Corpus
# dataset_info:
#   features:
#   - name: text
#     dtype: string
#   - name: label
#     dtype:
#       class_label:
#         names:
#           '0': World
#           '1': Sports
#           '2': Business
#           '3': Sci/Tech
#   splits:
#   - name: train
#     num_bytes: 29817303
#     num_examples: 120000
#   - name: test
#     num_bytes: 1879474
#     num_examples: 7600
#   download_size: 19820267
#   dataset_size: 31696777
# configs:
# - config_name: default
#   data_files:
#   - split: train
#     path: data/train-*
#   - split: test
#     path: data/test-*
# train-eval-index:
# - config: default
#   task: text-classification
#   task_id: multi_class_classification
#   splits:
#     train_split: train
#     eval_split: test
#   col_mapping:
#     text: text
#     label: target
#   metrics:
#   - type: accuracy
#     name: Accuracy
#   - type: f1
#     name: F1 macro
#     args:
#       average: macro
#   - type: f1
#     name: F1 micro
#     args:
#       average: micro
#   - type: f1
#     name: F1 weighted
#     args:
#       average: weighted
#   - type: precision
#     name: Precision macro
#     args:
#       average: macro
#   - type: precision
#     name: Precision micro
#     args:
#       average: micro
#   - type: precision
#     name: Precision weighted
#     args:
#       average: weighted
#   - type: recall
#     name: Recall macro
#     args:
#       average: macro
#   - type: recall
#     name: Recall micro
#     args:
#       average: micro
#   - type: recall
#     name: Recall weighted
#     args:
#       average: weighted
# ---
# 
# # Dataset Card for "ag_news"
# 
# ## Table of Contents
# - [Dataset Description](#dataset-description)
#   - [Dataset Summary](#dataset-summary)
#   - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
#   - [Languages](#languages)
# - [Dataset Structure](#dataset-structure)
#   - [Data Instances](#data-instances)
#   - [Data Fields](#data-fields)
#   - [Data Splits](#data-splits)
# - [Dataset Creation](#dataset-creation)
#   - [Curation Rationale](#curation-rationale)
#   - [Source Data](#source-data)
#   - [Annotations](#annotations)
#   - [Personal and Sensitive Information](#personal-and-sensitive-information)
# - [Considerations for Using the Data](#considerations-for-using-the-data)
#   - [Social Impact of Dataset](#social-impact-of-dataset)
#   - [Discussion of Biases](#discussion-of-biases)
#   - [Other Known Limitations](#other-known-limitations)
# - [Additional Information](#additional-information)
#   - [Dataset Curators](#dataset-curators)
#   - [Licensing Information](#licensing-information)
#   - [Citation Information](#citation-information)
#   - [Contributions](#contributions)
# 
# ## Dataset Description
# 
# - **Homepage:** [http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
# - **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# - **Paper:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# - **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# - **Size of downloaded dataset files:** 31.33 MB
# - **Size of the generated dataset:** 31.70 MB
# - **Total amount of disk used:** 63.02 MB
# 
# ### Dataset Summary
# 
# AG is a collection of more than 1 million news articles. News articles have been
# gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of
# activity. ComeToMyHead is an academic news search engine which has been running
# since July, 2004. The dataset is provided by the academic comunity for research
# purposes in data mining (clustering, classification, etc), information retrieval
# (ranking, search, etc), xml, data compression, data streaming, and any other
# non-commercial activity. For more information, please refer to the link
# http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .
# 
# The AG's news topic classification dataset is constructed by Xiang Zhang
# (xiang.zhang@nyu.edu) from the dataset above. It is used as a text
# classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann
# LeCun. Character-level Convolutional Networks for Text Classification. Advances
# in Neural Information Processing Systems 28 (NIPS 2015).
# 
# ## Dataset Structure
# 
# ### Data Instances
# 
# #### default
# 
# - **Size of downloaded dataset files:** 31.33 MB
# - **Size of the generated dataset:** 31.70 MB
# - **Total amount of disk used:** 63.02 MB
# 
# An example of 'train' looks as follows.
# ```
# {
#     "label": 3,
#     "text": "New iPad released Just like every other September, this one is no different. Apple is planning to release a bigger, heavier, fatter iPad that..."
# }
# ```
# 
# ### Data Fields
# 
# The data fields are the same among all splits.
# 
# #### default
# - `text`: a `string` feature.
# - `label`: a classification label, with possible values including `World` (0), `Sports` (1), `Business` (2), `Sci/Tech` (3).
# 
# ### Data Splits
# 
# | name  |train |test|
# |-------|-----:|---:|
# |default|120000|7600|
# 
# ### Citation Information
# 
# ```
# @inproceedings{Zhang2015CharacterlevelCN,
#   title={Character-level Convolutional Networks for Text Classification},
#   author={Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
#   booktitle={NIPS},
#   year={2015}
# }
# 
# ```
# 
# 
# ### Contributions
# 
# Thanks to [@jxmorris12](https://github.com/jxmorris12), [@thomwolf](https://github.com/thomwolf), [@lhoestq](https://github.com/lhoestq), [@lewtun](https://github.com/lewtun) for adding this dataset.

# In[ ]:


ag = load_normal("fancyzhx/ag_news").with_columns(label=pl.lit(0, dtype=pl.Int8))
logger.info("Loaded AG News articles")
# ag.head(5).collect()


# In[ ]:


# ag.group_by("label").agg(pl.len()).sort("label").collect()


# # [Imdb dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
# 
# Number of rows: 100,000
# Likes: 352
# Downloads last month: 171,036

# ---
# annotations_creators:
# - expert-generated
# language_creators:
# - expert-generated
# language:
# - en
# license:
# - other
# multilinguality:
# - monolingual
# size_categories:
# - 10K<n<100K
# source_datasets:
# - original
# task_categories:
# - text-classification
# task_ids:
# - sentiment-classification
# paperswithcode_id: imdb-movie-reviews
# pretty_name: IMDB
# dataset_info:
#   config_name: plain_text
#   features:
#   - name: text
#     dtype: string
#   - name: label
#     dtype:
#       class_label:
#         names:
#           '0': neg
#           '1': pos
#   splits:
#   - name: train
#     num_bytes: 33432823
#     num_examples: 25000
#   - name: test
#     num_bytes: 32650685
#     num_examples: 25000
#   - name: unsupervised
#     num_bytes: 67106794
#     num_examples: 50000
#   download_size: 83446840
#   dataset_size: 133190302
# configs:
# - config_name: plain_text
#   data_files:
#   - split: train
#     path: plain_text/train-*
#   - split: test
#     path: plain_text/test-*
#   - split: unsupervised
#     path: plain_text/unsupervised-*
#   default: true
# train-eval-index:
# - config: plain_text
#   task: text-classification
#   task_id: binary_classification
#   splits:
#     train_split: train
#     eval_split: test
#   col_mapping:
#     text: text
#     label: target
#   metrics:
#   - type: accuracy
#   - name: Accuracy
#   - type: f1
#     name: F1 macro
#     args:
#       average: macro
#   - type: f1
#     name: F1 micro
#     args:
#       average: micro
#   - type: f1
#     name: F1 weighted
#     args:
#       average: weighted
#   - type: precision
#     name: Precision macro
#     args:
#       average: macro
#   - type: precision
#     name: Precision micro
#     args:
#       average: micro
#   - type: precision
#     name: Precision weighted
#     args:
#       average: weighted
#   - type: recall
#     name: Recall macro
#     args:
#       average: macro
#   - type: recall
#     name: Recall micro
#     args:
#       average: micro
#   - type: recall
#     name: Recall weighted
#     args:
#       average: weighted
# ---
# 
# # Dataset Card for "imdb"
# 
# ## Table of Contents
# - [Dataset Description](#dataset-description)
#   - [Dataset Summary](#dataset-summary)
#   - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
#   - [Languages](#languages)
# - [Dataset Structure](#dataset-structure)
#   - [Data Instances](#data-instances)
#   - [Data Fields](#data-fields)
#   - [Data Splits](#data-splits)
# - [Dataset Creation](#dataset-creation)
#   - [Curation Rationale](#curation-rationale)
#   - [Source Data](#source-data)
#   - [Annotations](#annotations)
#   - [Personal and Sensitive Information](#personal-and-sensitive-information)
# - [Considerations for Using the Data](#considerations-for-using-the-data)
#   - [Social Impact of Dataset](#social-impact-of-dataset)
#   - [Discussion of Biases](#discussion-of-biases)
#   - [Other Known Limitations](#other-known-limitations)
# - [Additional Information](#additional-information)
#   - [Dataset Curators](#dataset-curators)
#   - [Licensing Information](#licensing-information)
#   - [Citation Information](#citation-information)
#   - [Contributions](#contributions)
# 
# ## Dataset Description
# 
# - **Homepage:** [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
# - **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# - **Paper:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# - **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
# - **Size of downloaded dataset files:** 84.13 MB
# - **Size of the generated dataset:** 133.23 MB
# - **Total amount of disk used:** 217.35 MB
# 
# ### Dataset Summary
# 
# Large Movie Review Dataset.
# This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.
# 
# ## Dataset Structure
# 
# ### Data Instances
# 
# #### plain_text
# 
# - **Size of downloaded dataset files:** 84.13 MB
# - **Size of the generated dataset:** 133.23 MB
# - **Total amount of disk used:** 217.35 MB
# 
# An example of 'train' looks as follows.
# ```
# {
#     "label": 0,
#     "text": "Goodbye world2\n"
# }
# ```
# 
# ### Data Fields
# 
# The data fields are the same among all splits.
# 
# #### plain_text
# - `text`: a `string` feature.
# - `label`: a classification label, with possible values including `neg` (0), `pos` (1).
# 
# ### Data Splits
# 
# |   name   |train|unsupervised|test |
# |----------|----:|-----------:|----:|
# |plain_text|25000|       50000|25000|
# 
# ```
# @InProceedings{maas-EtAl:2011:ACL-HLT2011,
#   author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
#   title     = {Learning Word Vectors for Sentiment Analysis},
#   booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
#   month     = {June},
#   year      = {2011},
#   address   = {Portland, Oregon, USA},
#   publisher = {Association for Computational Linguistics},
#   pages     = {142--150},
#   url       = {http://www.aclweb.org/anthology/P11-1015}
# }
# 
# ```
# 
# 
# ### Contributions
# 
# Thanks to [@ghazi-f](https://github.com/ghazi-f), [@patrickvonplaten](https://github.com/patrickvonplaten), [@lhoestq](https://github.com/lhoestq), [@thomwolf](https://github.com/thomwolf) for adding this dataset.

# In[ ]:


imdb = load_normal("stanfordnlp/imdb").with_columns(label=pl.lit(0, dtype=pl.Int8))
logger.info("Loaded IMDB reviews")
# imdb.head(5).collect()


# In[ ]:


# imdb.group_by("label").agg(pl.len()).sort("label").collect()


# # [AI-human-text](https://huggingface.co/datasets/andythetechnerd03/AI-human-text)
# 
# Number of rows: 487,235
# Likes: 8
# Downloads last month: 365

# ---
# dataset_info:
#   features:
#   - name: text
#     dtype: string
#   - name: generated
#     dtype: int8
#   splits:
#   - name: train
#     num_bytes: 1026814130.2626022
#     num_examples: 462873
#   - name: test
#     num_bytes: 54043432.73739777
#     num_examples: 24362
#   download_size: 570879675
#   dataset_size: 1080857563
# configs:
# - config_name: default
#   data_files:
#   - split: train
#     path: data/train-*
#   - split: test
#     path: data/test-*
# license: apache-2.0
# task_categories:
# - text-classification
# language:
# - en
# tags:
# - code
# pretty_name: AI vs Human Text
# size_categories:
# - 100K<n<1M
# ---
# This is a processed dataset of Human vs AI Text roughly 400k rows. This is taken from the Kaggle dataset https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data then processed and split into training and test sets.

# In[ ]:


ai_human = load_normal("andythetechnerd03/AI-human-text").rename({"generated": "label"})
logger.info("Loaded AI vs Human text")
# ai_human.head(5).collect()


# In[ ]:


# ai_human.group_by("label").agg(pl.len()).sort("label").collect()


# # [Human vs Machine](https://huggingface.co/datasets/NicolaiSivesind/human-vs-machine)
# 
# Number of rows: 320,000
# Likes: 19
# Downloads last month: 188

# ---
# license: cc
# task_categories:
# - text-classification
# pretty_name: Human vs Machine - Labled text segments produced by humans and LLMs
# size_categories:
# - 100K<n<1M
# language:
# - en
# tags:
# - chatgpt
# - gpt
# - research abstracts
# - wikipedia introductions
# ---
# # Human-vs-Machine
# This is a dataset collection created in relation to a bachelor thesis written by Nicolai Thorer Sivesind and Andreas Bentzen Winje. It contains human-produced and machine-generated text samples from two domains: Wikipedia introducions and Scientific research abstracts. 
# 
# Each of the two domains are already exisitng datasets reformatted for text-classification:
# 
# [GPT-wiki-intros:](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro)
# + Generated samples are produced using the GPT-3 model, _text-curie-001_
#   + Target content set by title of real wikipedia introduction and a starter sentence.
#   + Target word count of 200 words each.
# + Contains 150k data points of each class.
# + Created by Aaditya Bhat
# 
# [ChatGPT-Research-Abstracts](https://huggingface.co/datasets/NicolaiSivesind/ChatGPT-Research-Abstracts):
# + Generated samples are produced using the GPT-3.5 model, _GPT-3.5-turbo-0301_ (Snapshot of the model used in ChatGPT 1st of March, 2023).
#   + Target content set by title of real abstract.
#   + Target word count equal to the human-produced abstract
# + Contains 10k data points of each class.
# + Created by Nicolai Thorer Sivesind
# 
# ### Credits
# + [GPT-wiki-intro](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro), by Aaditya Bhat
# 
# ### Citation
# Please use the following citation:
# ```
# @misc {sivesind_2023,
#     author       = { {Nicolai Thorer Sivesind}, {Andreas Bentzen Winje}},
#     title        = { Human-vs-Machine },
#     year         = 2023,
#     publisher    = { Hugging Face }
# }
# ```
# 
# More information about the dataset will be added once the thesis is finished (end of may 2023).

# In[ ]:


from huggingface_hub import hf_hub_download

wiki_path = hf_hub_download(
    repo_id="NicolaiSivesind/human-vs-machine", filename="wiki-labeled.csv", repo_type="dataset"
)
abstracts_path = hf_hub_download(
    repo_id="NicolaiSivesind/human-vs-machine", filename="research-abstracts-labeled.csv", repo_type="dataset"
)


human_vs_machine = (
    pl.concat([pl.scan_csv(wiki_path), pl.scan_csv(abstracts_path)])
    .with_columns(cs.by_dtype(pl.Utf8).str.strip_chars(), dataset=pl.lit("human_vs_machine"))
    .drop(["title", "word_count"])
    .pipe(clean_text)
    .cast({"label": pl.Int8})
)
logger.info("Loaded Human vs Machine text")
# human_vs_machine.head(5).collect()


# In[ ]:


# human_vs_machine.group_by("label").agg(pl.len()).sort("label").collect()


# # [AI-and-Human-Generated-Text](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text)
# 
# Number of rows: 28,662
# Likes: 19
# Downloads last month: 486

# ---
# license: mit
# language:
# - en
# size_categories:
# - 10K<n<100K
# task_categories:
# - text-classification
# ---
# 
# # AI & Human Generated Text
# 
# ## I am Using this dataset for AI Text Detection for https://exnrt.com.
# Check Original DataSet GitHub Repository Here: https://github.com/panagiotisanagnostou/AI-GA
# 
# 
# ## Description
# The AI-GA dataset, short for Artificial Intelligence Generated Abstracts, comprises abstracts and titles. Half of these abstracts are generated by AI, while the remaining half are original. Primarily intended for research and experimentation in natural language processing, especially concerning language generation and machine learning, this dataset offers ample opportunities for exploration and analysis.
# 
# The AI-GA dataset comprises 28,662 samples, each containing an abstract, a title, and a label. It is evenly divided into two categories: "AI-generated abstracts" and "original abstracts." The label distinguishes between an original abstract (labeled 0) and an AI-generated one (labeled 1). Notably, the AI-generated abstracts are crafted using cutting-edge language generation techniques, notably leveraging the GPT-3 model.
# 
# ### Large Alternative:
# This compilation encompasses https://github.com/sakibsh/LLM both human-authored and LLM-generated (utilizing GPT-4 and BARD) texts spanning various genres such as essays, stories, poetry, and Python code. It serves as a valuable asset for investigating LLM text detection methodologies.

# In[ ]:


ai_and_human = (
    load_normal("Ateeqq/AI-and-Human-Generated-Text", {"abstract": "text"}).cast({"label": pl.Int8}).drop("title")
)
logger.info("Loaded AI vs Human text from Ateeqq/AI-and-Human-Generated-Text")
# ai_and_human.head(5).collect()


# In[ ]:


# ai_and_human.group_by("label").agg(pl.len()).sort("label").collect()


# # [AI generated movie reviews](https://huggingface.co/datasets/Milkyway-islander/AI_Human_generated_movie_reviews)
# 
# Number of rows: 10,460
# Likes: 3
# Downloads last month: 29
# 
# There are a good verity of AI models used to generate these texts.

# ---
# dataset_info:
#   features:
#   - name: text
#     dtype: string
#   - name: labels
#     dtype: int64
#   - name: models
#     dtype: string
#   - name: __index_level_0__
#     dtype: int64
#   splits:
#   - name: train
#     num_bytes: 15157689
#     num_examples: 10460
#   download_size: 8750952
#   dataset_size: 15157689
# configs:
# - config_name: default
#   data_files:
#   - split: train
#     path: data/train-*
# task_categories:
# - text-classification
# language:
# - en
# size_categories:
# - 10K<n<100K
# ---
# # Dataset Card for Dataset Name
# 
# 
# 
# This dataset card aims to be a base template for new datasets. It has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1).
# 
# ## Dataset Details
# 
# ### Dataset Description
# 
# <!-- Provide a longer summary of what this dataset is. -->
# 
# The "AI_Human_generated_movie_reviews" dataset consists of 5.23k AI-generated movie reviews alongside 5.23k human-written reviews from the Stanford IMDB dataset. The AI reviews were created using several models, including Gemini 1.5 Pro, GPT-3.5-Turbo, and GPT-4.0-Turbo-Preview, via the OpenAI API. Quality control measures were applied during generation, producing 3-5 reviews per session with multiple sessions (ranging from 20 to 100) for each review. Reviews with an average word length under 215 or over 345 were excluded from the dataset.
# 
# 
# - **Curated by:** [More Information Needed]
# - **Funded by [optional]:** [More Information Needed]
# - **Shared by [optional]:** [Amber Zhan]
# - **Language(s) (NLP):** [English]
# - **License:** [More Information Needed]
# 
# ## Dataset Structure
# 
# Dataset has 3 columns and 10460 rows. 
# 

# In[ ]:


ai_movie_reviews = (
    load_normal("Milkyway-islander/AI_Human_generated_movie_reviews")
    .rename({"labels": "label"})
    .cast({"label": pl.Int8})
    .drop("__index_level_0__")
)
logger.info("Loaded AI vs Human movie reviews")
# ai_movie_reviews.head(5).collect()


# In[ ]:


# ai_movie_reviews.group_by("models").agg(pl.len()).sort("models").collect()


# # [Human vs AI Sentences](https://huggingface.co/datasets/shahxeebhassan/human_vs_ai_sentences)
# 
# Number of rows: 105,000
# Likes: 9
# Downloads last month: 151

# ---
# license: mit
# task_categories:
# - text-classification
# language:
# - en
# size_categories:
# - 100K<n<1M
# ---
# ### Dataset Description
# This dataset contains 105,000 sentences, each labeled as either human-written (`0`) or AI-generated (`1`). It is designed for text classification tasks, particularly for distinguishing between human and AI-generated text.
# 
# ### Dataset Structure
# - **Number of Instances**: 105,000 sentences
# - **Labels**: 
#   - `0`: Human-written
#   - `1`: AI-generated
# 
# ### Usage
# This dataset can be used to train models for text classification tasks. Below is an example of how to load and use the dataset with the Hugging Face `datasets` library:
# 
# ```python
# from datasets import load_dataset
# 
# dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences")
# ```
# 
# ### Data Fields
# - **text**: The text of the sentence.
# - **label**: The label indicating whether the sentence is human-written (`0`) or AI-generated (`1`).
# 
# ### License
# This dataset is licensed under the MIT License.
# 
# ### Task Categories
# - Text Classification
# 
# ### Languages
# - English (`en`)
# 
# ### Size Categories
# - 100K < n < 1M
# 
# ### Source Data
# The sentences in this dataset were collected from various sources to ensure a diverse range of topics and writing styles.
# 
# ### Acknowledgements
# This dataset was created to support research and development in the field of AI and text classification.
# 
# ---

# In[ ]:


human_vs_ai_sentences = load_normal("shahxeebhassan/human_vs_ai_sentences").cast({"label": pl.Int8})
logger.info("Loaded Human vs AI sentences")
# human_vs_ai_sentences.head(5).collect()


# In[ ]:


# human_vs_ai_sentences.group_by("label").agg(pl.len()).sort("label").collect()


# # [Human Raid](https://huggingface.co/datasets/charisgao/human-raid)
# 
# Number of rows: 948,371
# Likes: 1
# Downloads last month: 10
# 
# Unsure about this one as it seems to be data taken from diffrent sources `reddit`, `recipes`, `reviews` which could quite easily be AI generated

# configs:
#   - config_name: default
#     data_files:
#       - split: train
#         path: data/human_data_v0.csv
# license: mit
# task_categories:
#   - text-classification
# language:
#   - en
# pretty_name: RAID-human
# size_categories:
#   - 1M<n<10M

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
    ds = load_normal(dataset_name, clean=False)
    dataset_name = dataset_name.rsplit("/", maxsplit=1)[-1]
    return (
        ds.rename({"ai": "1", "human": "0"})
        .select(["1", "0"])
        .unpivot()
        .rename({"variable": "label", "value": "text"})
        .with_columns(dataset=pl.lit(dataset_name))
        .cast({"label": pl.Int8})
        .pipe(clean_text)
    )


# ---
# license: mit
# task_categories:
# - text-classification
# - text-generation
# language:
# - en
# pretty_name: AI vs Human CNN Daily News
# size_categories:
# - 1K<n<10K
# ---
# 
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
# 
# ## Dataset Description
# This dataset contains pairs of original articles and their AI-generated completions.
# 
# ## Data Fields
# - `human`: The original complete article
# - `ai`: The AI-generated completion of a truncated version using GPT-3.5 Turbo
# 
# ## Usage
#     

# In[ ]:


ai_vs_human_gpt35t = load_ai_vs_human_collection("ilyasoulk/ai-vs-human")
logger.info("Loaded AI vs Human GPT-3.5-Turbo text")
# ai_vs_human_gpt35t.head(5).collect()


# ---
# license: mit
# task_categories:
# - text-classification
# - text-generation
# language:
# - en
# pretty_name: AI vs Human CNN Daily News
# size_categories:
# - 1K<n<10K
# ---
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
# 
# ## Dataset Description
# This dataset showcases pairs of truncated articles and their respective completions, crafted either by humans or an AI language model. 
# Each article was randomly truncated between 25% and 50% of its length. 
# The language model was then tasked with generating a completion that mirrored the characters count of the original human-written continuation.
# 
# ## Data Fields
# - 'human': The original human-authored continuation of the truncated article, preserved in its entirety.
# - 'ai': The AI-generated continuation of the truncated article, designed to match the original in length and coherence.
# 
# ## Model and Sampling Parameters
# The model used to generate the AI completions was HuggingFaceTB/SmolLM2-360M-Instruct.
# 
# The sampling parameters used were:
# {'frequency_penalty': 0.2, 'max_tokens': 1000, 'presence_penalty': 0.5, 'temperature': 0.5}
# 
# ## License
# MIT License
#     

# In[ ]:


ai_vs_human_smolLM2 = load_ai_vs_human_collection("zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-360M-Instruct")  # noqa: N816
logger.info("Loaded AI vs Human SmolLM2 text")
# ai_vs_human_smolLM2.head(5).collect()


# ---
# license: mit
# task_categories:
# - text-classification
# - text-generation
# language:
# - en
# pretty_name: AI vs Human CNN Daily News
# size_categories:
# - 1K<n<10K
# ---
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
# 
# ## Dataset Description
# This dataset showcases pairs of truncated articles and their respective completions, crafted either by humans or an AI language model. 
# Each article was randomly truncated between 25% and 50% of its length. 
# The language model was then tasked with generating a completion that mirrored the characters count of the original human-written continuation.
# 
# ## Data Fields
# - 'human': The original human-authored continuation of the truncated article, preserved in its entirety.
# - 'ai': The AI-generated continuation of the truncated article, designed to match the original in length and coherence.
# 
# ## Model and Sampling Parameters
# The model used to generate the AI completions was HuggingFaceTB/SmolLM2-1.7B-Instruct.
# 
# The sampling parameters used were:
# {'frequency_penalty': 0.2, 'max_tokens': 1000, 'presence_penalty': 0.5, 'temperature': 0.5}
# 
# ## License
# MIT License
#     

# In[ ]:


ai_vs_human_smolLM2_1_7B = load_ai_vs_human_collection("zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-1.7B-Instruct")  # noqa: N816
logger.info("Loaded AI vs Human SmolLM2 1.7B text")
# ai_vs_human_smolLM2_1_7B.head(5).collect()


# ---
# license: mit
# task_categories:
# - text-classification
# - text-generation
# language:
# - en
# pretty_name: AI vs Human CNN Daily News
# size_categories:
# - 1K<n<10K
# ---
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
# 
# ## Dataset Description
# This dataset showcases pairs of truncated articles and their respective completions, crafted either by humans or an AI language model. 
# Each article was randomly truncated between 25% and 50% of its length. 
# The language model was then tasked with generating a completion that mirrored the characters count of the original human-written continuation.
# 
# ## Data Fields
# - 'human': The original human-authored continuation of the truncated article, preserved in its entirety.
# - 'ai': The AI-generated continuation of the truncated article, designed to match the original in length and coherence.
# 
# ## Model and Sampling Parameters
# The model used to generate the AI completions was Qwen/Qwen2.5-1.5B-Instruct.
# 
# The sampling parameters used were:
# {'frequency_penalty': 0.2, 'max_tokens': 1000, 'presence_penalty': 0.5, 'temperature': 0.5}
# 
# ## License
# MIT License
#     

# In[ ]:


ai_vs_human_qwen = load_ai_vs_human_collection("zcamz/ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct")
logger.info("Loaded AI vs Human Qwen text")
# ai_vs_human_qwen.head(5).collect()


# ---
# license: mit
# task_categories:
# - text-classification
# - text-generation
# language:
# - en
# pretty_name: AI vs Human CNN Daily News
# size_categories:
# - 1K<n<10K
# ---
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
# 
# ## Dataset Description
# This dataset showcases pairs of truncated articles and their respective completions, crafted either by humans or an AI language model. 
# Each article was randomly truncated between 25% and 50% of its length. 
# The language model was then tasked with generating a completion that mirrored the characters count of the original human-written continuation.
# 
# ## Data Fields
# - 'human': The original human-authored continuation of the truncated article, preserved in its entirety.
# - 'ai': The AI-generated continuation of the truncated article, designed to match the original in length and coherence.
# 
# ## Model and Sampling Parameters
# The model used to generate the AI completions was google/gemma-2-2b-it.
# 
# The sampling parameters used were:
# {'frequency_penalty': 0.2, 'max_tokens': 1000, 'presence_penalty': 0.5, 'temperature': 0.5}
# 
# ## License
# MIT License
#     

# In[ ]:


ai_vs_human_gemma = load_ai_vs_human_collection("zcamz/ai-vs-human-google-gemma-2-2b-it")
logger.info("Loaded AI vs Human Gemma text")
# ai_vs_human_gemma.head(5).collect()


# ---
# license: mit
# task_categories:
# - text-classification
# - text-generation
# language:
# - en
# pretty_name: AI vs Human CNN Daily News
# size_categories:
# - 1K<n<10K
# ---
# # AI vs Human dataset on the [CNN Daily mails](https://huggingface.co/datasets/abisee/cnn_dailymail)
# 
# ## Dataset Description
# This dataset showcases pairs of truncated articles and their respective completions, crafted either by humans or an AI language model. 
# Each article was randomly truncated between 25% and 50% of its length. 
# The language model was then tasked with generating a completion that mirrored the characters count of the original human-written continuation.
# 
# ## Data Fields
# - 'human': The original human-authored continuation of the truncated article, preserved in its entirety.
# - 'ai': The AI-generated continuation of the truncated article, designed to match the original in length and coherence.
# 
# ## Model and Sampling Parameters
# The model used to generate the AI completions was meta-llama/Llama-3.2-1B-Instruct.
# 
# The sampling parameters used were:
# {'frequency_penalty': 0.2, 'max_tokens': 1000, 'presence_penalty': 0.5, 'temperature': 0.5}
# 
# ## License
# MIT License
#     

# In[ ]:


ai_vs_human_llama = load_ai_vs_human_collection("zcamz/ai-vs-human-meta-llama-Llama-3.2-1B-Instruct")
logger.info("Loaded AI vs Human Llama text")
# ai_vs_human_llama.head(5).collect()


# ---
# license: mit
# task_categories:
# - text-classification
# - text-generation
# language:
# - en
# pretty_name: AI vs Human OpenWebTxt
# size_categories:
# - 1K<n<10K
# ---
# # AI vs Human dataset on the [OpenWebTxt](https://huggingface.co/datasets/stas/openwebtext-10k)
# 
# ## Dataset Description
# This dataset showcases pairs of truncated text and their respective completions, crafted either by humans or an AI language model. 
# Each article was randomly truncated between 25% and 50% of its length. 
# The language model was then tasked with generating a completion that mirrored the characters count of the original human-written continuation.
# 
# ## Data Fields
# - 'human': The original human-authored continuation of the truncated text, preserved in its entirety.
# - 'ai': The AI-generated continuation of the truncated text, designed to match the original in length and coherence.
# 
# ## Model and Sampling Parameters
# The model used to generate the AI completions was meta-llama/Llama-3.1-8B-Instruct.
# 
# The sampling parameters used were:
# {'frequency_penalty': 0.2, 'max_tokens': 1000, 'presence_penalty': 0.5, 'temperature': 0.5}
# 
# ## License
# MIT License
#     

# In[ ]:


ai_vs_human_llama_8B = load_ai_vs_human_collection("ilyasoulk/ai-vs-human-meta-llama-Llama-3.1-8B-Instruct")  # noqa: N816
logger.info("Loaded AI vs Human Llama 8B text")
# ai_vs_human_llama_8B.head(5).collect()


# ## [LM Arena Search](https://huggingface.co/datasets/lmarena-ai/search-arena-24k)
# 

# In[ ]:


def load_lm_arena(dataset_name: str, *, clean: bool = False) -> pl.LazyFrame:
    ds_name = dataset_name.rsplit("/", maxsplit=1)[-1]

    lf = (
        load_normal(dataset_name, clean=False)
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
    )
    if clean:
        lf = lf.pipe(clean_text)
    return lf


# In[ ]:


search_arena = load_lm_arena("lmarena-ai/search-arena-24k")
logger.info("Loaded Search Arena text")
# search_arena.head(5).collect()


# # [Arena Expert 5k](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k)

# In[ ]:


import ast
import contextlib
import re

import numpy as np


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
    load_normal("lmarena-ai/arena-expert-5k", clean=False)
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
)
# expert_arena.head(5).collect()
# .cast({"conversation_a": pl.List(pl.Struct), "conversation_b": pl.List(pl.Utf8)}, strict=False)


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
# human_preference_140k.head(5).collect()


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

# # Dataset curation
# 
# I want random samples from diffrent datasets.
# 
# I want 200k samples total with a 50/50 split between human written and AI generated text.
# 
# So I will take:
# 
# ## AI vs Human datasets 
# 
# > Contains a good verity of AI models used to generate these texts.
# 
# Human samples: 35,000
# AI samples: 35,000
# 
# 
# ## AI-vs-human Sentences
# 
# Take a sample of 10,000 from each class.
# 
# Human samples: 5,000    - 40,000 total
# AI samples: 5,000       - 40,000 total
# 
# 
# ## AI generated movie reviews
# Take a sample of 5,000 from each class.
# Human samples: 5,000    - 45,000 total
# AI samples: 5,000       - 45,000 total
# 
# 
# ## AI-and-Human-Generated-Text
# Take a sample of 10,000 from each class.
# 
# Human samples: 10,000   - 55,000 total
# AI samples: 10,000      - 55,000 total
# 
# ## Human vs Machine
# Take a sample of 10,000 from each class.
# 
# Human samples: 10,000   - 65,000 total
# AI samples: 10,000      - 65,000 total
# 
# ## IMDB
# Take a sample of 5,000 human written movie reviews.
# 
# Human samples: 5,000    - 70,000 total
# AI samples: N/A     - 65,000 total
# 
# 
# ## AG News
# Take a sample of 10,000 human written news articles.
# 
# Human samples: 10,000   - 80,000 total
# AI samples: N/A     - 65,000 total
# 
# ## Rotten Tomatoes Movie Reviews
# Take a sample of 5,000 human written movie reviews.
# Human samples: 5,000    - 85,000 total
# AI samples: N/A     - 65,000 total
# 
# ## Newswire
# Take a sample of 27,492 human written news articles.
# 
# Human samples: 27,492   - 112,492 total
# AI samples: N/A     - 65,000 total
# 
# ## English Quote
# Take all 2,508 human written quotes.
# 
# Human samples: 2,508    - 115,000 total
# AI samples: N/A     - 65,000 total
# 
# 
# 
# ## LM Arena Datasets
# 
# ### Arena Expert 5k
# Take 10,000 of AI generated data (2 x 5,128 samples).
# 
# Human samples: N/A     - 115,000 total
# AI samples: 10,000     - 75,000 total
# 
# ### Search Arena 24k
# Take a sample of 20,000 of AI generated data (2 x 10,000 samples).
# 
# Human samples: N/A     - 115,000 total
# AI samples: 20,000     - 95,000 total
# 
# ### Arena Human Preference 140k
# Take a sample of 20,000 of AI generated data (2 x 10,000 samples).
# 
# Human samples: N/A     - 115,000 total
# AI samples: 20,000     - 115,000 total
# 

# In[ ]:


def strat_sample(df: pl.LazyFrame, n_per_stratum: int, stratify_by: str = "label", *, seed: int = 42) -> pl.LazyFrame:
    sample_h = (
        df.filter(pl.col(stratify_by) == 0)
        .unique(maintain_order=True)
        .collect()
        .sample(n=n_per_stratum, seed=seed, shuffle=True)
        .lazy()
    )
    sample_a = (
        df.filter(pl.col(stratify_by) == 1)
        .unique(maintain_order=True)
        .collect()
        .sample(n=n_per_stratum, seed=seed, shuffle=True)
        .lazy()
    )
    return pl.concat([sample_h, sample_a])


def sample(df: pl.LazyFrame, n: int, seed: int = 42) -> pl.LazyFrame:
    return df.unique(maintain_order=True).collect().sample(n=n, seed=seed, shuffle=True).lazy()


# In[ ]:


logger.info("Combining datasets...")
df = (
    pl.concat(
        [
            # AI vs Human datasets
            *[
                ai_vs_human_llama_8B,
                ai_vs_human_gemma,
                strat_sample(ai_vs_human_gpt35t, 4_500, seed=SEED),
                ai_vs_human_llama,
                ai_vs_human_qwen,
                ai_vs_human_smolLM2,
                ai_vs_human_smolLM2_1_7B,
            ],
            # There are duplicates in the Human vs AI sentences dataset
            # (specifically for the human class so we will
            # oversample below on the human data so when we drop duplicates later we still have enough human data)
            strat_sample(human_vs_ai_sentences, 5_500, seed=SEED),
            strat_sample(ai_movie_reviews, n_per_stratum=5_000, seed=SEED),
            # Human and AI generated text datasets
            strat_sample(ai_and_human, n_per_stratum=10_000, seed=SEED),
            strat_sample(human_vs_machine, n_per_stratum=10_000, seed=SEED),
            # Human text datasets
            *[
                sample(imdb, 7_500, seed=SEED),
                sample(ag, 15_000, seed=SEED),
                sample(rt, 7_500, seed=SEED),
                sample(newswire, 35_492, seed=SEED),
                english_quotes,
            ],
            # AI text datasets
            *[
                sample(expert_arena, 10_000, seed=SEED),
                sample(search_arena, 20_000, seed=SEED),
                sample(human_preference_140k, 20_000, seed=SEED),
            ],
        ],
        how="diagonal",
    )
    .drop("models")
    .unique(["text", "label"], maintain_order=True)
    .cast({"dataset": pl.Categorical})
)


# In[ ]:


logger.info("Saving curated dataset to Parquet...")
df.sink_parquet(DATA_PATH)
df = pl.scan_parquet(DATA_PATH)
logger.info("Curated dataset saved.")
logger.info("Dataset summary:")
summary = df.group_by("label").agg(pl.len()).sort("label").collect()
logger.info(f"{summary}")


# In[ ]:


from polars_splitters import split_into_train_eval

logger.info("Splitting into train and eval sets...")
df_train, df_test = split_into_train_eval(df.collect(), eval_rel_size=0.2, stratify_by=["dataset", "label"], seed=SEED)


# In[ ]:


logger.info("Saving train set to Parquet...")
df_train.sample(fraction=1, shuffle=True, seed=SEED).write_parquet(TRAIN_PATH)


# In[ ]:


logger.info("Saving eval set to Parquet...")
df_test.sample(fraction=1, shuffle=True, seed=SEED).write_parquet(TEST_PATH)


# In[ ]:


logger.info("Data splitting and saving complete.")

