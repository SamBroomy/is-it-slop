# Old convo context

> This text was written for previous iterations of the project but gives useful context. The outputs below however are from the training pipeline notebook notebooks/train.ipynb. Because notebooks take a lot of space for an llm to parse I have converted the notebook to a script and included some of the important outputs below.

So I have a working implementation of my slop detection tool. It works end to end and now what I have done is made a pipeline to build the training dataset (before I was using a single kaggle dataset but I have now build notebooks/dataset_curation.ipynb) a wide variety of data sources from huggingface. As you will see these datasources vary but the key being we have a bunch of different data sources (some a mix of ai and human text, some just human and some just ai text). What I want to do now is really try and optimise my training pipeline to produce a really good model (or even set of models).
What I really want you to do first is understand the problem I am trying to solve and how I am currently going about it and why (why not) it makes sense to go the route I am going.
The whole idea of this kinda stems from the fact that I saw a kaggle competition from a few years ago and most of the best solutions created there own tokeniser (which makes a lot of sense for this type of problem. This got me thinking why dont we just use and existing tokeniser (like the open ai one tiktoken) as our tokeniser. I think my intuition for this is that llms talk via this kinda filter layer (the tokeniser, inputs and outputs get tokenised) This kinda gets me thinking that the way llms speak would leave some sort of signal that we should be able to detect in this tokeniser layer (because it feels intuitively that the llm speak would pass easier through this layer (the shapes created (tokens from words and sentences) much easier than the messy/ inconsistent, missspelt output humans produce. My thinking therefore was if we use these pre-existing tokenisers (and specifically Open-AIs tiktoken as it has the largest market share (most slop will probably be some form of open ai model as they are the largest ai company)) as our kinda input, we might be able to detect artefacts at this layer which would help us classify between ai and human text.
A few things we do are generate token ngrams. At the moment they are set to (3, 5) but maybe (2,4) would be better as longer chains of unusual tokens are less likely. What I really want to try and hone down on and really try and exploit is that the AI's will likely use a similar phrases or sentence structures or even words (made up of 1-many tokens) that will often be usual or 'clean' tokens (more used patterns). Where as I can imagine humans get a bit more messy and use 'less used tokens'. Like it feels the llms way of talking is much more streamlined so what we are essentially trying to do is to find out weather it matches the streamlined version or not?? Like as humans have more variance in there text rather than trying to explicitly predict if text is human or not, what may work better is trying to predict if the text is ai or not. I know these are the same thing really but I think its important to understand that the AI text would most likely be a bit more easier to predict, less variance and more clustered together.
Now want I want to do is to understand how I could improve my training pipeline (what parameters would make most sense in my tfidf vectorizer (are there params I am missing), what models should I look at for this problem, and what would be the best way to go about tuning them. While accuracy is important its probably more important that the models are fast at inference time as that is what I am trying to optimise for (while also working with sparse data (as thats what our tfidf vectoriser looks like about 99.98% sparcity.

Eventually i want to do something like this, but not yet:

Ok it looks like we are maybe getting artefacts from the dataset that we need to try and clean up. I think what would be interesting is a kinda batched approach (I think it may be a bit difficult to do properly (where naively would be splitting up on strings not tokens). Basically I have had an idea where we limit our model input to only 500 tokens, and what we do is break up strings to 500 tokens. I was thinking for inference that when we get a long string, we break up into chunks of (for now 500 tokens but this may even be too long),  and what we do is run all the chunks against the model and for each whole input, there is basically a vote (or we take the average of the probs, and the final prediction is basically that. It may be a bit harder to implement because we would need a training pipeline that reflects such (takes in text (documents) and splits it up where for all splits / chunks from the original document get the same classification as the original chunk. This way we are training on more (smaller samples so hopefully these artefacts at the start and end of sentences should be reduced) but then when it comes to inference our model is trained for this particular use? Does this make sense? Would this work?

Here are some current outputs from our training pipeline:

```
2025-11-27 10:53:04.171 | INFO     | __main__:<module>:3 - Feature matrix: (181926, 191842)
2025-11-27 10:53:04.172 | INFO     | __main__:<module>:4 - Sparsity: 99.90%
```

```
nb         - Human max prob: 1.0000, AI min prob: 0.0000
           -> Overlap region: 1.0000
sgd        - Human max prob: 1.0000, AI min prob: 0.0000
           -> Overlap region: 1.0000
logreg     - Human max prob: 0.9532, AI min prob: 0.0221
           -> Overlap region: 0.9311
svc        - Human max prob: 0.9974, AI min prob: 0.0011
           -> Overlap region: 0.9963
```

```
Accuracy by Dataset Source:
shape: (19, 4)
┌─────────────────────────────────┬───────┬──────────┬─────────────┐
│ dataset                         ┆ count ┆ accuracy ┆ avg_prob_ai │
│ ---                             ┆ ---   ┆ ---      ┆ ---         │
│ cat                             ┆ u32   ┆ f64      ┆ f64         │
╞═════════════════════════════════╪═══════╪══════════╪═════════════╡
│ human_vs_ai_sentences           ┆ 2200  ┆ 0.815909 ┆ 0.460189    │
│ ai-vs-human-HuggingFaceTB-Smol… ┆ 999   ┆ 0.830831 ┆ 0.762806    │
│ english_quotes                  ┆ 501   ┆ 0.892216 ┆ 0.161394    │
│ ai-vs-human-Qwen-Qwen2.5-1.5B-… ┆ 998   ┆ 0.928858 ┆ 0.904642    │
│ human_vs_machine                ┆ 4000  ┆ 0.93525  ┆ 0.504072    │
│ …                               ┆ …     ┆ …        ┆ …           │
│ search-arena-24k                ┆ 4000  ┆ 0.99225  ┆ 0.983635    │
│ ai-vs-human                     ┆ 1800  ┆ 0.995    ┆ 0.508504    │
│ newswire                        ┆ 7098  ┆ 0.997605 ┆ 0.010808    │
│ imdb                            ┆ 1421  ┆ 0.997889 ┆ 0.015067    │
│ AI_Human_generated_movie_revie… ┆ 2000  ┆ 0.998    ┆ 0.502585    │
└─────────────────────────────────┴───────┴──────────┴─────────────┘

Easiest datasets (might be artifacts):
shape: (5, 4)
┌─────────────────────────────────┬───────┬──────────┬─────────────┐
│ dataset                         ┆ count ┆ accuracy ┆ avg_prob_ai │
│ ---                             ┆ ---   ┆ ---      ┆ ---         │
│ cat                             ┆ u32   ┆ f64      ┆ f64         │
╞═════════════════════════════════╪═══════╪══════════╪═════════════╡
│ search-arena-24k                ┆ 4000  ┆ 0.99225  ┆ 0.983635    │
│ ai-vs-human                     ┆ 1800  ┆ 0.995    ┆ 0.508504    │
│ newswire                        ┆ 7098  ┆ 0.997605 ┆ 0.010808    │
│ imdb                            ┆ 1421  ┆ 0.997889 ┆ 0.015067    │
│ AI_Human_generated_movie_revie… ┆ 2000  ┆ 0.998    ┆ 0.502585    │
└─────────────────────────────────┴───────┴──────────┴─────────────┘

Hardest datasets (more realistic):
shape: (5, 4)
┌─────────────────────────────────┬───────┬──────────┬─────────────┐
│ dataset                         ┆ count ┆ accuracy ┆ avg_prob_ai │
│ ---                             ┆ ---   ┆ ---      ┆ ---         │
│ cat                             ┆ u32   ┆ f64      ┆ f64         │
╞═════════════════════════════════╪═══════╪══════════╪═════════════╡
│ human_vs_ai_sentences           ┆ 2200  ┆ 0.815909 ┆ 0.460189    │
│ ai-vs-human-HuggingFaceTB-Smol… ┆ 999   ┆ 0.830831 ┆ 0.762806    │
│ english_quotes                  ┆ 501   ┆ 0.892216 ┆ 0.161394    │
│ ai-vs-human-Qwen-Qwen2.5-1.5B-… ┆ 998   ┆ 0.928858 ┆ 0.904642    │
│ human_vs_machine                ┆ 4000  ┆ 0.93525  ┆ 0.504072    │
└─────────────────────────────────┴───────┴──────────┴─────────────┘
```

```
DEBUG:slop_pre_processing.pre_processor.vectorizer.count_vectorizer:Decoding vocabulary for the first time (will be cached) vocab_size=191842
Top 100 features predicting AI text:
  '. The': 15.2302
  '.

The': 10.9811
  '
-': 9.9075
  '.

**': 9.4186
  '[1': 8.1984
  '.
The': 7.3887
  'presents': 7.3820
  ' due to': 7.1019
  ':

**': 6.7245
  '- **': 6.6456
  '. Despite': 6.5935
  ', with': 6.4534
  ':
-': 6.2510
  '.
-': 6.2106
  '**:': 6.0243
  ' such as': 5.9847
  ', including': 5.9501
  '.

In': 5.9334
  '```

': 5.7750
  '.
*': 5.0483
  ':**
': 5.0480
  '.

As': 4.8769
  '
•': 4.8448
  ' a significant': 4.8170
  '.
•': 4.7768
  '. Results': 4.7429
  '. This': 4.7085
  '* **': 4.6376
  ' 202': 4.6359
  ' insights into': 4.6352
  'However,': 4.5435
  'examines': 4.5418
  '[2': 4.5179
  '. Through': 4.4761
  ' associated with': 4.4655
  ' has also': 4.4444
  '** (': 4.4323
  ' |
|': 4.3910
  ' not just': 4.3887
  '2025': 4.3800
  ', что': 4.3367
  ' like,': 4.3027
  ', making': 4.2317
  'Like,': 4.2297
  '. **': 4.1862
  'Additionally,': 4.1752
  ' The incident': 4.1113
  ' insight into': 4.0661
  ':

-': 4.0366
  '.

Born': 4.0211
  '.
**': 4.0210
  'ines the': 3.9758
  'examines the': 3.9679
  '. The incident': 3.9662
  '. "': 3.9403
  ':**

': 3.9332
  '1].

': 3.8719
  '. He': 3.7948
  '.

In ': 3.7861
  'So,': 3.7801
  ', where': 3.7380
  ' implications for': 3.7160
  ' has sparked': 3.7130
  '.

Born in': 3.6978
  '1].': 3.6720
  ' leading to': 3.6583
  'resents a': 3.6429
  ', which': 3.6423
  'presents a': 3.6422
  '

**': 3.6214
  ')
-': 3.6055
  ' highlights the': 3.6016
  ', who': 3.5978
  '[1].

': 3.5842
  '**.': 3.5572
  ' stated that': 3.5436
  ' concerns about': 3.5188
  ' related to': 3.4988
  ' has been': 3.4931
  '**
-': 3.4921
  ' 2025': 3.4912
  '.

---

': 3.4520
  '[3': 3.4165
  ' a comprehensive': 3.3787
  '. It's': 3.3743
  'Okay,': 3.3451
  ' has since': 3.3240
  ' was born in': 3.3224
  '. With': 3.3015
  ' 201': 3.2944
  '. Our': 3.2881
  ' argue that': 3.2791
  ', while': 3.2328
  'Finally,': 3.2233
  ' compared to': 3.2199
  '."

The': 3.2168
  ' emphasized that': 3.2103
  '
*': 3.2100
  '" (': 3.2096
  ' revealed that': 3.1875

Top 100 features predicting Human text:
  ' 's': -10.8965
  ' of the': -10.1153
  '. ': -7.2792
  ' in the': -6.0521
  ' , but': -5.7594
  ' said the': -5.7575
  ' , and': -5.6175
  ' this movie': -5.4531
  ', the': -5.3865
  ' and the': -5.1113
  '. —': -4.7745
  ', in': -4.6321
  ' to the': -4.4961
  ' and in': -4.2803
  '. A': -4.2644
  ' this film': -4.2373
  '. We': -4.1389
  ' for the': -4.1270
  ') and': -4.1157
  '. Although': -4.0107
  ' . .': -4.0027
  ' with the': -3.9549
  ' Mrs.': -3.9411
  ' all the': -3.8852
  ' said.': -3.8479
  ' , the': -3.8125
  ' the movie': -3.7893
  '. There': -3.7719
  ' and,': -3.7490
  ' and ': -3.6723
  ' said.

': -3.5578
  ' Inc.': -3.5274
  ', for': -3.4732
  '. All': -3.4299
  ' of a': -3.4122
  ' career
': -3.3714
  ' movie.': -3.3712
  ' in this': -3.2937
  ' June ': -3.2752
  ' as the': -3.2555
  ' the first': -3.2540
  ' because of': -3.2376
  ' Corp.': -3.2030
  ', although': -3.1944
  ' from the': -3.1872
  ' in a': -3.1667
  ' in which': -3.1553
  ' the film': -3.0997
  ' » .': -3.0639
  ' last night': -3.0576
  ' 't': -3.0229
  ' J.': -3.0012
  ' today.

': -2.9962
  ' . '': -2.9270
  ' said on': -2.9243
  ' April ': -2.9218
  ' of ': -2.9185
  ' out of': -2.8864
  ' to be': -2.8859
  ' on the': -2.8724
  ', said': -2.8454
  ' today to': -2.8436
  '. Early': -2.7965
  ' and also': -2.7862
  ' Here,': -2.7701
  ' most of': -2.7694
  ', of': -2.7649
  ', N': -2.7615
  ' said today': -2.7533
  ' film.': -2.7192
  '
The': -2.7097
  ' of an': -2.7008
  ' said it': -2.6874
  ' WASHINGTON': -2.6787
  ' at least': -2.6713
  ' Here, we': -2.6707
  ' when the': -2.6497
  'the film': -2.6475
  ' was the': -2.6459
  ' smog': -2.6096
  ' St.': -2.6081
  ' movie is': -2.6059
  ' In the': -2.5833
  '.-The': -2.5796
  ' at the': -2.5779
  ' LONDON': -2.5655
  '. Most': -2.5551
  '. Here,': -2.5541
  ' oil prices': -2.5471
  ' a little': -2.5414
  ' NEW YORK': -2.5349
  ' the .': -2.5347
  ' it .': -2.5207
  ' that,': -2.5205
  ' , a': -2.5122
  ' this one': -2.4937
  '. I': -2.4908
  ' is very': -2.4899
  ' today that': -2.4832
  ',000': -2.4731
```

```
Best threshold (F1 grid): 0.421 F1: 0.963375240090798

Confusion: 21735 785 893 22069
Precision, Recall, F1: 0.9656515270849741 0.9611096594373313 0.963375240090798

Best threshold (PR): 0.4212230772560469 F1: 0.9634172960225497
Best threshold (Youden): 0.4212230772560469 Youden: 0.9263405652987877
ROC AUC: 0.9935717390366013



Metrics (threshold=0.42):
  Accuracy:  0.9632
  Precision: 0.9657
  Recall:    0.9611
  F1 Score:  0.9634
  TP: 22069, FP: 783, TN: 21737, FN: 893

Confusion Matrix:
              Predicted
              0      1
Actual  0    21737    783
        1      893  22069
```

```
================================================================================
DATASET COMPOSITION
================================================================================

Dataset sources and counts:
dataset
newswire                                           35492
AI-and-Human-Generated-Text                        20000
human_vs_machine                                   20000
search-arena-24k                                   19998
arena-human-preference-140k                        19827
ag_news                                            15000
human_vs_ai_sentences                              11000
AI_Human_generated_movie_reviews                   10000
ai-vs-human-google-gemma-2-2b-it                   10000
arena-expert-5k                                    10000
ai-vs-human-meta-llama-Llama-3.1-8B-Instruct       10000
ai-vs-human                                         9000
rotten_tomatoes                                     7500
imdb                                                7104
ai-vs-human-HuggingFaceTB-SmolLM2-1.7B-Instruct     5000
ai-vs-human-meta-llama-Llama-3.2-1B-Instruct        4999
ai-vs-human-HuggingFaceTB-SmolLM2-360M-Instruct     4993
ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct              4988
english_quotes                                      2507
Name: count, dtype: int64

Total unique datasets: 19


================================================================================
LABEL DISTRIBUTION
================================================================================

Overall label distribution:
label
1    114805
0    112603
Name: count, dtype: int64

Label 0 (human): 112603 (49.5%)
Label 1 (AI): 114805 (50.5%)

================================================================================
LABEL DISTRIBUTION BY DATASET
================================================================================
                                                 total  ai_count  ai_ratio  \
dataset
newswire                                         35492         0       0.0
AI-and-Human-Generated-Text                      20000     10000       0.5
human_vs_machine                                 20000     10000       0.5
search-arena-24k                                 19998     19998       1.0
arena-human-preference-140k                      19827     19827       1.0
ag_news                                          15000         0       0.0
human_vs_ai_sentences                            11000      5500       0.5
AI_Human_generated_movie_reviews                 10000      5000       0.5
ai-vs-human-google-gemma-2-2b-it                 10000      5000       0.5
arena-expert-5k                                  10000     10000       1.0
ai-vs-human-meta-llama-Llama-3.1-8B-Instruct     10000      5000       0.5
ai-vs-human                                       9000      4500       0.5
rotten_tomatoes                                   7500         0       0.0
imdb                                              7104         0       0.0
ai-vs-human-HuggingFaceTB-SmolLM2-1.7B-Instruct   5000      5000       1.0
ai-vs-human-meta-llama-Llama-3.2-1B-Instruct      4999      4999       1.0
ai-vs-human-HuggingFaceTB-SmolLM2-360M-Instruct   4993      4993       1.0
ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct            4988      4988       1.0
english_quotes                                    2507         0       0.0

                                                 human_count
dataset
newswire                                               35492
AI-and-Human-Generated-Text                            10000
human_vs_machine                                       10000
search-arena-24k                                           0
arena-human-preference-140k                                0
ag_news                                                15000
human_vs_ai_sentences                                   5500
AI_Human_generated_movie_reviews                        5000
ai-vs-human-google-gemma-2-2b-it                        5000
arena-expert-5k                                            0
ai-vs-human-meta-llama-Llama-3.1-8B-Instruct            5000
ai-vs-human                                             4500
rotten_tomatoes                                         7500
imdb                                                    7104
ai-vs-human-HuggingFaceTB-SmolLM2-1.7B-Instruct            0
ai-vs-human-meta-llama-Llama-3.2-1B-Instruct               0
ai-vs-human-HuggingFaceTB-SmolLM2-360M-Instruct            0
ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct                     0
english_quotes                                          2507

> This was written before our cleaning pipeline but the outputs are about our cleaned data

================================================================================
ARTIFACT PATTERN ANALYSIS
================================================================================

Artifact pattern frequencies (total occurrences):
news_said_period         :  14291 ( 6.28%)
academic_citation_1      :  12044 ( 5.30%)
academic_citation_2      :   9748 ( 4.29%)
news_said_the            :   8711 ( 3.83%)
markdown_header          :   4375 ( 1.92%)
markdown_link            :   1915 ( 0.84%)
academic_this_study      :   1886 ( 0.83%)
wiki_history             :    967 ( 0.43%)
academic_this_paper      :    893 ( 0.39%)
wiki_early_life          :    257 ( 0.11%)
wiki_biography           :     50 ( 0.02%)
news_ap                  :     22 ( 0.01%)
news_reuters             :      0 ( 0.00%)
html_entity_39           :      0 ( 0.00%)
html_br                  :      0 ( 0.00%)
html_entity_any          :      0 ( 0.00%)

================================================================================
ARTIFACT PATTERNS BY DATASET (top patterns only)
================================================================================

news_ap:
  search-arena-24k                                  :    22 / 19998 (  0.1%)

news_said_the:
  newswire                                          :  4957 / 35492 ( 14.0%)
  ai-vs-human                                       :  1358 /  9000 ( 15.1%)
  ai-vs-human-google-gemma-2-2b-it                  :  1100 / 10000 ( 11.0%)
  ai-vs-human-meta-llama-Llama-3.1-8B-Instruct      :   585 / 10000 (  5.9%)
  ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct            :   214 /  4988 (  4.3%)

news_said_period:
  newswire                                          :  3859 / 35492 ( 10.9%)
  ai-vs-human                                       :  3129 /  9000 ( 34.8%)
  ai-vs-human-google-gemma-2-2b-it                  :  2500 / 10000 ( 25.0%)
  ai-vs-human-meta-llama-Llama-3.1-8B-Instruct      :  2129 / 10000 ( 21.3%)
  ai-vs-human-meta-llama-Llama-3.2-1B-Instruct      :   758 /  4999 ( 15.2%)

academic_citation_1:
  search-arena-24k                                  : 12044 / 19998 ( 60.2%)

academic_citation_2:
  search-arena-24k                                  :  9748 / 19998 ( 48.7%)

academic_this_paper:
  AI-and-Human-Generated-Text                       :   794 / 20000 (  4.0%)
  human_vs_machine                                  :    51 / 20000 (  0.3%)
  search-arena-24k                                  :    30 / 19998 (  0.2%)
  arena-expert-5k                                   :     7 / 10000 (  0.1%)
  arena-human-preference-140k                       :     7 / 19827 (  0.0%)

academic_this_study:
  AI-and-Human-Generated-Text                       :  1760 / 20000 (  8.8%)
  search-arena-24k                                  :    43 / 19998 (  0.2%)
  human_vs_machine                                  :    27 / 20000 (  0.1%)
  ai-vs-human-meta-llama-Llama-3.1-8B-Instruct      :    27 / 10000 (  0.3%)
  ai-vs-human-google-gemma-2-2b-it                  :    13 / 10000 (  0.1%)

wiki_early_life:
  human_vs_machine                                  :   252 / 20000 (  1.3%)
  AI-and-Human-Generated-Text                       :     2 / 20000 (  0.0%)
  search-arena-24k                                  :     2 / 19998 (  0.0%)
  arena-expert-5k                                   :     1 / 10000 (  0.0%)

wiki_biography:
  human_vs_machine                                  :    32 / 20000 (  0.2%)
  ai-vs-human                                       :     6 /  9000 (  0.1%)
  ai-vs-human-meta-llama-Llama-3.1-8B-Instruct      :     4 / 10000 (  0.0%)
  search-arena-24k                                  :     4 / 19998 (  0.0%)
  ai-vs-human-google-gemma-2-2b-it                  :     2 / 10000 (  0.0%)


================================================================================
ARTIFACT PATTERNS BY LABEL (Human=0, AI=1)
================================================================================

news_said_the:
  Human:  7850 / 112603 ( 6.97%)
  AI:      861 / 114805 ( 0.75%)
  Ratio (AI/Human): 0.11x

news_said_period:
  Human: 10384 / 112603 ( 9.22%)
  AI:     3907 / 114805 ( 3.40%)
  Ratio (AI/Human): 0.37x

academic_citation_1:
  Human:     0 / 112603 ( 0.00%)
  AI:    12044 / 114805 (10.49%)
  Ratio (AI/Human): infx

academic_citation_2:
  Human:     0 / 112603 ( 0.00%)
  AI:     9748 / 114805 ( 8.49%)
  Ratio (AI/Human): infx

academic_this_paper:
  Human:   145 / 112603 ( 0.13%)
  AI:      748 / 114805 ( 0.65%)
  Ratio (AI/Human): 5.06x

academic_this_study:
  Human:   755 / 112603 ( 0.67%)
  AI:     1131 / 114805 ( 0.99%)
  Ratio (AI/Human): 1.47x

wiki_early_life:
  Human:   252 / 112603 ( 0.22%)
  AI:        5 / 114805 ( 0.00%)
  Ratio (AI/Human): 0.02x


================================================================================
COMPREHENSIVE ARTIFACT SUMMARY
================================================================================
/var/folders/cs/b5nrw9r531s_zxp5gwn2k5w00000gn/T/ipykernel_3369/360814804.py:13: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  by_dataset = df.groupby("dataset")[f"has_{pattern}"].sum().sort_values(ascending=False)
            pattern  total  human    ai                                          primary_datasets  human_pct  ai_pct  ratio
            news_ap     22      0    22                                     search-arena-24k (22)       0.00    0.02 999.00
       news_reuters      0      0     0                                                                 0.00    0.00    NaN
      news_said_the   8711   7850   861                       newswire (4957); ai-vs-human (1358)       6.97    0.75   0.11
   news_said_period  14291  10384  3907                       newswire (3859); ai-vs-human (3129)       9.22    3.40   0.37
academic_citation_1  12044      0 12044                                  search-arena-24k (12044)       0.00   10.49 999.00
academic_citation_2   9748      0  9748                                   search-arena-24k (9748)       0.00    8.49 999.00
academic_this_paper    893    145   748  AI-and-Human-Generated-Text (794); human_vs_machine (51)       0.13    0.65   5.00
academic_this_study   1886    755  1131 AI-and-Human-Generated-Text (1760); search-arena-24k (43)       0.67    0.99   1.48
            html_br      0      0     0                                                                 0.00    0.00    NaN
    wiki_early_life    257    252     5   human_vs_machine (252); AI-and-Human-Generated-Text (2)       0.22    0.00   0.00
     wiki_biography     50     38    12                    human_vs_machine (32); ai-vs-human (6)       0.03    0.01   0.33



================================================================================
DATASET CHARACTERISTICS
================================================================================

HUMAN-ONLY DATASETS (potential human artifacts):
  newswire                                          :  35492 texts
  ag_news                                           :  15000 texts
  rotten_tomatoes                                   :   7500 texts
  imdb                                              :   7104 texts
  english_quotes                                    :   2507 texts

AI-ONLY DATASETS (potential AI artifacts):
  search-arena-24k                                  :  19998 texts
  arena-human-preference-140k                       :  19827 texts
  arena-expert-5k                                   :  10000 texts
  ai-vs-human-HuggingFaceTB-SmolLM2-1.7B-Instruct   :   5000 texts
  ai-vs-human-meta-llama-Llama-3.2-1B-Instruct      :   4999 texts
  ai-vs-human-HuggingFaceTB-SmolLM2-360M-Instruct   :   4993 texts
  ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct            :   4988 texts

MIXED DATASETS (both human and AI):
  AI-and-Human-Generated-Text                       :  20000 texts (50% AI)
  human_vs_machine                                  :  20000 texts (50% AI)
  human_vs_ai_sentences                             :  11000 texts (50% AI)
  ai-vs-human-meta-llama-Llama-3.1-8B-Instruct      :  10000 texts (50% AI)
  ai-vs-human-google-gemma-2-2b-it                  :  10000 texts (50% AI)
  AI_Human_generated_movie_reviews                  :  10000 texts (50% AI)
  ai-vs-human                                       :   9000 texts (50% AI)

Summary:
  Human-only datasets: 5 (67603 texts, 29.7%)
  AI-only datasets: 7 (69805 texts, 30.7%)
  Mixed datasets: 7 (90000 texts, 39.6%)
