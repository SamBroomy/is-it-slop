#!/usr/bin/env python

# In[1]:


import gc

import pandas as pd
from datasets import Dataset
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

# ### This notebook is based upon the work presented in [this](https://www.kaggle.com/code/datafan07/train-your-own-tokenizer) amazing notebook by [ERTUĞRUL DEMIR](https://www.kaggle.com/datafan07)

# In[2]:


test = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
sub = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")
org_train = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")

train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=",")


# In[3]:


kf_df = pd.read_csv("/kaggle/input/kf-dataset/kf_df.csv")
kf_df = kf_df.rename(columns={"prompt_title": "prompt_name"})
kf_df["label"] = 1
kf_df["source"] = "kf"
kf_df["RDizzl3_seven"] = False


# In[4]:


train = pd.concat([train, kf_df[train.columns].sample(10000, random_state=42)])


# In[5]:


train["label"].value_counts()


# In[6]:


train = train.drop_duplicates(subset=["text"])
train.reset_index(drop=True, inplace=True)


# In[7]:


test.text.values


# In[8]:


LOWERCASE = False
VOCAB_SIZE = 30522


# In[9]:


# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
# Creating huggingface dataset object
dataset = Dataset.from_pandas(test[["text"]])


def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
tokenized_texts_test = []

for text in tqdm(test["text"].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []

for text in tqdm(train["text"].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


# In[10]:


tokenized_texts_test[1]


# In[11]:


def dummy(text):
    return text


# In[12]:


vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    analyzer="word",
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents="unicode",
)

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

print(vocab)

vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    vocabulary=vocab,
    analyzer="word",
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    strip_accents="unicode",
)

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

del vectorizer
gc.collect()


# In[13]:


y_train = train["label"].values


# In[14]:


clf = MultinomialNB(alpha=0.02)
clf2 = MultinomialNB(alpha=0.01)
sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
p6 = {
    "n_iter": 1000,
    "verbose": -1,
    "learning_rate": 0.005689066836106983,
    "colsample_bytree": 0.8915976762048253,
    "colsample_bynode": 0.5942203285139224,
    "lambda_l1": 7.6277555139102864,
    "lambda_l2": 6.6591278779517808,
    "min_data_in_leaf": 156,
    "max_depth": 11,
    "max_bin": 813,
}
lgb = LGBMClassifier(**p6)

ensemble = VotingClassifier(
    estimators=[("mnb", clf), ("sgd", sgd_model), ("lgb", lgb)], weights=[0.3, 0.3, 0.4], voting="soft", n_jobs=-1
)
ensemble.fit(tf_train, y_train)

gc.collect()


# In[15]:


final_preds = ensemble.predict_proba(tf_test)[:, 1]


# In[16]:


sub["generated"] = final_preds
sub.to_csv("submission.csv", index=False)
sub


# In[ ]:
