#!/usr/bin/env python

# I've been diving into tokenizer experimentations lately, as it's widely acknowledged that they wield a considerable influence on "lb" scores.
#
# Recent trend in the competition involves correcting typos through preprocessing steps. This entails utilizing either a library with licensing questions or deploying time-consuming T5 like models. Both approaches are commendable and served as valuable sources of inspiration for me. Rather than opting for the conventional route of normalizing datasets by correcting typos, my inclination led me to integrate them into the vocabulary during the tokenization phase.
#
# This approach doesn't tap into any hidden training set; it solely leverages the widely-used public training dataset "DAIGT V2." The classification component sticks to the basics, employing a VotingClassifier with minimal tweaks drawn from public notebooks. Feel free to experiment on these sides. Feel free to experiment with these aspects.
#
# With that brief introduction, let's jump right into the process.
#
#
# ### Important Note: This approach heavily relies on the public leaderboard score and the test set, so proceed with caution.

# In[1]:


import gc

import pandas as pd
from datasets import Dataset
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

# In[2]:


test = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
sub = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")
org_train = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")

train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=",")


# In[3]:


train = train.drop_duplicates(subset=["text"])

train.reset_index(drop=True, inplace=True)


# In[4]:


test.text.values


# ### Feel free to experiment with various combinations of hyperparameters below. I opted for a straightforward BERT vocab size, though it's doesn't mean much since BERT uses WordPiece, which you can also tinker with as part of your experimentations.

# In[5]:


LOWERCASE = False
VOCAB_SIZE = 30522


# In[6]:


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
    """A generator function for iterating over a dataset in chunks."""
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


# Training from iterator REMEMBER it's training on test set...
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

# Tokenize test set with new tokenizer
for text in tqdm(test["text"].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))


# Tokenize train set
tokenized_texts_train = []

for text in tqdm(train["text"].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


# In[7]:


tokenized_texts_test[1]


# In[8]:


def dummy(text):
    """A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it."""
    return text


# In[9]:


# Fitting TfidfVectoizer on test set

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


# Here we fit our vectorizer on train set but this time we use vocabulary from test fit.
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


# In[10]:


y_train = train["label"].values


# Just some sanity checks...

# In[11]:


tf_train


# In[12]:


tf_train.shape


# In[13]:


tf_test.shape


# ### A basic classifier pipeline with minimal tweaks from public notebooks.

# In[14]:


bayes_model = MultinomialNB(alpha=0.02)
sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")


ensemble = VotingClassifier(
    estimators=[("sgd", sgd_model), ("nb", bayes_model)], weights=[0.7, 0.3], voting="soft", n_jobs=-1
)
ensemble.fit(tf_train, y_train)


gc.collect()


# In[15]:


final_preds = ensemble.predict_proba(tf_test)[:, 1]


# In[16]:


sub["generated"] = final_preds
sub.to_csv("submission.csv", index=False)
sub


#
# ### I'm sharing this work to pave the way for new methods to enhance our pipelines. If you stumble upon something valuable or discover an improvement using any part of this approach, please do share it back with the community. I believe it's the kaggle way =)
#
# ### And that's it! Please don't forget to vote if you find this approach useful.
