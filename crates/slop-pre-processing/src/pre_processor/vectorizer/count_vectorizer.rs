use ahash::HashMap;
use serde::{Deserialize, Serialize};
use sprs::CsMat;
use tracing::debug;

use super::{ngrams, params::VectorizerParams, tokenizer};
use crate::pre_processor::vectorizer::tokenizer::reverse_tokenize;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CountVectorizer {
    params: VectorizerParams,
    vocab: HashMap<String, usize>, // Does it have to be String?
}
impl CountVectorizer {
    pub fn fit<T: AsRef<str> + Sync>(texts: &[T], params: VectorizerParams) -> Self {
        debug!(num_texts = texts.len(), "Fitting CountVectorizer");

        let tokenized_texts = tokenizer::tokenize(texts);

        debug!("Building vocabulary");
        let vocab_df = ngrams::build_vocabulary(&tokenized_texts, params.ngram_range());
        let vocab_size = vocab_df.len();

        debug!(min_df = params.min_df(), "Applying min_df filtering");
        let filtered_vocab = vocab_df
            .into_iter()
            .filter(|(_, df)| *df >= params.min_df())
            .map(|(token, _)| token)
            .collect::<Vec<_>>();
        debug!(
            original_size = vocab_size,
            filtered_size = filtered_vocab.len(),
            "Vocabulary filtered by min_df"
        );

        let mut sorted_tokens = filtered_vocab;
        sorted_tokens.sort();
        let vocab = sorted_tokens
            .into_iter()
            .enumerate()
            .map(|(idx, token)| (token, idx))
            .collect::<HashMap<String, usize>>();

        debug!(vocab_size = vocab.len(), "CountVectorizer fitting complete");

        CountVectorizer { params, vocab }
    }

    pub fn transform<T: AsRef<str> + Sync>(&self, texts: &[T]) -> CsMat<f64> {
        debug!(
            num_texts = texts.len(),
            "Transforming texts using CountVectorizer"
        );

        let tokenized_texts = tokenizer::tokenize(texts);

        // Build CSR format directly
        let mut indptr = Vec::with_capacity(texts.len() + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();

        indptr.push(0);

        for tokens in &tokenized_texts {
            let ngrams = ngrams::count_ngrams(tokens, self.params.ngram_range());

            let mut row_entries = ngrams
                .iter()
                .filter_map(|(ngram_tokens, &count)| {
                    let ngram = ngram_tokens
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<String>>()
                        .join(" ");
                    self.vocab
                        .get(&ngram)
                        .map(|&col_idx| (col_idx, count as f64))
                })
                .collect::<Vec<_>>();
            row_entries.sort_by_key(|(col_idx, _)| *col_idx);
            for (col_idx, count) in row_entries {
                indices.push(col_idx);
                data.push(count);
            }
            indptr.push(indices.len());
        }
        debug!(
            non_zero_entries = data.len(),
            "Text transformation complete"
        );
        CsMat::new((texts.len(), self.num_features()), indptr, indices, data)
    }

    pub fn fit_transform<T: AsRef<str> + Sync>(
        texts: &[T],
        params: VectorizerParams,
    ) -> (Self, CsMat<f64>) {
        let vectorizer = Self::fit(texts, params);
        let transformed = vectorizer.transform(texts);
        (vectorizer, transformed)
    }

    pub fn num_features(&self) -> usize {
        self.vocab.len()
    }

    pub fn vocabulary(&self) -> HashMap<String, usize> {
        // For all strings, reverse the tokenization process
        self.vocab
            .iter()
            .map(|(k, v)| {
                let tokens = k
                    .split(' ')
                    .map(|s| s.parse().unwrap())
                    .collect::<Vec<u32>>();
                let ngram = reverse_tokenize(&tokens);
                (ngram, *v)
            })
            .collect()
    }
}
