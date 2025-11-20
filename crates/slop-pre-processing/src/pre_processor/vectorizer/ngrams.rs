use ahash::AHashMap as HashMap;
use dashmap::DashMap;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

// TODO: Optimize n-gram counting
pub fn count_ngrams(tokens: &[u32], ngram_range: &[usize]) -> HashMap<Vec<u32>, usize> {
    let mut ngram_counter = HashMap::new();

    for &n in ngram_range {
        for window in tokens.windows(n) {
            *ngram_counter.entry(window.to_vec()).or_insert(0) += 1;
        }
    }
    ngram_counter
}

pub fn build_vocabulary(
    tokenized_texts: &[Vec<u32>],
    ngram_range: &[usize],
) -> DashMap<String, usize, ahash::RandomState> {
    let vocab_df = DashMap::with_hasher(ahash::RandomState::default());

    tokenized_texts.par_iter().progress().for_each(|tokens| {
        let ngrams = count_ngrams(tokens, ngram_range);
        for tokens in ngrams.into_keys() {
            let token = tokens
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>()
                .join(" ");
            vocab_df
                .entry(token)
                .and_modify(|e| *e += 1)
                .or_insert(1usize);
        }
    });
    vocab_df
}
