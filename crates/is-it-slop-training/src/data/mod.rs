//! Dataset and batching for training.
//!
//! Not built yet — the model and training loop land first. The plan: encoded documents
//! (`[chunks, chunk_size]` compact ids from the preprocessing crate) batched into
//! `Tensor<B, 3, Int>` via Burn's `Batcher` trait, following the text-classification example.
