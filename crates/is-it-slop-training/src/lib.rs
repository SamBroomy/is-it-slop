//! Burn training pipeline for the is-it-slop classifier.
//!
//! Rust port of `notebooks/hierarchical_train.py`, carrying only the configuration that the
//! hyperparameter campaign froze (see EXPERIMENTS.md, "Hyperparameter campaign") — rejected
//! experiment arms (mean+max chunk pooling, chunk-position embeddings, dropped chunk loss) were
//! deliberately not ported. `CutMix` is likewise absent until the chunk-output contract question in
//! `NEXT_RELEASE.md` is decided.
//!
//! Preprocessing (cleaning, tokenizing, chunking, vocabulary) belongs to the
//! `is-it-slop-preprocessing` crate, which both this crate and inference depend on so the two
//! sides cannot drift.

pub mod data;
pub mod evaluation;
pub mod training;

pub use is_it_slop_model as model;
