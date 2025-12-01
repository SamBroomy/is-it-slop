use std::sync::{LazyLock, Mutex};

use is_it_slop_preprocessing::pre_processor::TfidfVectorizer;
use ort::session::{Session, builder::GraphOptimizationLevel};
mod threshold;
pub use threshold::CLASSIFICATION_THRESHOLD;

// pub static MODEL_BYTES: &[u8] = include_bytes!("../../../../model_artifacts/slop-classifier.onnx");
pub static MODEL_BYTES: &[u8] = include_bytes!(concat!(
    "../../../../model_artifacts/",
    env!("CARGO_PKG_VERSION"),
    "/slop-classifier.onnx"
));
pub static MODEL: LazyLock<Mutex<Session>> = LazyLock::new(|| {
    Mutex::new(
        Session::builder()
            .expect("Unable to create ONNX Runtime session builder")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("Unable to set optimization level")
            .with_intra_threads(4)
            .expect("Unable to set intra threads")
            .commit_from_memory(MODEL_BYTES)
            .expect("Unable to load model from memory"),
    )
});
pub static TOKENIZER_BYTES: &[u8] = include_bytes!(concat!(
    "../../../../model_artifacts/",
    env!("CARGO_PKG_VERSION"),
    "/tfidf_vectorizer.bin"
));
pub static PRE_PROCESSOR: LazyLock<TfidfVectorizer> = LazyLock::new(|| {
    TfidfVectorizer::from_bytes(TOKENIZER_BYTES).expect("Unable to load tokenizer from memory")
});
