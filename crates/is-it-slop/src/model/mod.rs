use std::sync::{LazyLock, Mutex};

use is_it_slop_preprocessing::pre_processor::TfidfVectorizer;
use ort::session::{Session, builder::GraphOptimizationLevel};

include!(concat!(env!("OUT_DIR"), "/threshold.rs"));

/// Current model version
///
/// This is set during build time based on the model artifacts used.
/// The model version is used to ensure that the underlying model and tokenizer are compatible.
pub const MODEL_VERSION: &str = env!("MODEL_VERSION");

pub static MODEL_BYTES: &[u8] = include_bytes!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("CLASSIFIER_MODEL_FILENAME")
));
pub static MODEL: LazyLock<Mutex<Session>> = LazyLock::new(|| {
    let session = Session::builder()
        .expect("Unable to create ONNX Runtime session builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("Unable to set optimization level")
        .with_intra_threads(4)
        .expect("Unable to set intra threads")
        .commit_from_memory(MODEL_BYTES)
        .expect("Unable to load model from static bytes");

    Mutex::new(session)
});

pub static TOKENIZER_BYTES: &[u8] = include_bytes!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("TOKENIZER_FILENAME"),
));
pub static PRE_PROCESSOR: LazyLock<TfidfVectorizer> = LazyLock::new(|| {
    TfidfVectorizer::from_bytes(TOKENIZER_BYTES)
        .expect("Unable to load tokenizer from static bytes")
});
