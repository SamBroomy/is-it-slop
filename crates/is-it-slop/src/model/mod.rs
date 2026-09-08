//! Model artifact loading and global state management.
//!
//! This module embeds the trained model bundle at compile time using `include_bytes!` macros
//! and loads it lazily on first access via [`LazyLock`].
//!
//! # Model Artifacts
//!
//! - **Weights** (`model.f16.safetensors`): the hierarchical classifier, half precision
//! - **Vocabulary** (`token_vocabulary.safetensors`): raw token id → compact embedding row
//! - **Encoder config** (`encoder_config.json`): chunking and vocabulary parameters
//! - **Model config** (`model_config.json`): network shape
//! - **Decision contract**: [`CLASSIFICATION_THRESHOLD`] and [`CALIBRATION_TEMPERATURE`],
//!   baked into `threshold.rs` by the build script
//!
//! All artifacts are downloaded from GitHub releases during build if not present locally.
//!
//! # Global Statics
//!
//! - [`MODEL`]: the classifier on the CPU backend (immutable after initialization)
//! - [`ENCODER`]: the document encoder reconstructing the bundle's exact text→grid encoding

use std::sync::LazyLock;

use burn::backend::Flex;
use burn::backend::flex::FlexDevice;
use burn_store::{HalfPrecisionAdapter, ModuleSnapshot, SafetensorsStore};
use is_it_slop_model::{HierarchicalModelConfig, HierarchicalSlopClassifier};
use is_it_slop_preprocessing::encoder::DocumentEncoder;

include!(concat!(env!("OUT_DIR"), "/threshold.rs"));

/// The inference backend: Flex, burn's pure-Rust CPU backend.
///
/// Inference must run on a non-autodiff backend — forward passes on an autodiff backend
/// accumulate graph state and leak.
pub type Backend = Flex;

/// Current model version
///
/// This is set during build time based on the model artifacts used.
/// The model version is used to ensure that the underlying model and encoder are compatible.
pub const MODEL_VERSION: &str = env!("MODEL_VERSION");

/// Raw model weights embedded at compile time (half-precision safetensors).
static WEIGHTS_BYTES: &[u8] = include_bytes!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("MODEL_WEIGHTS_FILENAME")
));

/// Raw vocabulary lookup table embedded at compile time.
static VOCABULARY_BYTES: &[u8] = include_bytes!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("VOCABULARY_FILENAME")
));

/// The bundle's encoder configuration JSON.
static ENCODER_CONFIG: &str = include_str!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("ENCODER_CONFIG_FILENAME")
));

/// The bundle's model configuration JSON.
static MODEL_CONFIG: &str = include_str!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("MODEL_CONFIG_FILENAME")
));

/// The classifier, initialized from the embedded config and weights on first access.
///
/// Burn modules are immutable at inference time, so no lock is needed; the forward pass
/// takes `&self` and is safe to share across threads.
pub static MODEL: LazyLock<HierarchicalSlopClassifier<Backend>> = LazyLock::new(|| {
    let config: HierarchicalModelConfig =
        serde_json::from_str(MODEL_CONFIG).expect("embedded model_config.json parses");
    config
        .validate()
        .expect("embedded model_config.json satisfies the model contract");
    let mut model = config.init::<Backend>(&FlexDevice);
    let mut store = SafetensorsStore::from_bytes(Some(WEIGHTS_BYTES.to_vec()))
        .with_from_adapter(HalfPrecisionAdapter::new());
    model
        .load_from(&mut store)
        .expect("embedded weights load into the configured model");
    model
});

/// The document encoder, reconstructing the exact encoding the bundle was trained with
/// (inference cleaning on for raw user input).
pub static ENCODER: LazyLock<DocumentEncoder> = LazyLock::new(|| {
    DocumentEncoder::from_bundle(VOCABULARY_BYTES, ENCODER_CONFIG)
        .expect("embedded bundle artifacts agree")
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version_format() {
        // Verify MODEL_VERSION is in semver-like format (X.Y.Z)
        assert!(!MODEL_VERSION.is_empty());

        let parts: Vec<&str> = MODEL_VERSION.split('.').collect();
        assert!(
            parts.len() >= 2,
            "Model version should have at least major.minor"
        );

        // Each part should be parseable as a number
        for part in parts {
            assert!(
                part.parse::<u32>().is_ok(),
                "Version part '{part}' should be a number"
            );
        }
    }

    #[test]
    fn test_artifacts_non_empty() {
        assert!(
            WEIGHTS_BYTES.len() > 1000,
            "Weights should be at least 1KB (sanity check)"
        );
        assert!(
            !VOCABULARY_BYTES.is_empty(),
            "Vocabulary bytes should be non-empty"
        );
        assert!(
            !ENCODER_CONFIG.is_empty(),
            "Encoder config should be non-empty"
        );
    }

    #[test]
    fn test_model_lazy_initialization() {
        use burn::module::Module;
        // Loading is fallible only on a corrupt bundle; both accesses hit the same instance.
        let params = MODEL.num_params();
        assert!(params > 100_000, "model should have real parameters");
        assert_eq!(params, MODEL.num_params());
    }

    #[test]
    fn test_encoder_matches_bundle() {
        let config: serde_json::Value = serde_json::from_str(ENCODER_CONFIG).unwrap();
        assert_eq!(
            ENCODER.chunk_size(),
            config["chunk_size"].as_u64().unwrap() as usize
        );
        assert_eq!(
            ENCODER.vocabulary().vocab_size(),
            config["vocab_size"].as_u64().unwrap() as usize
        );
    }

    #[test]
    fn test_decision_contract_valid() {
        const {
            assert!(
                CLASSIFICATION_THRESHOLD >= 0.0 && CLASSIFICATION_THRESHOLD <= 1.0,
                "Classification threshold should be in [0, 1]"
            );
        }
        const {
            assert!(
                CALIBRATION_TEMPERATURE > 0.0,
                "Calibration temperature must be positive"
            );
        }
    }
}
