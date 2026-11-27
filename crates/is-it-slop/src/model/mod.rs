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
//! - **Decision contract**: [`CLASSIFICATION_THRESHOLD`] and [`CALIBRATION_TEMPERATURE`], baked
//!   into `threshold.rs` by the build script
//!
//! All artifacts are downloaded from GitHub releases during build if not present locally.
//!
//! # Global Statics
//!
//! - [`MODEL`]: the classifier on the primary [`device`] (immutable after initialization), and in
//!   accelerated builds `CPU_MODEL`, the same weights on the [`cpu_device`]
//! - [`ENCODER`]: the document encoder reconstructing the bundle's exact text→grid encoding

use std::sync::LazyLock;

use burn::{
    store::{HalfPrecisionAdapter, ModuleSnapshot, SafetensorsStore},
    tensor::Device,
};
use is_it_slop_model::{HierarchicalModelConfig, HierarchicalSlopClassifier};
use is_it_slop_preprocessing::encoder::DocumentEncoder;

include!(concat!(env!("OUT_DIR"), "/threshold.rs"));

/// The CPU backend's name, for logs and benchmark labels.
pub const CPU_BACKEND: &str = if cfg!(feature = "cpu") {
    "cpu"
} else if cfg!(feature = "accelerate") {
    "ndarray-accelerate"
} else if cfg!(feature = "ndarray") {
    "ndarray"
} else if cfg!(feature = "apple-amx") {
    "flex-amx"
} else {
    "flex"
};

/// The primary backend's name, for logs and benchmark labels. Precedence: `cuda`, `rocm`, `metal`,
/// `vulkan`, `webgpu`, `wgpu` (accelerators, which set `cfg(accelerated)`), then the CPU backend.
pub const BACKEND: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "rocm") {
    "rocm"
} else if cfg!(feature = "metal") {
    "metal"
} else if cfg!(feature = "vulkan") {
    "vulkan"
} else if cfg!(feature = "webgpu") {
    "webgpu"
} else if cfg!(feature = "wgpu") {
    "wgpu"
} else {
    CPU_BACKEND
};

/// The primary inference device, chosen at compile time by cargo feature in the same precedence
/// as [`BACKEND`] (the wgpu family implies `wgpu`, so the more specific one wins).
///
/// Inference must run on a device without autodiff: forward passes with autodiff on accumulate
/// graph state and leak.
#[must_use]
pub fn device() -> Device {
    #[cfg(feature = "cuda")]
    return Device::cuda(0);
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    return Device::rocm(0);
    #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "rocm"))))]
    return Device::metal(burn::tensor::DeviceKind::default());
    #[cfg(all(
        feature = "vulkan",
        not(any(feature = "cuda", feature = "rocm", feature = "metal"))
    ))]
    return Device::vulkan(burn::tensor::DeviceKind::default());
    #[cfg(all(
        feature = "webgpu",
        not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan"
        ))
    ))]
    return Device::webgpu(burn::tensor::DeviceKind::default());
    #[cfg(all(
        feature = "wgpu",
        not(any(
            feature = "cuda",
            feature = "rocm",
            feature = "metal",
            feature = "vulkan",
            feature = "webgpu"
        ))
    ))]
    return Device::wgpu(burn::tensor::DeviceKind::default());
    #[cfg(not(accelerated))]
    return cpu_device();
}

/// The CPU device: burn-cpu (`cpu`), else ndarray (`ndarray`, which `accelerate` implies), else
/// Flex, burn's pure-Rust CPU backend. It is [`device`] itself in a CPU build and the fallback for
/// small calls in an accelerated one (see `pipeline::accelerated`).
#[must_use]
pub fn cpu_device() -> Device {
    #[cfg(feature = "cpu")]
    return Device::cpu();
    #[cfg(all(feature = "ndarray", not(feature = "cpu")))]
    return Device::ndarray();
    #[cfg(not(any(feature = "cpu", feature = "ndarray")))]
    return Device::flex();
}

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
pub static MODEL: LazyLock<HierarchicalSlopClassifier> = LazyLock::new(|| load(&device()));

/// The same classifier on the [`cpu_device`], for the calls too small to be worth a GPU pass.
/// Loaded on first such call, so a build that only ever batches never pays for it.
#[cfg(accelerated)]
pub(crate) static CPU_MODEL: LazyLock<HierarchicalSlopClassifier> =
    LazyLock::new(|| load(&cpu_device()));

/// The embedded config and weights on `device`.
fn load(device: &Device) -> HierarchicalSlopClassifier {
    let config: HierarchicalModelConfig =
        serde_json::from_str(MODEL_CONFIG).expect("embedded model_config.json parses");
    config
        .validate()
        .expect("embedded model_config.json satisfies the model contract");
    let mut model = config.init(device);
    let mut store = SafetensorsStore::from_bytes(Some(WEIGHTS_BYTES.to_vec()))
        .with_from_adapter(HalfPrecisionAdapter::new());
    model
        .load_from(&mut store)
        .expect("embedded weights load into the configured model");
    model
}

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
        assert_ne!(MODEL_VERSION, "");

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
        let chunk_size =
            usize::try_from(config["chunk_size"].as_u64().unwrap()).expect("chunk_size fits usize");
        assert_eq!(ENCODER.chunk_size(), chunk_size);
        let vocab_size =
            usize::try_from(config["vocab_size"].as_u64().unwrap()).expect("vocab_size fits usize");
        assert_eq!(ENCODER.vocabulary().vocab_size(), vocab_size);
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
