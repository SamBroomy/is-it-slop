//! Compile-time backend selection for the training binary.
//!
//! Mirrors Burn's text-classification example: each backend is a cargo feature, and when several
//! are enabled the highest-priority one wins — metal > cuda > rocm > vulkan > wgpu > tch-gpu >
//! tch-cpu > remote > flex (the no-feature fallback, so tests and plain builds always compile).
//!
//! The float element is always f32. A pure-f16 element is deliberately not offered: without loss
//! scaling this recipe overflows to NaN from epoch 1 (measured on a full-data Metal run). Burn's
//! `flex32` relaxed element is the untried middle ground if training throughput ever matters.

use burn::{backend::Autodiff, tensor::Device};

#[cfg(feature = "metal")]
mod selected {
    pub type Inner = burn::backend::Metal<f32, i32>;
    pub const NAME: &str = "metal";
}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
mod selected {
    pub type Inner = burn::backend::Cuda<f32, i32>;
    pub const NAME: &str = "cuda";
}

#[cfg(all(feature = "rocm", not(any(feature = "metal", feature = "cuda"))))]
mod selected {
    pub type Inner = burn::backend::Rocm<f32, i32>;
    pub const NAME: &str = "rocm";
}

#[cfg(all(
    feature = "vulkan",
    not(any(feature = "metal", feature = "cuda", feature = "rocm"))
))]
mod selected {
    pub type Inner = burn::backend::Vulkan<f32, i32>;
    pub const NAME: &str = "vulkan";
}

#[cfg(all(
    feature = "wgpu",
    not(any(
        feature = "metal",
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan"
    ))
))]
mod selected {
    pub type Inner = burn::backend::Wgpu<f32, i32>;
    pub const NAME: &str = "wgpu";
}

#[cfg(all(
    any(feature = "tch-gpu", feature = "tch-cpu"),
    not(any(
        feature = "metal",
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "wgpu"
    ))
))]
mod selected {
    pub type Inner = burn::backend::libtorch::LibTorch<f32>;
    #[cfg(feature = "tch-gpu")]
    pub const NAME: &str = "tch-gpu";
    #[cfg(not(feature = "tch-gpu"))]
    pub const NAME: &str = "tch-cpu";
}

#[cfg(all(
    feature = "remote",
    not(any(
        feature = "metal",
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "wgpu",
        feature = "tch-gpu",
        feature = "tch-cpu"
    ))
))]
mod selected {
    pub type Inner = burn::backend::RemoteBackend;
    pub const NAME: &str = "remote";
}

#[cfg(not(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu",
    feature = "tch-gpu",
    feature = "tch-cpu",
    feature = "remote"
)))]
mod selected {
    pub type Inner = burn::backend::Flex;
    pub const NAME: &str = "flex";
}

/// The training backend the enabled features selected.
pub type Selected = Autodiff<selected::Inner>;

/// Human-readable name of the selected backend, for logs.
pub const NAME: &str = selected::NAME;

/// The device to train on.
///
/// Every backend's default device is right except libtorch, whose default is CPU: `tch-gpu`
/// wants MPS on macOS and the first CUDA device elsewhere.
pub fn device() -> Device<Selected> {
    #[cfg(all(
        feature = "tch-gpu",
        not(any(
            feature = "metal",
            feature = "cuda",
            feature = "rocm",
            feature = "vulkan",
            feature = "wgpu"
        ))
    ))]
    {
        #[cfg(target_os = "macos")]
        return burn::backend::libtorch::LibTorchDevice::Mps;
        #[cfg(not(target_os = "macos"))]
        return burn::backend::libtorch::LibTorchDevice::Cuda(0);
    }
    // The concrete device type differs per selected backend, so the trait-level default is the
    // point here.
    #[allow(unreachable_code, clippy::default_trait_access)]
    Default::default()
}
