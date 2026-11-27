//! Runtime backend selection for the training binary.
//!
//! Burn 0.22 picks the backend at runtime through a [`Device`]; cargo features only decide which
//! backends are compiled in. Several can be enabled together and `--backend` (or the default
//! priority below) chooses among them: cuda > rocm > metal > vulkan > wgpu > flex. Flex is always
//! compiled in, so plain builds and tests work everywhere.
//!
//! The float element is the device default (f32). A pure-f16 element is deliberately not used:
//! without loss scaling this recipe overflows to NaN from epoch 1 (measured on a full-data Metal
//! run).

use burn::tensor::Device;

/// A backend this binary can train on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Backend {
    /// NVIDIA GPU through CUDA.
    #[cfg(feature = "cuda")]
    Cuda,
    /// AMD GPU through ROCm.
    #[cfg(feature = "rocm")]
    Rocm,
    /// Apple GPU through Metal.
    #[cfg(feature = "metal")]
    Metal,
    /// GPU through Vulkan.
    #[cfg(feature = "vulkan")]
    Vulkan,
    /// GPU through WGPU's automatically chosen graphics API.
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// Burn's pure-Rust CPU backend.
    Flex,
}

impl Backend {
    /// Every backend compiled into this binary, in default-priority order.
    #[must_use]
    pub(crate) fn available() -> Vec<Self> {
        vec![
            #[cfg(feature = "cuda")]
            Self::Cuda,
            #[cfg(feature = "rocm")]
            Self::Rocm,
            #[cfg(feature = "metal")]
            Self::Metal,
            #[cfg(feature = "vulkan")]
            Self::Vulkan,
            #[cfg(feature = "wgpu")]
            Self::Wgpu,
            Self::Flex,
        ]
    }

    /// The highest-priority backend compiled in.
    #[must_use]
    pub(crate) fn best() -> Self {
        Self::available()[0]
    }

    /// The name used on the command line and in logs.
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
            #[cfg(feature = "rocm")]
            Self::Rocm => "rocm",
            #[cfg(feature = "metal")]
            Self::Metal => "metal",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "vulkan",
            #[cfg(feature = "wgpu")]
            Self::Wgpu => "wgpu",
            Self::Flex => "flex",
        }
    }

    /// Look a backend up by name among those compiled in.
    pub(crate) fn from_name(name: &str) -> Result<Self, String> {
        Self::available()
            .into_iter()
            .find(|b| b.name() == name)
            .ok_or_else(|| {
                let names: Vec<_> = Self::available().into_iter().map(Self::name).collect();
                format!(
                    "backend `{name}` is not compiled into this binary (available: {})",
                    names.join(", ")
                )
            })
    }

    /// The device to run on. The first device of the backend's kind.
    #[must_use]
    pub(crate) fn device(self) -> Device {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => Device::cuda(0),
            #[cfg(feature = "rocm")]
            Self::Rocm => Device::rocm(0),
            #[cfg(feature = "metal")]
            Self::Metal => Device::metal(burn::tensor::DeviceKind::default()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan => Device::vulkan(burn::tensor::DeviceKind::default()),
            #[cfg(feature = "wgpu")]
            Self::Wgpu => Device::wgpu(burn::tensor::DeviceKind::default()),
            Self::Flex => Device::flex(),
        }
    }
}
