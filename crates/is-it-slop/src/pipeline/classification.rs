use core::fmt;

/// Binary classification result for text.
///
/// Represents whether text is classified as human-written or AI-generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Classification {
    /// Text is classified as human-written
    Human,
    /// Text is classified as AI-generated
    AI,
}

impl Classification {
    /// Returns `true` if this classification is `Human`.
    #[must_use]
    pub const fn is_human(&self) -> bool {
        matches!(self, Self::Human)
    }

    /// Returns `true` if this classification is `AI`.
    #[must_use]
    pub const fn is_ai(&self) -> bool {
        matches!(self, Self::AI)
    }
}

impl std::ops::Not for Classification {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::Human => Self::AI,
            Self::AI => Self::Human,
        }
    }
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Human => f.pad("Human"),
            Self::AI => f.pad("AI"),
        }
    }
}

impl From<Classification> for i64 {
    fn from(class: Classification) -> Self {
        match class {
            Classification::Human => 0,
            Classification::AI => 1,
        }
    }
}
