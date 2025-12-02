use core::fmt;

/// Binary classification result for text.
///
/// Represents whether text is classified as human-written or AI-generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Classification {
    /// Text is classified as human-written
    Human,
    /// Text is classified as AI-generated
    AI,
}

impl Classification {
    /// Returns `true` if this classification is `Human`.
    #[must_use]
    pub fn is_human(&self) -> bool {
        matches!(self, Self::Human)
    }

    /// Returns `true` if this classification is `AI`.
    #[must_use]
    pub fn is_ai(&self) -> bool {
        matches!(self, Self::AI)
    }
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Human => write!(f, "Human"),
            Self::AI => write!(f, "AI"),
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
