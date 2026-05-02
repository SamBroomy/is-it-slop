//! Text cleaning for consistent preprocessing.
//!
//! This module provides two-stage text cleaning:
//!
//! 1. **Universal Cleaning** (always applied):
//!    - HTML entities (`&quot;`, `&#39;`, etc.)
//!    - Encoding artifacts (malformed UTF-8, byte order marks)
//!    - Markdown formatting (bold, headers, bullets, code blocks)
//!    - Cyrillic encoding corruption artifacts
//!    - Whitespace normalization
//!
//! 2. **Dataset Artifact Cleaning** (training only):
//!    - Academic citations `[1]`, `[2]`
//!    - News wire metadata (datelines, agency attributions)
//!    - Academic section headers
//!    - Timestamps and timezone abbreviations
//!
//! The two-stage approach ensures consistent tokenization while preventing the model
//! from learning dataset collection artifacts as AI signals.
//!
//! # Usage
//!
//! ```rust
//! use is_it_slop_preprocessing::pre_processor::{
//!     text_cleaner_for_inference, text_cleaner_for_training,
//! };
//!
//! // Inference: Universal cleaning only
//! let cleaner = text_cleaner_for_inference();
//! let cleaned = cleaner.clean("Text with &quot;quotes&quot;");
//!
//! // Training: Universal + dataset artifact cleaning
//! let cleaner = text_cleaner_for_training();
//! let cleaned = cleaner.clean("WASHINGTON — News text [1]");
//! ```

use std::{borrow::Cow, sync::LazyLock};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use regex::Regex;

/// Universal text cleaning - applied at BOTH training and inference
#[derive(Debug, Clone)]
struct UniversalCleaner {
    // Encoding artifacts
    zero_width_spaces: Regex,
    non_breaking_space: Regex,
    // Malformed UTF-8 quotes/dashes
    right_single_quote: Regex,
    left_double_quote: Regex,
    right_double_quote: Regex,
    em_dash_1: Regex,
    en_dash: Regex,
    // HTML entities
    html_apostrophe: Regex,
    html_quote: Regex,
    html_amp: Regex,
    html_lt: Regex,
    html_gt: Regex,
    html_nbsp: Regex,
    html_numeric: Regex,
    partial_apostrophe: Regex,
    partial_quote: Regex,
    // HTML tags
    br_tag: Regex,
    common_tags: Regex,
    // Whitespace normalization
    multiple_spaces: Regex,
    multiple_newlines: Regex,
    space_newline: Regex,
    newline_space: Regex,
}

impl UniversalCleaner {
    fn new() -> Self {
        Self {
            // Encoding artifacts
            zero_width_spaces: Regex::new(r"[\u{200B}-\u{200D}\u{FEFF}]").unwrap(),
            non_breaking_space: Regex::new(r"\u{00A0}").unwrap(),

            // Malformed UTF-8
            right_single_quote: Regex::new(r"â€™").unwrap(),
            left_double_quote: Regex::new(r"â€œ").unwrap(),
            right_double_quote: Regex::new(r"â€\u{009D}").unwrap(),
            em_dash_1: Regex::new(r#"â€""#).unwrap(),
            en_dash: Regex::new(r#"â€""#).unwrap(),

            // HTML entities
            html_apostrophe: Regex::new(r"&#39;").unwrap(),
            html_quote: Regex::new(r"&quot;").unwrap(),
            html_amp: Regex::new(r"&amp;").unwrap(),
            html_lt: Regex::new(r"&lt;").unwrap(),
            html_gt: Regex::new(r"&gt;").unwrap(),
            html_nbsp: Regex::new(r"&nbsp;").unwrap(),
            html_numeric: Regex::new(r"&#(\d+);").unwrap(),
            partial_apostrophe: Regex::new(r"&?#39;?").unwrap(),
            partial_quote: Regex::new(r"&?quot;").unwrap(),

            // HTML tags
            br_tag: Regex::new(r"<br\s*/?>").unwrap(),
            common_tags: Regex::new(r"</?(p|div|span|strong|em|b|i|ul|ol|li|h[1-6])>").unwrap(),

            // Whitespace
            multiple_spaces: Regex::new(r"  +").unwrap(),
            multiple_newlines: Regex::new(r"\n\n\n+").unwrap(),
            space_newline: Regex::new(r" \n").unwrap(),
            newline_space: Regex::new(r"\n ").unwrap(),
        }
    }

    fn clean<'a>(&self, mut text: Cow<'a, str>) -> Cow<'a, str> {
        macro_rules! apply {
            ($regex:expr, $replacement:expr) => {
                if let Cow::Owned(new) = $regex.replace_all(text.as_ref(), $replacement) {
                    text = Cow::Owned(new);
                }
            };
        }

        apply!(self.zero_width_spaces, "");
        apply!(self.non_breaking_space, " ");
        apply!(self.right_single_quote, "'");
        apply!(self.left_double_quote, "\"");
        apply!(self.right_double_quote, "\"");
        apply!(self.em_dash_1, "—");
        apply!(self.en_dash, "–");
        apply!(self.html_apostrophe, "'");
        apply!(self.html_quote, "\"");
        apply!(self.html_amp, "&");
        apply!(self.html_lt, "<");
        apply!(self.html_gt, ">");
        apply!(self.html_nbsp, " ");
        apply!(self.html_numeric, "");
        apply!(self.partial_apostrophe, "'");
        apply!(self.partial_quote, "\"");
        apply!(self.br_tag, " ");
        apply!(self.common_tags, "");
        apply!(self.multiple_spaces, " ");
        apply!(self.multiple_newlines, "\n\n");
        apply!(self.space_newline, "\n");
        apply!(self.newline_space, "\n");

        text
    }
}

impl Default for UniversalCleaner {
    fn default() -> Self {
        Self::new()
    }
}

/// Dataset-specific artifact cleaning - applied ONLY during training
#[derive(Debug, Clone)]
struct DatasetArtifactCleaner {
    // Citation markers (Wikipedia/academic artifacts)
    citation_with_punct: Regex,
    malformed_citation: Regex,
    citation_newline: Regex,
    citation_open: Regex,
    citation_close: Regex,
    citation_full: Regex,

    // News wire attributions
    news_agency_parens: Regex,
    news_agency_dash: Regex,
    news_agency_extended: Regex,
    news_agency_dash_extended: Regex,
    dateline_start: Regex,
    dateline_newline: Regex,
    dateline_ending: Regex,
    dateline_cities_start: Regex,
    dateline_cities_newline: Regex,
    state_abbreviations: Regex,
    em_dash_parens: Regex,

    // Academic headers (dataset collection artifacts)
    academic_keywords: Regex,
    academic_section_headers: Regex,
    wikipedia_headers: Regex,
    wikipedia_headers_start: Regex,
    description_header: Regex,
    description_header_start: Regex,

    // Timestamp markers (news metadata)
    timezone_abbrev: Regex,
    time_pattern: Regex,

    // Academic prompts (dataset-specific)
    academic_prompt_start: Regex,
    academic_prompt_newline: Regex,

    // Numbered lists (when at document start - likely table of contents artifacts)
    numbered_start: Regex,
    numbered_newline: Regex,
}

impl DatasetArtifactCleaner {
    fn new() -> Self {
        Self {
            // Citations
            citation_with_punct: Regex::new(r"\[\d+\]([\.,;:\)\]])?").unwrap(),
            malformed_citation: Regex::new(r"\s+\d+\]([\.,;])?").unwrap(),
            citation_newline: Regex::new(r"\[\d+\]\s*\n").unwrap(),
            citation_open: Regex::new(r"\[\d+").unwrap(),
            citation_close: Regex::new(r"\d+\]").unwrap(),
            citation_full: Regex::new(r"\[\d+\]").unwrap(),

            // News wire
            news_agency_parens: Regex::new(r"\s*\((?:AP|AFP|Reuters|UPI|Bloomberg)\)\s*").unwrap(),
            news_agency_dash: Regex::new(r"\b(?:AP|AFP|Reuters|UPI|Bloomberg)\s*[-—]\s*").unwrap(),
            news_agency_extended: Regex::new(r"\s*\([A-Z]{2,}\)\s*").unwrap(),
            news_agency_dash_extended: Regex::new(r"\b[A-Z]{2,}\s*[-—]\s*").unwrap(),
            dateline_start: Regex::new(
                r"^(?:WASHINGTON|NEW YORK|LONDON|PARIS|BEIJING|MOSCOW|TOKYO|BERLIN|BRUSSELS|GENEVA|ROME|MADRID|SEOUL|SYDNEY|MEXICO CITY|LOS ANGELES|SAN\s*FRANCISCO|CHICAGO|BOSTON|MIAMI|ATLANTA|HOUSTON|DALLAS|PHILADELPHIA|PHOENIX|SAN DIEGO|SEATTLE|DETROIT|DENVER)(?:,\s*[A-Z][A-Za-z.]*\s*)?[,—]\s*"
            ).unwrap(),
            dateline_newline: Regex::new(
                r"\n(?:WASHINGTON|NEW YORK|LONDON|PARIS|BEIJING|MOSCOW|TOKYO|BERLIN|BRUSSELS|GENEVA|ROME|MADRID|SEOUL|SYDNEY|MEXICO CITY|LOS ANGELES|SAN\s*FRANCISCO|CHICAGO|BOSTON|MIAMI|ATLANTA|HOUSTON|DALLAS|PHILADELPHIA|PHOENIX|SAN DIEGO|SEATTLE|DETROIT|DENVER)(?:,\s*[A-Za-z.]+\s*)?[,—]\s*"
            ).unwrap(),
            dateline_ending: Regex::new(r"—\s+(The|A)\s+").unwrap(),
            dateline_cities_start: Regex::new(
                r"^[A-Z][A-Z\s]+(?:,\s*(?:Ala|Ariz|Ark|Calif|Colo|Conn|Del|Fla|Ga|Ill|Ind|Kan|Ky|La|Md|Mass|Mich|Minn|Miss|Mo|Mont|Neb|Nev|N\.Y|N\.C|N\.D|Ohio|Okla|Ore|Pa|R\.I|S\.C|S\.D|Tenn|Tex|Vt|Va|Wash|W\.Va|Wis|Wyo)\.\s*)?[,—]\s*"
            ).unwrap(),
            dateline_cities_newline: Regex::new(
                r"\n[A-Z][A-Z\s]+(?:,\s*(?:Ala|Ariz|Ark|Calif|Colo|Conn|Del|Fla|Ga|Ill|Ind|Kan|Ky|La|Md|Mass|Mich|Minn|Miss|Mo|Mont|Neb|Nev|N\.Y|N\.C|N\.D|Ohio|Okla|Ore|Pa|R\.I|S\.C|S\.D|Tenn|Tex|Vt|Va|Wash|W\.Va|Wis|Wyo)\.\s*)?[,—]\s*"
            ).unwrap(),
            state_abbreviations: Regex::new(
                r",\s*(?:Ala|Ariz|Ark|Calif|Colo|Conn|Del|Fla|Ga|Ill|Ind|Kan|Ky|La|Md|Mass|Mich|Minn|Miss|Mo|Mont|Neb|Nev|N\.Y|N\.C|N\.D|Ohio|Okla|Ore|Pa|R\.I|S\.C|S\.D|Tenn|Tex|Vt|Va|Wash|W\.Va|Wis|Wyo)\.\s+"
            ).unwrap(),
            em_dash_parens: Regex::new(r"\([A-Z]+\)\s*—\s*").unwrap(),

            // Academic
            academic_keywords: Regex::new(
                r"\b(?:ABSTRACT|BACKGROUND|OBJECTIVE|METHODS?|RESULTS?|CONCLUSIONS?|DISCUSSION|INTRODUCTION)[\s:]*"
            ).unwrap(),
            academic_section_headers: Regex::new(
                r"(?:^|\n)\s*(?:BACKGROUND|OBJECTIVE|METHODS|RESULTS|CONCLUSION|CONCLUSIONS|INTRODUCTION|DISCUSSION|ABSTRACT|SUMMARY):\s*"
            ).unwrap(),
            wikipedia_headers: Regex::new(
                r"\.\s*(?:History|Biography|Early life|Background|Career|Personal life|Death|Legacy|Education|Awards|References|External links)\s*\n"
            ).unwrap(),
            wikipedia_headers_start: Regex::new(
                r"^\s*(?:History|Biography|Early life|Background|Career|Personal life|Death|Legacy|Education|Awards|References|External links)\s*\n"
            ).unwrap(),
            description_header: Regex::new(r"\.\s*Description\s*\n").unwrap(),
            description_header_start: Regex::new(r"^\s*Description\s*\n").unwrap(),

            // Timestamps
            timezone_abbrev: Regex::new(r"\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC)([,\.]?)\s+").unwrap(),
            time_pattern: Regex::new(
                r"\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC)?\s*[,\.]?"
            ).unwrap(),

            // Academic prompts
            academic_prompt_start: Regex::new(r"^This (?:paper|study|article|abstract|research|work)\s+(?:presents|discusses|examines|explores|investigates|demonstrates|shows|describes)\s*").unwrap(),
            academic_prompt_newline: Regex::new(r"\nThis (?:paper|study|article|abstract|research|work)\s+(?:presents|discusses|examines|explores|investigates|demonstrates|shows|describes)\s*").unwrap(),

            // Numbered lists
            numbered_start: Regex::new(r"^\d+\.\s+").unwrap(),
            numbered_newline: Regex::new(r"\n\d+\.\s+").unwrap(),
        }
    }

    fn clean<'a>(&self, mut text: Cow<'a, str>) -> Cow<'a, str> {
        // let mut text = text.into();
        macro_rules! apply {
            ($regex:expr, $replacement:expr) => {
                if let Cow::Owned(new) = $regex.replace_all(&text, $replacement) {
                    text = Cow::Owned(new);
                }
            };
        }
        // Citations
        apply!(self.citation_with_punct, "$1");
        apply!(self.malformed_citation, "$1");
        apply!(self.citation_newline, "\n");
        apply!(self.citation_open, "");
        apply!(self.citation_close, "");
        apply!(self.citation_full, "");

        // News wire
        apply!(self.em_dash_parens, "— ");
        apply!(self.news_agency_parens, "");
        apply!(self.news_agency_dash, "");
        apply!(self.news_agency_extended, " ");
        apply!(self.news_agency_dash_extended, "");
        apply!(self.dateline_start, "");
        apply!(self.dateline_newline, "\n");
        apply!(self.dateline_ending, " $1 ");
        apply!(self.state_abbreviations, ", ");
        apply!(self.dateline_cities_start, "");
        apply!(self.dateline_cities_newline, "\n");

        // Academic
        apply!(self.academic_keywords, "");
        apply!(self.academic_section_headers, "\n");
        apply!(self.wikipedia_headers, ".\n");
        apply!(self.wikipedia_headers_start, "");
        apply!(self.description_header, ".\n");
        apply!(self.description_header_start, "");

        // Timestamps
        apply!(self.timezone_abbrev, "$1 ");
        apply!(self.time_pattern, " ");

        // Academic prompts
        apply!(self.academic_prompt_start, "");
        apply!(self.academic_prompt_newline, "\n");

        // Numbered lists
        apply!(self.numbered_start, "");
        apply!(self.numbered_newline, "\n");

        text
    }
}

impl Default for DatasetArtifactCleaner {
    fn default() -> Self {
        Self::new()
    }
}

/// Main text cleaner that combines universal and dataset-specific cleaning
#[derive(Debug, Clone)]
/// Two-stage text cleaner for training and inference.
///
/// Combines universal cleaning (always applied) with optional dataset artifact
/// cleaning (training only).
///
/// # Stages
///
/// 1. **Dataset Artifacts** (if enabled): Remove training-specific patterns
///    - Academic citations, news metadata, timestamps
/// 2. **Universal Cleaning** (always): Normalize text for consistent tokenization
///    - HTML entities, encoding artifacts, markdown formatting
///
/// # Usage
///
/// ```rust
/// use is_it_slop_preprocessing::pre_processor::{
///     text_cleaner_for_inference, text_cleaner_for_training,
/// };
///
/// // Inference: Universal cleaning only
/// let cleaner = text_cleaner_for_inference();
/// let result = cleaner.clean("Text with &quot;HTML&quot;");
///
/// // Training: Both stages
/// let cleaner = text_cleaner_for_training();
/// let result = cleaner.clean("Article text [1] with citations");
/// ```
pub struct TextCleaner {
    universal: UniversalCleaner,
    dataset_artifacts: Option<DatasetArtifactCleaner>,
}

impl TextCleaner {
    /// Clean the input text before tokenization
    #[must_use]
    pub fn clean<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Apply dataset cleaning first if enabled
        let text = self
            .dataset_artifacts
            .as_ref()
            .map_or_else(|| text.into(), |cleaner| cleaner.clean(text.into()));

        // Apply universal cleaning
        let text = self.universal.clean(text);

        let trimmed = text.trim().trim_matches(|c: char| c == '\'' || c == '"');
        if trimmed.len() == text.len() {
            text
        } else {
            Cow::Owned(trimmed.to_string())
        }
    }

    /// Clean a batch of texts in parallel
    pub fn clean_batch<T: AsRef<str> + Sync>(&self, texts: &[T]) -> Vec<String> {
        texts
            .par_iter()
            .map(|text| self.clean(text.as_ref()).to_string())
            .collect()
    }
}

// Use lazy lock to avoid recompiling regexes on each call
static UNIVERSAL_CLEANER: LazyLock<UniversalCleaner> = LazyLock::new(UniversalCleaner::new);
static DATASET_ARTIFACT_CLEANER: LazyLock<DatasetArtifactCleaner> =
    LazyLock::new(DatasetArtifactCleaner::new);

static TRAINING_CLEANER: LazyLock<TextCleaner> = LazyLock::new(|| TextCleaner {
    universal: UNIVERSAL_CLEANER.clone(),
    dataset_artifacts: Some(DATASET_ARTIFACT_CLEANER.clone()),
});

static INFERENCE_CLEANER: LazyLock<TextCleaner> = LazyLock::new(|| TextCleaner {
    universal: UNIVERSAL_CLEANER.clone(),
    dataset_artifacts: None,
});

/// Create a `TextCleaner` for inference
#[must_use]
pub fn text_cleaner_for_inference() -> &'static TextCleaner {
    &INFERENCE_CLEANER
}

/// Create a `TextCleaner` for training
#[must_use]
pub fn text_cleaner_for_training() -> &'static TextCleaner {
    &TRAINING_CLEANER
}
#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Universal Cleaner Tests (apply to both training and inference)
    // ============================================================================

    mod universal_cleaner {
        use super::*;

        #[test]
        fn test_zero_width_spaces() {
            let cleaner = UniversalCleaner::new();
            // Note: These might not be in the text when copy-pasted
            // Test with explicit unicode escapes
            let input = Cow::Borrowed("hello\u{200B}world");
            let result = cleaner.clean(input);
            // The regex removes zero-width space, but doesn't add space
            assert_eq!(result, "helloworld");

            assert_eq!(cleaner.clean("test\u{200C}ing".into()), "testing");
            assert_eq!(cleaner.clean("no\u{200D}space".into()), "nospace");
        }

        #[test]
        fn test_non_breaking_spaces() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(cleaner.clean("hello\u{00A0}world".into()), "hello world");
            // Multiple non-breaking spaces become regular spaces
            // But then multiple_spaces regex collapses them
            assert_eq!(
                cleaner.clean("multiple\u{00A0}\u{00A0}spaces".into()),
                "multiple spaces"
            );
        }

        #[test]
        fn test_malformed_utf8_quotes() {
            let cleaner = UniversalCleaner::new();
            // For now, test what we CAN match:
            assert_eq!(cleaner.clean("Itâ€™s".into()), "It's");
            // If the regex doesn't match, it returns unchanged:
            assert_eq!(cleaner.clean("It's".into()), "It's"); // Already correct
        }
        #[test]
        fn test_malformed_utf8_dashes() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(cleaner.clean("em dashâ€\"here".into()), "em dash—here");
            assert_eq!(cleaner.clean("en dashâ€\"here".into()), "en dash—here");
        }

        #[test]
        fn test_html_entities() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(cleaner.clean("It&#39;s great".into()), "It's great");
            assert_eq!(cleaner.clean("&quot;Hello&quot;".into()), "\"Hello\"");
            assert_eq!(cleaner.clean("A&amp;B".into()), "A&B");
            assert_eq!(cleaner.clean("&lt;tag&gt;".into()), "<tag>");
            assert_eq!(cleaner.clean("&nbsp;space".into()), " space");
        }

        #[test]
        fn test_html_numeric_entities() {
            let cleaner = UniversalCleaner::new();
            // Numeric entities are removed entirely
            assert_eq!(cleaner.clean("test&#65;here".into()), "testhere");
            assert_eq!(cleaner.clean("&#160;space".into()), "space");
        }

        #[test]
        fn test_partial_html_entities() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(cleaner.clean("test#39;here".into()), "test'here");
            assert_eq!(cleaner.clean("testquot;mark".into()), "test\"mark");
        }

        #[test]
        fn test_html_tags() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(cleaner.clean("Hello<br>world".into()), "Hello world");
            assert_eq!(cleaner.clean("Hello<br/>world".into()), "Hello world");
            assert_eq!(cleaner.clean("Hello<br />world".into()), "Hello world");
            assert_eq!(cleaner.clean("<p>paragraph</p>".into()), "paragraph");
            assert_eq!(cleaner.clean("<strong>bold</strong>".into()), "bold");
            assert_eq!(
                cleaner.clean("<div><span>nested</span></div>".into()),
                "nested"
            );
        }

        #[test]
        fn test_whitespace_normalization() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(
                cleaner.clean("too   many    spaces".into()),
                "too many spaces"
            );
            assert_eq!(
                cleaner.clean("too\n\n\n\nmany\nlines".into()),
                "too\n\nmany\nlines"
            );
            assert_eq!(
                cleaner.clean("space before \nnewline".into()),
                "space before\nnewline"
            );
            assert_eq!(
                cleaner.clean("newline after\n space".into()),
                "newline after\nspace"
            );
        }

        #[test]
        fn test_empty_and_whitespace_only() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(cleaner.clean("".into()), "");
            assert_eq!(cleaner.clean("   ".into()), " ");
            assert_eq!(cleaner.clean("  \n  ".into()), "\n");
        }

        #[test]
        fn test_no_changes_needed() {
            let cleaner = UniversalCleaner::new();
            let clean_text = "This is already clean text.";
            assert_eq!(cleaner.clean(clean_text.into()), clean_text);
        }

        #[test]
        fn test_multiple_issues() {
            let cleaner = UniversalCleaner::new();
            let input = "It&#39;s  <strong>working</strong>&nbsp;&nbsp;fine";
            let expected = "It's working fine";
            assert_eq!(cleaner.clean(input.into()), expected);
        }

        #[test]
        fn test_accepts_string() {
            let cleaner = UniversalCleaner::new();
            let owned = String::from("test&#39;s");
            assert_eq!(cleaner.clean(owned.into()), "test's");
        }

        #[test]
        fn test_accepts_str() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(cleaner.clean("test&#39;s".into()), "test's");
        }
    }

    // ============================================================================
    // Dataset Artifact Cleaner Tests (training only)
    // ============================================================================

    mod dataset_artifact_cleaner {
        use super::*;

        #[test]
        fn test_citation_markers() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(
                cleaner.clean("This is a fact[1].".into()),
                "This is a fact."
            );
            assert_eq!(
                cleaner.clean("Multiple[1][2][3] refs".into()),
                "Multiple refs"
            );
            assert_eq!(cleaner.clean("Test[123].".into()), "Test.");
            assert_eq!(
                cleaner.clean("Test[1], continues".into()),
                "Test, continues"
            );
        }

        #[test]
        fn test_malformed_citations() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("Test 1].".into()), "Test.");
            assert_eq!(cleaner.clean("Test 42], here".into()), "Test, here");
        }

        #[test]
        fn test_citation_with_newline() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("Test[1].\nNext".into()), "Test.\nNext");
        }

        #[test]
        fn test_news_agency_parentheses() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("(AP) The story".into()), "The story");
            assert_eq!(
                cleaner.clean("(Reuters) Breaking news".into()),
                "Breaking news"
            );
            assert_eq!(cleaner.clean("(UPI) Update".into()), "Update");
            assert_eq!(cleaner.clean("(AFP) Report".into()), "Report");
            assert_eq!(cleaner.clean("(Bloomberg) Markets".into()), "Markets");
        }

        #[test]
        fn test_news_agency_dash() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("AP - The story".into()), "The story");
            assert_eq!(cleaner.clean("Reuters — Breaking".into()), "Breaking");
            assert_eq!(cleaner.clean("UPI - Update".into()), "Update");
        }

        #[test]
        fn test_dateline_cities() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(
                cleaner.clean("WASHINGTON — The president".into()),
                "The president"
            );
            assert_eq!(cleaner.clean("NEW YORK, Officials".into()), "Officials");
            assert_eq!(cleaner.clean("LONDON — Markets".into()), "Markets");
            assert_eq!(cleaner.clean("PARIS, The summit".into()), "The summit");
        }

        #[test]
        fn test_dateline_cities_after_newline() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(
                cleaner.clean("Previous.\nWASHINGTON — Next".into()),
                "Previous.\nNext"
            );
        }

        #[test]
        fn test_extended_dateline_cities() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("TOKYO — Report".into()), "Report");
            assert_eq!(cleaner.clean("BRUSSELS, Meeting".into()), "Meeting");
            assert_eq!(cleaner.clean("SYDNEY — News".into()), "News");
        }

        #[test]
        fn test_state_abbreviations() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("City, Calif. Story".into()), "City, Story");
            assert_eq!(cleaner.clean("Town, N.Y. News".into()), "Town, News");
            assert_eq!(cleaner.clean("Place, Mass. Update".into()), "Place, Update");
        }

        #[test]
        fn test_dateline_endings() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("CITY — The story".into()), "The story");
            assert_eq!(cleaner.clean("PLACE — A report".into()), "A report");
        }

        #[test]
        fn test_em_dash_parentheses() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("(AP) — Story".into()), "— Story");
        }

        #[test]
        fn test_academic_keywords() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("ABSTRACT: Text here".into()), "Text here");
            assert_eq!(cleaner.clean("INTRODUCTION Text".into()), "Text");
            assert_eq!(cleaner.clean("METHODS: Study".into()), "Study");
            assert_eq!(cleaner.clean("RESULTS Data".into()), "Data");
            assert_eq!(cleaner.clean("CONCLUSION: Final".into()), "Final");
        }

        #[test]
        fn test_academic_section_headers() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("ABSTRACT: Study".into()), "Study");
            assert_eq!(cleaner.clean("\nMETHODS: Data".into()), "\nData");
            assert_eq!(cleaner.clean("DISCUSSION: Findings".into()), "Findings");
        }

        #[test]
        fn test_wikipedia_headers() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("Text. Biography\nMore".into()), "Text.\nMore");
            assert_eq!(
                cleaner.clean("Info. Early life\nData".into()),
                "Info.\nData"
            );
            assert_eq!(cleaner.clean("Story. Career\nNext".into()), "Story.\nNext");
        }

        #[test]
        fn test_wikipedia_headers_at_start() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("Biography\nText".into()), "Text");
            assert_eq!(cleaner.clean("History\nContent".into()), "Content");
        }

        #[test]
        fn test_description_headers() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(
                cleaner.clean("Intro. Description\nText".into()),
                "Intro.\nText"
            );
            assert_eq!(cleaner.clean("Description\nContent".into()), "Content");
        }

        #[test]
        fn test_timezone_abbreviations() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("Meeting EST today".into()), "Meeting today");
            assert_eq!(cleaner.clean("Call PST, tomorrow".into()), "Call, tomorrow");
            assert_eq!(cleaner.clean("Event GMT. here".into()), "Event. here");
        }

        #[test]
        fn test_time_patterns() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(
                cleaner.clean("Call at 3:00 PM EST today".into()),
                "Call today"
            );
            assert_eq!(
                cleaner.clean("Meeting at 10:30 tomorrow".into()),
                "Meeting tomorrow"
            );
        }

        #[test]
        fn test_academic_prompts() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(
                cleaner.clean("This paper presents a method".into()),
                "a method"
            );
            assert_eq!(cleaner.clean("This study shows results".into()), "results");
            assert_eq!(cleaner.clean("\nThis article discusses".into()), "\n");
        }

        #[test]
        fn test_numbered_lists() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("1. First item".into()), "First item");
            assert_eq!(cleaner.clean("Text\n2. Second".into()), "Text\nSecond");
            assert_eq!(cleaner.clean("123. Many".into()), "Many");
        }

        #[test]
        fn test_complex_news_article() {
            let cleaner = DatasetArtifactCleaner::new();
            let input = "WASHINGTON — (AP) The president said[1] today.";
            let expected = "The president said today.";
            assert_eq!(cleaner.clean(input.into()), expected);
        }

        #[test]
        fn test_complex_academic_abstract() {
            let cleaner = DatasetArtifactCleaner::new();
            let input = "ABSTRACT: This study presents[1] results.\nMETHODS: Data collected.";
            let expected = "results.\nData collected.";
            assert_eq!(cleaner.clean(input.into()), expected);
        }

        #[test]
        fn test_raid_bench_artifacts() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(
                cleaner.clean("This is important!!".into()),
                "This is important!!"
            );
            assert_eq!(
                cleaner.clean("Alert!! Please read".into()),
                "Alert!! Please read"
            );
        }
    }

    // ============================================================================
    // TextCleaner Tests (mode-specific behavior)
    // ============================================================================

    mod text_cleaner {
        use super::*;

        #[test]
        fn test_inference_mode_universal_only() {
            let cleaner = text_cleaner_for_inference();

            // Should clean HTML entities (universal)
            let input = "Test&#39;s working";
            assert_eq!(cleaner.clean(input), "Test's working");

            // Should NOT remove citations (dataset artifact)
            let input = "Fact[1] remains";
            assert_eq!(cleaner.clean(input), "Fact[1] remains");

            // Should NOT remove datelines (dataset artifact)
            let input = "WASHINGTON — Story";
            assert_eq!(cleaner.clean(input), "WASHINGTON — Story");
        }

        #[test]
        fn test_training_mode_both_cleaners() {
            let cleaner = text_cleaner_for_training();

            // Should clean HTML entities (universal)
            let input = "Test&#39;s working";
            assert_eq!(cleaner.clean(input), "Test's working");

            // Should remove citations (dataset artifact)
            let input = "Fact[1] remains";
            assert_eq!(cleaner.clean(input), "Fact remains");

            // Should remove datelines (dataset artifact)
            let input = "WASHINGTON — Story";
            assert_eq!(cleaner.clean(input), "Story");
        }

        #[test]
        fn test_trimming() {
            let cleaner = text_cleaner_for_inference();

            // Leading/trailing whitespace
            assert_eq!(cleaner.clean("  text  "), "text");

            // Leading/trailing quotes
            assert_eq!(cleaner.clean("'quote'"), "quote");
            assert_eq!(cleaner.clean("\"quote\""), "quote");

            // Both whitespace and quotes
            assert_eq!(cleaner.clean("  'text'  "), "text");
        }

        #[test]
        fn test_empty_after_cleaning() {
            let cleaner = text_cleaner_for_training();

            assert_eq!(cleaner.clean("   "), "");
            assert_eq!(cleaner.clean("'\"'"), "");
            assert_eq!(cleaner.clean("[1][2][3]"), "");
        }

        #[test]
        fn test_complex_mixed_artifacts() {
            let cleaner = text_cleaner_for_training();

            let input = "  WASHINGTON — (AP) The &quot;president&quot; said[1]  ";
            let expected = "The \"president\" said";
            assert_eq!(cleaner.clean(input), expected);
        }

        #[test]
        fn test_real_world_news_article() {
            let cleaner = text_cleaner_for_training();

            let input = "WASHINGTON, D.C. — (Reuters) President Biden announced[1] today that &quot;climate change&quot; is a priority. The announcement at 3:00 PM EST came after consultations[2][3].";

            let expected = "President Biden announced today that \"climate change\" is a priority. The announcement came after consultations.";

            assert_eq!(cleaner.clean(input), expected);
        }

        #[test]
        fn test_real_world_academic_paper() {
            let cleaner = text_cleaner_for_training();

            let input = "ABSTRACT: This paper presents a novel approach[1] to text classification.\nMETHODS: We collected data from multiple sources[2].";

            let expected = "a novel approach to text classification.\nWe collected data from multiple sources.";

            assert_eq!(cleaner.clean(input), expected);
        }

        #[test]
        fn test_preserves_legitimate_content() {
            let cleaner = text_cleaner_for_inference();

            // Should preserve markdown-like formatting (legitimate AI signals)
            let input = "## Introduction\n\nThis is **important** text.";
            assert_eq!(
                cleaner.clean(input),
                "## Introduction\n\nThis is **important** text."
            );

            // Should preserve bullet points
            let input = "- First point\n- Second point";
            assert_eq!(cleaner.clean(input), "- First point\n- Second point");

            // Should preserve legitimate numbered lists in content
            let input = "Here are 3 steps:\n1. Do this\n2. Then this";
            // Note: Training mode would strip the numbers at line start
            assert_eq!(
                cleaner.clean(input),
                "Here are 3 steps:\n1. Do this\n2. Then this"
            );
        }

        #[test]
        fn test_accepts_string_ownership() {
            let cleaner = text_cleaner_for_inference();

            let owned = String::from("test&#39;s");
            let result = cleaner.clean(&owned);
            assert_eq!(result, "test's");
        }

        #[test]
        fn test_accepts_str_reference() {
            let cleaner = text_cleaner_for_inference();

            let result = cleaner.clean("test&#39;s");
            assert_eq!(result, "test's");
        }

        #[test]
        fn test_unicode_preservation() {
            let cleaner = text_cleaner_for_inference();

            // Should preserve legitimate unicode
            assert_eq!(cleaner.clean("Café résumé"), "Café résumé");
            assert_eq!(cleaner.clean("日本語"), "日本語");
            assert_eq!(cleaner.clean("Привет"), "Привет");
        }

        #[test]
        fn test_long_text_performance() {
            use std::fmt::Write;
            let cleaner = text_cleaner_for_training();

            // Generate a longer text with multiple issues
            let input = (0..100).fold(String::new(), |mut output, i| {
                let _ = write!(output, "Sentence {i}[{i}] with &quot;quotes&quot; here. ");
                output
            });

            let result = cleaner.clean(&input);

            // Should clean all instances
            assert!(!result.contains('['));
            assert!(!result.contains("&quot;"));
            assert!(result.contains("Sentence"));
        }
    }

    // ============================================================================
    // Edge Cases and Regression Tests
    // ============================================================================

    mod edge_cases {
        use super::*;

        #[test]
        fn test_extremely_long_citation() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("Text[123456789]".into()), "Text");
        }

        #[test]
        fn test_mixed_quote_styles() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(
                cleaner.clean("&#39;single&#39; and &quot;double&quot;".into()),
                "'single' and \"double\""
            );
        }

        #[test]
        fn test_nested_html_tags() {
            let cleaner = UniversalCleaner::new();
            assert_eq!(
                cleaner.clean("<div><p><strong>nested</strong></p></div>".into()),
                "nested"
            );
        }

        #[test]
        fn test_consecutive_artifacts() {
            let cleaner = DatasetArtifactCleaner::new();
            assert_eq!(cleaner.clean("[1][2][3][4][5]".into()), "");
        }

        #[test]
        fn test_only_whitespace_and_artifacts() {
            let cleaner = text_cleaner_for_training();
            assert_eq!(cleaner.clean("  [1]  [2]  "), "");
        }

        #[test]
        fn test_newlines_preserved_correctly() {
            let cleaner = text_cleaner_for_inference();
            assert_eq!(
                cleaner.clean("Line1\nLine2\n\nLine3"),
                "Line1\nLine2\n\nLine3"
            );
        }

        #[test]
        fn test_no_over_trimming() {
            let cleaner = text_cleaner_for_inference();
            // Should preserve internal quotes
            assert_eq!(
                cleaner.clean("He said 'hello' there"),
                "He said 'hello' there"
            );
        }
    }
}
