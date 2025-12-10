//! CLI module for the is-it-slop text classification tool.
//!
//! This module provides command-line interface functionality for detecting
//! AI-generated text. It handles argument parsing, input processing, and
//! formatted output.

use std::{collections::HashMap, io::Read, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use crate::{CLASSIFICATION_THRESHOLD, Classification, Predictor};

/// Command-line arguments structure
#[derive(Parser)]
#[command(name = "is-it-slop")]
#[command(about = "Detect AI-generated text", long_about = None)]
pub struct Cli {
    /// Text to analyze (if not provided, reads from stdin)
    #[arg(value_name = "TEXT")]
    pub text: Option<String>,

    /// Read text from file
    #[arg(short, long, value_name = "PATH", conflicts_with = "text")]
    pub file: Option<PathBuf>,

    /// Batch process texts (one per line)
    #[arg(short, long, value_name = "PATH", conflicts_with_all = ["text", "file"])]
    pub batch: Option<PathBuf>,

    /// Batch process from JSON array
    #[arg(long, value_name = "PATH", conflicts_with_all = ["text", "file", "batch"])]
    pub batch_json: Option<PathBuf>,

    /// Output format
    #[arg(short = 'o', long, value_enum, default_value = "probability")]
    pub format: OutputFormat,

    /// Quiet mode (minimal output)
    #[arg(short, long)]
    pub quiet: bool,

    /// Verbose mode (detailed output)
    #[arg(short, long, conflicts_with = "quiet")]
    pub verbose: bool,

    /// Classification threshold
    #[arg(short = 't', long, default_value_t = CLASSIFICATION_THRESHOLD)]
    pub threshold: f32,

    /// Custom class labels (comma-separated: label0,label1)
    #[arg(long, value_delimiter = ',', num_args = 2, default_values = ["human", "ai"])]
    pub labels: Vec<String>,

    /// Disable colored output
    #[arg(long)]
    pub no_color: bool,
}

/// Output format options
#[derive(ValueEnum, Clone, Copy)]
pub enum OutputFormat {
    /// Output just the class label (0 or 1)
    Class,
    /// Output AI probability as a float 0-1 (default)
    Probability,
    /// Output as JSON
    Json,
    /// Human-readable output with confidence
    Human,
}

/// Verbosity level
#[derive(Clone, Copy)]
enum Verbosity {
    Quiet,
    Normal,
    Verbose,
}

/// Input source type
enum InputSource {
    Single(String),
    Batch(Vec<String>),
}

/// Structured prediction result
struct PredictionResult {
    class: Classification,
    class_label: String,
    probabilities: [f32; 2],
    label_names: Vec<String>,
}

/// Main entry point for CLI execution.
///
/// This function orchestrates the entire CLI workflow:
/// 1. Determines input source (arg, file, batch, stdin)
/// 2. Processes input (single or batch)
/// 3. Outputs results in the requested format
pub fn run(cli: &Cli) -> Result<()> {
    // Determine input source
    let input_source = determine_input_source(cli)?;

    // Determine verbosity
    let verbosity = match (cli.quiet, cli.verbose) {
        (true, _) => Verbosity::Quiet,
        (_, true) => Verbosity::Verbose,
        _ => Verbosity::Normal,
    };

    // Process input
    match input_source {
        InputSource::Single(text) => {
            let result = process_single(&text, cli, verbosity)?;
            output_result(&result, cli)?;
        }
        InputSource::Batch(texts) => {
            let results = process_batch(&texts, cli, verbosity)?;
            output_batch_results(&results, cli)?;
        }
    }

    Ok(())
}

/// Determine input source from CLI args.
///
/// Priority: text arg > file > batch > `batch_json` > stdin
fn determine_input_source(cli: &Cli) -> Result<InputSource> {
    if let Some(text) = &cli.text {
        return Ok(InputSource::Single(text.clone()));
    }

    if let Some(path) = &cli.file {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;
        return Ok(InputSource::Single(text));
    }

    if let Some(path) = &cli.batch {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read batch file: {}", path.display()))?;
        let texts: Vec<String> = contents.lines().map(String::from).collect();
        return Ok(InputSource::Batch(texts));
    }

    if let Some(path) = &cli.batch_json {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read JSON batch file: {}", path.display()))?;
        let texts: Vec<String> =
            serde_json::from_str(&contents).with_context(|| "Failed to parse JSON array")?;
        return Ok(InputSource::Batch(texts));
    }

    // Read from stdin
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .context("Failed to read from stdin")?;
    Ok(InputSource::Single(buffer))
}

/// Process a single text and return prediction result.
fn process_single(text: &str, cli: &Cli, verbosity: Verbosity) -> Result<PredictionResult> {
    let start = matches!(verbosity, Verbosity::Verbose).then(Instant::now);

    let predictor = Predictor::new().with_threshold(cli.threshold);
    let prediction = predictor.predict(text)?;
    let class = prediction.classification(cli.threshold);

    if let Some(start_time) = start {
        eprintln!("Inference time: {:?}", start_time.elapsed());
    }

    let class_label = cli
        .labels
        .get(class as usize)
        .cloned()
        .unwrap_or_else(|| class.to_string());

    Ok(PredictionResult {
        class,
        class_label,
        probabilities: [prediction.human_probability(), prediction.ai_probability()],
        label_names: cli.labels.clone(),
    })
}

/// Process multiple texts and return batch results.
fn process_batch(
    texts: &[String],
    cli: &Cli,
    verbosity: Verbosity,
) -> Result<Vec<PredictionResult>> {
    let show_progress = matches!(verbosity, Verbosity::Normal | Verbosity::Verbose)
        && texts.len() > 10
        && !matches!(cli.format, OutputFormat::Json);

    let mut results = Vec::with_capacity(texts.len());

    for (i, text) in texts.iter().enumerate() {
        if show_progress && i % 10 == 0 {
            eprintln!("Processing {}/{}", i + 1, texts.len());
        }
        results.push(process_single(text, cli, verbosity)?);
    }

    if show_progress {
        eprintln!("Completed processing {} texts", texts.len());
    }

    Ok(results)
}

/// Output single result based on format.
fn output_result(result: &PredictionResult, cli: &Cli) -> Result<()> {
    match cli.format {
        OutputFormat::Class => {
            println!("{}", result.class);
        }
        OutputFormat::Probability => {
            // Output just the AI probability (class 1) as a float for pipeline automation
            let ai_prob = result.probabilities.get(1).unwrap_or(&0.0);
            println!("{ai_prob:.4}");
        }
        OutputFormat::Json => {
            let json_output = serde_json::json!({
                "class": result.class,
                "class_label": result.class_label,
                "probabilities": result.label_names.iter()
                    .zip(&result.probabilities)
                    .map(|(label, prob)| (label.clone(), prob))
                    .collect::<HashMap<_, _>>(),
            });
            println!("{}", serde_json::to_string(&json_output)?);
        }
        OutputFormat::Human => {
            let class_idx = i64::from(result.class) as usize;
            let confidence = result.probabilities[class_idx] * 100.0;
            println!("Result: {}", result.class_label);
            println!("Confidence: {confidence:.1}%");
        }
    }
    Ok(())
}

/// Output batch results based on format.
fn output_batch_results(results: &[PredictionResult], cli: &Cli) -> Result<()> {
    match cli.format {
        OutputFormat::Json => {
            // Output as JSON array for batch mode
            let json_array: Vec<_> = results
                .iter()
                .map(|result| {
                    serde_json::json!({
                        "class": result.class,
                        "class_label": result.class_label,
                        "probabilities": result.label_names.iter()
                            .zip(&result.probabilities)
                            .map(|(label, prob)| (label.clone(), prob))
                            .collect::<HashMap<_, _>>(),
                    })
                })
                .collect();
            println!("{}", serde_json::to_string(&json_array)?);
        }
        _ => {
            // For other formats, output each result on its own line
            for result in results {
                output_result(result, cli)?;
            }
        }
    }
    Ok(())
}
