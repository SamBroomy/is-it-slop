use std::{collections::HashMap, path::PathBuf, time::Instant};

use anyhow::Result;
use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(name = "slop-cli")]
#[command(about = "Detect AI-generated text", long_about = None)]
struct Cli {
    /// Text to analyze (if not provided, reads from stdin)
    #[arg(value_name = "TEXT")]
    text: Option<String>,

    /// Read text from file
    #[arg(short, long, value_name = "PATH", conflicts_with = "text")]
    file: Option<PathBuf>,

    /// Batch process texts (one per line)
    #[arg(short, long, value_name = "PATH", conflicts_with_all = ["text", "file"])]
    batch: Option<PathBuf>,

    /// Batch process from JSON array
    #[arg(long, value_name = "PATH", conflicts_with_all = ["text", "file", "batch"])]
    batch_json: Option<PathBuf>,

    /// Output format
    #[arg(short = 'o', long, value_enum, default_value = "probability")]
    format: OutputFormat,

    /// Quiet mode (minimal output)
    #[arg(short, long)]
    quiet: bool,

    /// Verbose mode (detailed output)
    #[arg(short, long, conflicts_with = "quiet")]
    verbose: bool,

    /// Classification threshold
    #[arg(short = 't', long, default_value = "0.5")]
    threshold: f32,

    /// Custom class labels (comma-separated: label0,label1)
    #[arg(long, value_delimiter = ',', num_args = 2, default_values = ["human", "ai"])]
    labels: Vec<String>,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,
}

#[derive(ValueEnum, Clone, Copy)]
enum OutputFormat {
    /// Output just the class label (0 or 1)
    Class,
    /// Output AI probability as a float 0-1 (default)
    Probability,
    /// Output as JSON
    Json,
    /// Human-readable output with confidence
    Human,
}

#[derive(Clone, Copy)]
enum Verbosity {
    Quiet,
    Normal,
    Verbose,
}

enum InputSource {
    Single(String),
    Batch(Vec<String>),
}

/// Structured prediction result
struct PredictionResult {
    class: i64,
    class_label: String,
    probabilities: Vec<f32>,
    label_names: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Determine input source
    let input_source = determine_input_source(&cli)?;

    // Determine verbosity
    let verbosity = match (cli.quiet, cli.verbose) {
        (true, _) => Verbosity::Quiet,
        (_, true) => Verbosity::Verbose,
        _ => Verbosity::Normal,
    };

    // Process input
    match input_source {
        InputSource::Single(text) => {
            let result = process_single(&text, &cli, verbosity)?;
            output_result(&result, &cli)?;
        }
        InputSource::Batch(texts) => {
            let results = process_batch(&texts, &cli, verbosity)?;
            output_batch_results(&results, &cli)?;
        }
    }

    Ok(())
}

/// Determine input source from CLI args
fn determine_input_source(cli: &Cli) -> Result<InputSource> {
    cli.text.as_ref().map_or_else(
        || {
            cli.file.as_ref().map_or_else(
                || {
                    cli.batch.as_ref().map_or_else(
                        || {
                            cli.batch_json.as_ref().map_or_else(
                                || {
                                    // Read from stdin
                                    todo!("Implement stdin reading");
                                },
                                |path| {
                                    todo!(
                                        "Implement JSON batch reading: parse JSON array from {:?}",
                                        path
                                    );
                                },
                            )
                        },
                        |path| {
                            todo!("Implement batch file reading: read lines from {:?}", path);
                        },
                    )
                },
                |path| {
                    todo!("Implement file reading: read text from {:?}", path);
                },
            )
        },
        |text| Ok(InputSource::Single(text.clone())),
    )
}

/// Process a single text
fn process_single(text: &str, cli: &Cli, verbosity: Verbosity) -> Result<PredictionResult> {
    let start = matches!(verbosity, Verbosity::Verbose).then(Instant::now);

    // Call inference pipeline
    let (labels, probs) = slop_inference::predict(text)?;

    if let Some(start_time) = start {
        eprintln!("Inference time: {:?}", start_time.elapsed());
    }

    // Convert to structured result
    let class = labels[0];
    let class_label = cli
        .labels
        .get(class as usize)
        .cloned()
        .unwrap_or_else(|| class.to_string());

    let prediction_result = PredictionResult {
        class,
        class_label,
        probabilities: probs[0].clone(),
        label_names: cli.labels.clone(),
    };

    Ok(prediction_result)
}

/// Process multiple texts
fn process_batch(
    _texts: &[String],
    _cli: &Cli,
    _verbosity: Verbosity,
) -> Result<Vec<PredictionResult>> {
    todo!("Implement batch processing - can call process_single in a loop for now");
}

/// Output single result based on format
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
            let confidence = result.probabilities[result.class as usize] * 100.0;
            println!("Result: {}", result.class_label);
            println!("Confidence: {confidence:.1}%");
        }
    }
    Ok(())
}

/// Output batch results
fn output_batch_results(results: &[PredictionResult], cli: &Cli) -> Result<()> {
    for result in results {
        output_result(result, cli)?;
    }
    Ok(())
}
