use std::{collections::HashMap, path::PathBuf, time::Instant};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use slop_inference::CLASSIFICATION_THRESHOLD;

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
    #[arg(short = 't', long, default_value_t = CLASSIFICATION_THRESHOLD)]
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
    class: slop_inference::Classification,
    class_label: String,
    probabilities: [f32; 2],
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
    use std::io::Read;

    use anyhow::Context;

    // Priority: text arg > file > batch > batch_json > stdin
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

/// Process a single text
fn process_single(text: &str, cli: &Cli, verbosity: Verbosity) -> Result<PredictionResult> {
    let start = matches!(verbosity, Verbosity::Verbose).then(Instant::now);

    // Use new Predictor API
    let predictor = slop_inference::Predictor::new().with_threshold(cli.threshold);
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

/// Process multiple texts
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
            let class_idx = i64::from(result.class) as usize;
            let confidence = result.probabilities[class_idx] * 100.0;
            println!("Result: {}", result.class_label);
            println!("Confidence: {confidence:.1}%");
        }
    }
    Ok(())
}

/// Output batch results
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
