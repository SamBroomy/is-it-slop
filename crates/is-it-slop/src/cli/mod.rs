//! CLI module for the is-it-slop text classification tool.
//!
//! Detects AI-generated text from positional arguments, files, or stdin.
//!
//! # Examples
//!
//! ```bash
//! is-it-slop "some text"
//! is-it-slop --label "some text"
//! is-it-slop --score "some text"      # bare float for scripting
//! is-it-slop --label --score "text"   # "Human 0.2340"
//! is-it-slop --json "some text"       # full JSON
//! is-it-slop --jsonl "some text"      # JSON line (for streaming)
//! echo "text" | is-it-slop
//! is-it-slop -f document.txt
//! is-it-slop -b texts.txt             # one per line
//! is-it-slop -b texts.json            # JSON array (auto-detected)
//! is-it-slop --jsonl -b texts.txt     # streaming JSONL batch
//! ```

use std::{io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;

use crate::{Predictor, Threshold, UnifiedPrediction};

fn parse_threshold(s: &str) -> std::result::Result<Threshold, String> {
    s.try_into()
}

/// Command-line arguments for the is-it-slop text classifier
#[derive(Parser)]
#[command(name = "is-it-slop")]
#[command(version, about = "Detect AI-generated text", long_about = None)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// Text to analyze (reads from stdin if not provided, or use "-")
    #[arg(value_name = "TEXT")]
    pub text: Vec<String>,

    /// Read text from file
    #[arg(short, long, value_name = "PATH", conflicts_with_all = ["text", "batch"])]
    pub file: Option<PathBuf>,

    /// Batch process texts from file (one per line, or .json for JSON array)
    #[arg(short, long, value_name = "PATH", conflicts_with_all = ["text", "file"])]
    pub batch: Option<PathBuf>,

    /// Output as JSON
    #[arg(long, conflicts_with_all = ["label", "score", "jsonl"])]
    pub json: bool,

    /// Output as JSON lines (one JSON object per line)
    #[arg(long, conflicts_with_all = ["json", "label", "score"])]
    pub jsonl: bool,

    /// Output only the classification label (Human or AI)
    #[arg(long, conflicts_with = "json")]
    pub label: bool,

    /// Output only the AI probability score (0.0-1.0)
    #[arg(long, conflicts_with = "json")]
    pub score: bool,

    /// Classification threshold [default: model default]
    #[arg(
        short = 't',
        long,
        default_value_t = Threshold::classification_threshold(),
        value_parser = parse_threshold
    )]
    pub threshold: Threshold,
}

fn read_inputs(cli: &Cli) -> Result<Vec<String>> {
    if !cli.text.is_empty() {
        return Ok(cli.text.clone());
    }

    if let Some(path) = &cli.file {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;
        return Ok(vec![text]);
    }

    if let Some(path) = &cli.batch {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read batch file: {}", path.display()))?;

        let is_json = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));

        if is_json {
            let texts: Vec<String> =
                serde_json::from_str(&contents).context("Failed to parse JSON array")?;
            return Ok(texts);
        }

        let texts: Vec<String> = contents
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        return Ok(texts);
    }

    // Read from stdin
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .context("Failed to read from stdin")?;
    Ok(vec![buffer])
}

/// Run the CLI with the given arguments.
///
/// Prints results to stdout. Errors are returned to the caller for display.
pub fn run(cli: &Cli) -> Result<()> {
    let texts = read_inputs(cli)?;

    if texts.is_empty() {
        anyhow::bail!(
            "No input text provided. Use positional arguments, --file, --batch, or pipe text to stdin."
        );
    }

    let predictor = Predictor::default().with_threshold(cli.threshold);

    if texts.len() == 1 {
        let prediction = predictor.predict(&texts[0])?;
        output_single(&prediction, cli, 0)?;
        return Ok(());
    }

    eprintln!("Processing {} texts...", texts.len());
    let predictions = predictor.predict_batch(&texts)?;
    eprintln!("Done.");

    if cli.json {
        output_batch_json(&predictions, cli.threshold)?;
    } else if cli.jsonl {
        output_batch_jsonl(&predictions, cli.threshold)?;
    } else {
        for (i, pred) in predictions.iter().enumerate() {
            if cli.label || cli.score {
                output_single(pred, cli, i)?;
            } else {
                println!("--- [{}/{}] ---", i + 1, predictions.len());
                output_single(pred, cli, i)?;
                println!();
            }
        }
    }

    Ok(())
}

fn prediction_json(prediction: &UnifiedPrediction, threshold: Threshold) -> serde_json::Value {
    let class = prediction.classification(threshold);
    serde_json::json!({
        "status": "ok",
        "class": class,
        "class_label": class.to_string(),
        "probabilities": {
            "human": prediction.prediction.human_probability(),
            "ai": prediction.prediction.ai_probability(),
        },
        "confidence": prediction.confidence_metrics(threshold),
        "chunk_info": prediction.chunk_info(),
        "chunk_predictions": prediction.chunk_predictions.iter().map(|p| {
            (p.human_probability(), p.ai_probability())
        }).collect::<Vec<_>>(),
    })
}

fn output_single(prediction: &UnifiedPrediction, cli: &Cli, index: usize) -> Result<()> {
    if cli.json || cli.jsonl {
        println!(
            "{}",
            serde_json::to_string(&prediction_json(prediction, cli.threshold))?
        );
        return Ok(());
    }

    let prefix = if index > 0 || cli.batch.is_some() {
        format!("[{}] ", index + 1)
    } else {
        String::new()
    };

    if cli.label && cli.score {
        println!(
            "{}{} ({:.4})",
            prefix,
            prediction.classification(cli.threshold),
            prediction.prediction.ai_probability()
        );
    } else if cli.label {
        println!("{}{}", prefix, prediction.classification(cli.threshold));
    } else if cli.score {
        println!("{}{:.4}", prefix, prediction.prediction.ai_probability());
    } else {
        // Human-readable (default)
        let ai_prob = prediction.prediction.ai_probability();
        let human_prob = prediction.prediction.human_probability();
        let metrics = prediction.confidence_metrics(cli.threshold);
        let class = prediction.classification(cli.threshold);

        println!("Classification: {class}");
        println!("Probabilities:");
        println!("  Human: {:.1}%", human_prob * 100.0);
        println!("  AI:    {:.1}%", ai_prob * 100.0);
        println!();
        println!("Confidence Metrics:");
        println!("  Model:     {:.1}%", metrics.model_confidence * 100.0);
        println!("  Threshold: {:.1}%", metrics.threshold_distance * 100.0);
        println!("  Entropy:   {:.1}%", metrics.entropy_confidence * 100.0);
        println!("  Overall:   {:.1}%", metrics.overall * 100.0);

        if prediction.chunk_predictions.len() > 1 {
            let chunk_info = prediction.chunk_info();
            println!();
            println!("Chunk Analysis:");
            println!("  Chunks:    {}", chunk_info.num_chunks);
            println!("  Agreement: {:.1}%", chunk_info.chunk_agreement * 100.0);
            if chunk_info.chunk_agreement < 0.7 {
                println!("  ⚠️  Chunks disagree - mixed content detected");
            }
        }

        if metrics.overall < 0.6 {
            println!();
            println!("⚠️  Low confidence - prediction uncertain");
        }
    }
    Ok(())
}

fn output_batch_json(predictions: &[UnifiedPrediction], threshold: Threshold) -> Result<()> {
    let json_array: Vec<_> = predictions
        .iter()
        .map(|pred| prediction_json(pred, threshold))
        .collect();
    println!("{}", serde_json::to_string(&json_array)?);
    Ok(())
}

fn output_batch_jsonl(predictions: &[UnifiedPrediction], threshold: Threshold) -> Result<()> {
    for pred in predictions {
        println!(
            "{}",
            serde_json::to_string(&prediction_json(pred, threshold))?
        );
    }
    Ok(())
}
