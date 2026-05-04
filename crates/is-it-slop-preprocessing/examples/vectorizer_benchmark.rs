//! Vectorizer benchmark example
use std::{path::PathBuf, time::Instant};

use is_it_slop_preprocessing::pre_processor::{TfidfVectorizer, VectorizerParams};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    // Initialize tracing with timing information
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_level(true)
                .with_timer(tracing_subscriber::fmt::time::uptime()),
        )
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "vectorizer_benchmark=info,is_it_slop_preprocessing=debug".into()
            }),
        )
        .init();

    info!("Starting vectorizer benchmark");

    // Load training data
    info!("Loading training data from ./train_texts.csv");
    let train_start = Instant::now();
    let train_texts = csv::Reader::from_path(PathBuf::from(
        "crates/is-it-slop-preprocessing/examples/train_texts.csv",
    ))
    .expect("Valid file")
    .records()
    .filter_map(Result::ok)
    .map(|r| r.get(0).unwrap_or("").to_string())
    .collect::<Vec<String>>();
    info!(
        num_samples = train_texts.len(),
        elapsed_ms = train_start.elapsed().as_millis(),
        "Training data loaded"
    );

    // Load test data
    info!("Loading test data from ./test_texts.csv");
    let test_start = Instant::now();
    let test_texts = csv::Reader::from_path(PathBuf::from(
        "crates/is-it-slop-preprocessing/examples/test_texts.csv",
    ))
    .expect("Valid file")
    .records()
    .filter_map(Result::ok)
    .map(|r| r.get(0).unwrap_or("").to_string())
    .collect::<Vec<String>>();
    info!(
        num_samples = test_texts.len(),
        elapsed_ms = test_start.elapsed().as_millis(),
        "Test data loaded"
    );

    // Configure vectorizer parameters (matching your training pipeline)
    let params = VectorizerParams::new(50.0, 0.8, true);
    info!(
        ngram_range = ?params.ngram_range(),
        min_df = params.min_df(),
        max_df = params.max_df(),
        sublinear_tf = params.sublinear_tf(),
        "Vectorizer parameters configured"
    );

    // Fit vectorizer on training data
    info!("Fitting TF-IDF vectorizer on training data");
    let fit_start = Instant::now();
    let vectorizer = TfidfVectorizer::fit(&train_texts, params);
    let fit_elapsed = fit_start.elapsed();
    info!(
        vocab_size = vectorizer.num_features(),
        elapsed_secs = fit_elapsed.as_secs_f64(),
        "Vectorizer fitted"
    );

    // Transform training data
    info!("Transforming training data");
    let train_transform_start = Instant::now();
    let train_matrix = vectorizer.transform(&train_texts);
    let train_transform_elapsed = train_transform_start.elapsed();
    let train_sparsity = 100.0
        * (1.0 - train_matrix.nnz() as f64 / (train_matrix.rows() * train_matrix.cols()) as f64);
    info!(
        matrix_shape = ?(train_matrix.rows(), train_matrix.cols()),
        nnz = train_matrix.nnz(),
        sparsity_percent = format!("{:.2}", train_sparsity),
        elapsed_secs = train_transform_elapsed.as_secs_f64(),
        "Training data transformed"
    );

    // Transform test data
    info!("Transforming test data");
    let test_transform_start = Instant::now();
    let test_matrix = vectorizer.transform(&test_texts);
    let test_transform_elapsed = test_transform_start.elapsed();
    let test_sparsity =
        100.0 * (1.0 - test_matrix.nnz() as f64 / (test_matrix.rows() * test_matrix.cols()) as f64);
    info!(
        matrix_shape = ?(test_matrix.rows(), test_matrix.cols()),
        nnz = test_matrix.nnz(),
        sparsity_percent = format!("{:.2}", test_sparsity),
        elapsed_secs = test_transform_elapsed.as_secs_f64(),
        "Test data transformed"
    );

    // Summary
    println!("\n=== BENCHMARK SUMMARY ===");
    println!("Training samples: {}", train_texts.len());
    println!("Test samples: {}", test_texts.len());
    println!("Vocabulary size: {}", vectorizer.num_features());
    println!("\nTiming:");
    println!("  Fit:                {:.2}s", fit_elapsed.as_secs_f64());
    println!(
        "  Transform (train):  {:.2}s ({:.0} samples/sec)",
        train_transform_elapsed.as_secs_f64(),
        train_texts.len() as f64 / train_transform_elapsed.as_secs_f64()
    );
    println!(
        "  Transform (test):   {:.2}s ({:.0} samples/sec)",
        test_transform_elapsed.as_secs_f64(),
        test_texts.len() as f64 / test_transform_elapsed.as_secs_f64()
    );
    println!("\nMatrix Statistics:");
    println!(
        "  Train: {}x{} ({:.2}% sparse)",
        train_matrix.rows(),
        train_matrix.cols(),
        train_sparsity
    );
    println!(
        "  Test:  {}x{} ({:.2}% sparse)",
        test_matrix.rows(),
        test_matrix.cols(),
        test_sparsity
    );
}
