//! Training pipeline profiling example
//!
//! Profile the complete training pipeline including:
//! - Text cleaning (training mode with dataset artifacts removal)
//! - Tokenization
//! - Chunking
//! - Vocabulary building (CountVectorizer.fit)
//! - TF-IDF computation (TfidfVectorizer.fit)
//!
//! Run with: cargo flamegraph --profile profiling --example `train_profile`

use std::{path::PathBuf, time::Instant};

use is_it_slop_preprocessing::pre_processor::{
    TfidfVectorizer, VectorizerParams, text_cleaner_for_training,
};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_level(true)
                .with_timer(tracing_subscriber::fmt::time::uptime()),
        )
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "train_profile=info,is_it_slop_preprocessing=info".into()),
        )
        .init();

    info!("=== Training Pipeline Profiler ===");

    // Load training data
    info!("Loading training data");
    let train_start = Instant::now();
    let train_texts = load_data();
    info!(
        num_samples = train_texts.len(),
        elapsed_ms = train_start.elapsed().as_millis(),
        "Training data loaded"
    );

    // Run multiple iterations for better profiling data
    let iterations = std::env::var("PROFILE_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    info!(iterations, "Starting profiling iterations");

    for iteration in 1..=iterations {
        info!(iteration, "=== Iteration {} ===", iteration);

        // Step 1: Text cleaning (training mode) - PARALLELIZED
        info!("Step 1: Text cleaning (training mode)");
        let clean_start = Instant::now();
        let cleaner = text_cleaner_for_training();
        let cleaned_texts = cleaner.clean_batch(&train_texts);
        info!(
            elapsed_ms = clean_start.elapsed().as_millis(),
            "Text cleaning complete"
        );

        // Step 2: Configure vectorizer parameters
        let params = VectorizerParams::new(10.0, 0.8, true);
        info!(
            ngram_range = ?params.ngram_range(),
            min_df = params.min_df(),
            max_df = params.max_df(),
            sublinear_tf = params.sublinear_tf(),
            "Vectorizer parameters configured"
        );

        // Step 3: Fit TF-IDF vectorizer (includes tokenization, chunking, vocab building)
        info!("Step 3: Fitting TF-IDF vectorizer (tokenization + chunking + vocab + IDF)");
        let fit_start = Instant::now();
        let vectorizer = TfidfVectorizer::fit(&cleaned_texts, params.clone());
        let fit_elapsed = fit_start.elapsed();
        info!(
            vocab_size = vectorizer.num_features(),
            elapsed_secs = fit_elapsed.as_secs_f64(),
            "Vectorizer fitted"
        );

        // Step 4: Transform training data
        info!("Step 4: Transforming training data");
        let transform_start = Instant::now();
        let matrix = vectorizer.transform(&cleaned_texts);
        let transform_elapsed = transform_start.elapsed();
        let sparsity = 100.0 * (1.0 - matrix.nnz() as f64 / (matrix.rows() * matrix.cols()) as f64);
        info!(
            matrix_shape = ?(matrix.rows(), matrix.cols()),
            nnz = matrix.nnz(),
            sparsity_percent = format!("{:.2}", sparsity),
            elapsed_secs = transform_elapsed.as_secs_f64(),
            "Training data transformed"
        );

        // Step 5: Fit transform
        info!("Step 5: Fit-transforming training data");
        let fit_transform_start = Instant::now();
        let (vectorizer, matrix) = TfidfVectorizer::fit_transform(&cleaned_texts, params);
        let fit_transform_elapsed = fit_transform_start.elapsed();
        let sparsity = 100.0 * (1.0 - matrix.nnz() as f64 / (matrix.rows() * matrix.cols()) as f64);
        info!(
            vocab_size = vectorizer.num_features(),
            matrix_shape = ?(matrix.rows(), matrix.cols()),
            nnz = matrix.nnz(),
            sparsity_percent = format!("{:.2}", sparsity),
            elapsed_secs = fit_transform_elapsed.as_secs_f64(),
            "Fit-transform complete"
        );
    }

    info!("=== Profiling Complete ===");
}

fn load_data() -> Vec<String> {
    let path = PathBuf::from("profile/test.csv");

    if !path.exists() {
        eprintln!("Error: profile/test.csv not found, generating synthetic data");
        panic!("Please provide a profile/test.csv file with training data for profiling")
    }

    info!(path = ?path, "Loading data from file");

    let mut reader = csv::Reader::from_path(&path).expect("Valid CSV file");

    // Get text column index (looks for 'generation', 'text', etc. or uses column 1)
    let headers = reader.headers().expect("CSV headers");
    let text_col_idx = headers
        .iter()
        .position(|name| {
            matches!(
                name.to_lowercase().as_str(),
                "text" | "generation" | "content" | "body"
            )
        })
        .unwrap_or(1); // Default to column 1 (skip ID in column 0)

    info!(
        text_column = headers.get(text_col_idx).unwrap(),
        "Using text column"
    );

    let texts: Vec<String> = reader
        .records()
        .filter_map(Result::ok)
        .map(|r| r.get(text_col_idx).unwrap_or("").to_string())
        .filter(|s| !s.trim().is_empty())
        .collect();

    info!(num_texts = texts.len(), "Loaded texts from CSV");
    texts
}
