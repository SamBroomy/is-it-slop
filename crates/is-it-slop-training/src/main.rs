//! Train the hierarchical slop classifier with the frozen recipe.
//!
//! The backend is a compile-time feature, following Burn's text-classification example: build
//! with `--features metal` (Apple GPU), `wgpu`, or `cuda`; the default `ndarray` runs on CPU.
//! The metrics dashboard renders in the terminal while training.
//!
//! ```sh
//! cargo run --release -p is-it-slop-training --features metal -- \
//!     --data-dir data/3.0.0 --artifact-dir model_artifacts/burn/v16384-d64
//! ```

use std::path::PathBuf;

use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::InMemDataset;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use is_it_slop_training::data::{
    DocumentEncoder, EncodedDocument, SlopBatcher, SplitData, Vocabulary, read_split,
};
use is_it_slop_training::evaluation::{
    ThresholdMetrics, roc_auc, score, threshold_at_best_f1, threshold_at_precision,
};
use is_it_slop_training::model::HierarchicalModelConfig;
use is_it_slop_training::training::{
    Batch, TrainableClassifier, TrainingConfig, doc_scores, train,
};
use tracing::info;

/// Train the slop classifier with the frozen campaign recipe.
#[derive(Parser, Debug)]
struct Args {
    /// Directory holding the released split parquets (train/validation/test).
    #[arg(long, default_value = "data/3.0.0")]
    data_dir: PathBuf,
    /// Where checkpoints, metric logs, the vocabulary, and the trained model land.
    #[arg(long, default_value = "model_artifacts/burn")]
    artifact_dir: PathBuf,
    /// Vocabulary rows: 16384 for the accuracy/balanced picks, 8192 for minimal.
    #[arg(long, default_value_t = 16384)]
    vocab_size: usize,
    /// Embedding width: 64 for the accuracy pick, 48 for balanced/minimal.
    #[arg(long, default_value_t = 64)]
    embed_dim: usize,
    /// Varies init, shuffling, and dropout draws.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Rows per split, for smoke runs.
    #[arg(long)]
    limit: Option<usize>,
}

mod backend;

fn main() {
    let args = Args::parse();
    launch::<backend::Selected>(&args, backend::NAME, &backend::device());
}

fn load(args: &Args, name: &str) -> SplitData {
    let mut split = read_split(&args.data_dir.join(format!("{name}.parquet")))
        .unwrap_or_else(|e| panic!("reading the {name} split: {e}"));
    if let Some(limit) = args.limit
        && limit < split.len()
    {
        // Stride-sample across the split: the parquets are grouped by source and label, so a
        // prefix would be single-class (Python's --limit drew a seeded random sample).
        let len = split.len();
        split.texts = (0..limit)
            .map(|i| std::mem::take(&mut split.texts[i * len / limit]))
            .collect();
        split.labels = (0..limit).map(|i| split.labels[i * len / limit]).collect();
    }
    split
}

/// Full-split `(labels, AI probabilities)` of the trained model over already-encoded documents,
/// batched to bound memory.
fn predictions<B: burn::tensor::backend::Backend>(
    model: &TrainableClassifier<B>,
    batcher: &SlopBatcher,
    encoded: &[EncodedDocument],
    device: &B::Device,
) -> (Vec<f32>, Vec<f32>) {
    use burn::data::dataloader::batcher::Batcher;

    let (mut labels, mut probs) = (Vec::new(), Vec::new());
    for chunk in encoded.chunks(512) {
        let batch: Batch<B> = batcher.batch(chunk.to_vec(), device);
        let (batch_labels, batch_logits) = doc_scores(&model.forward_step(batch));
        labels.extend(batch_labels);
        probs.extend(batch_logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())));
    }
    (labels, probs)
}

/// Both fitted thresholds scored on one split, shaped like the Python `results.json` blocks.
#[derive(serde::Serialize)]
struct SplitResults {
    #[serde(rename = "precision_0.99")]
    precision_0_99: ThresholdMetrics,
    best_f1: ThresholdMetrics,
}

impl SplitResults {
    fn new(labels: &[f32], probs: &[f32], thr_precision: f32, thr_f1: f32) -> Self {
        Self {
            precision_0_99: score(labels, probs, thr_precision),
            best_f1: score(labels, probs, thr_f1),
        }
    }
}

/// The run summary written next to the model, the useful subset of Python's `results.json`.
#[derive(serde::Serialize)]
struct RunResults<'a> {
    model_config: &'a HierarchicalModelConfig,
    training_config: &'a TrainingConfig,
    parameters: usize,
    train_seconds: f64,
    /// Recovered from the newest checkpoint on disk; absent if none were written.
    epochs_trained: Option<usize>,
    validation: SplitResults,
    test: SplitResults,
}

/// The last epoch that wrote a checkpoint — the number of epochs actually run.
fn epochs_trained(artifact_dir: &std::path::Path) -> Option<usize> {
    let entries = std::fs::read_dir(artifact_dir.join("checkpoint")).ok()?;
    entries
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_prefix("model-")?
                .strip_suffix(".mpk")?
                .parse()
                .ok()
        })
        .max()
}

fn launch<B: AutodiffBackend>(args: &Args, backend: &str, device: &B::Device) {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    info!(backend, ?device, "starting");

    let train_split = load(args, "train");
    let valid_split = load(args, "validation");

    let vocabulary = Vocabulary::build(&train_split.texts, args.vocab_size);
    std::fs::create_dir_all(&args.artifact_dir).expect("artifact dir");
    // Clear the previous run's training outputs (as Burn's guide does) so the summary, metric
    // logs, and checkpoints describe this run only. Nothing resumes from old checkpoints, and the
    // model, vocabulary, and configs are re-saved below.
    std::fs::remove_file(args.artifact_dir.join("experiment.log")).ok();
    for stale in ["train", "valid", "checkpoint"] {
        std::fs::remove_dir_all(args.artifact_dir.join(stale)).ok();
    }
    vocabulary
        .save(&args.artifact_dir.join("token_vocabulary.safetensors"))
        .expect("saving the vocabulary");
    let encoder = DocumentEncoder::new(vocabulary);
    let batcher = SlopBatcher {
        max_chunks: encoder.max_chunks(),
        chunk_size: encoder.chunk_size(),
    };

    info!(rows = train_split.len(), "encoding");
    let encoded_train = encoder.encode_batch(&train_split.texts, &train_split.labels);
    let encoded_valid = encoder.encode_batch(&valid_split.texts, &valid_split.labels);

    let config = TrainingConfig::new().with_seed(args.seed);
    let model_config = HierarchicalModelConfig::new()
        .with_vocab_size(args.vocab_size)
        .with_embed_dim(args.embed_dim);
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(2)
        .build(InMemDataset::new(encoded_train));
    // The valid loader is shuffled because the dashboard AUROC (which drives early stopping) is
    // computed per batch: the parquets are grouped by label, and single-class batches score 0,
    // collapsing the epoch mean. Python didn't need this — it computed AUC over the whole split.
    // The encoded documents stay resident (cloned into the dataset) so the post-training
    // evaluation doesn't re-encode the split.
    let dataloader_valid = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(2)
        .build(InMemDataset::new(encoded_valid.clone()));

    let artifact_dir = args.artifact_dir.to_str().expect("utf-8 artifact dir");
    config
        .save(format!("{artifact_dir}/training_config.json"))
        .expect("saving config");
    model_config
        .save(format!("{artifact_dir}/model_config.json"))
        .expect("saving config");
    std::fs::write(
        args.artifact_dir.join("encoder_config.json"),
        serde_json::to_string_pretty(&encoder.bundle_config()).expect("serializing encoder config"),
    )
    .expect("saving encoder config");

    let started = std::time::Instant::now();
    let result = train::<B>(
        artifact_dir,
        model_config.init::<B>(device),
        &config,
        dataloader_train,
        dataloader_valid,
        device,
        None::<burn::train::renderer::CliMetricsRenderer>,
    );
    let train_seconds = started.elapsed().as_secs_f64();

    result
        .model
        .clone()
        .save_file(
            format!("{artifact_dir}/model"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("saving the trained model");

    // The parity-gate numbers: full-split AUC, the statistic the campaign tables report
    // (combo band: validation 0.9960-0.9961, test 0.9968).
    let inner_device = Default::default();
    let (val_labels, val_probs) =
        predictions(&result.model, &batcher, &encoded_valid, &inner_device);
    let val_auc = roc_auc(&val_labels, &val_probs);
    println!("validation AUC (full split): {val_auc:.6}");
    let test_split = load(args, "test");
    let encoded_test = encoder.encode_batch(&test_split.texts, &test_split.labels);
    let (test_labels, test_probs) =
        predictions(&result.model, &batcher, &encoded_test, &inner_device);
    println!(
        "test AUC (full split): {:.6}",
        roc_auc(&test_labels, &test_probs)
    );

    // Thresholds are fit on validation, never test, and are per-artifact: the bundle ships the
    // best-F1 one as `classification_threshold.txt`, exactly as the Python trainer did.
    let thr_precision = threshold_at_precision(&val_labels, &val_probs, 0.99);
    let thr_f1 = threshold_at_best_f1(&val_labels, &val_probs);
    println!("thresholds: precision>=0.99 at {thr_precision}, best F1 at {thr_f1}");
    std::fs::write(
        args.artifact_dir.join("classification_threshold.txt"),
        thr_f1.to_string(),
    )
    .expect("saving the threshold");

    let results = RunResults {
        model_config: &model_config,
        training_config: &config,
        parameters: result.model.num_params(),
        train_seconds,
        epochs_trained: epochs_trained(&args.artifact_dir),
        validation: SplitResults::new(&val_labels, &val_probs, thr_precision, thr_f1),
        test: SplitResults::new(&test_labels, &test_probs, thr_precision, thr_f1),
    };
    std::fs::write(
        args.artifact_dir.join("results.json"),
        serde_json::to_string_pretty(&results).expect("serializing results"),
    )
    .expect("saving results");
    println!("artifacts in {artifact_dir}");
}
