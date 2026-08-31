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
use is_it_slop_training::data::{DocumentEncoder, SlopBatcher, SplitData, Vocabulary, read_split};
use is_it_slop_training::model::HierarchicalModelConfig;
use is_it_slop_training::training::{Batch, TrainableClassifier, TrainingConfig, roc_auc, train};
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

/// Full-split AUC of the trained model, batched to bound memory.
fn evaluate<B: burn::tensor::backend::Backend>(
    model: &TrainableClassifier<B>,
    batcher: &SlopBatcher,
    encoder: &DocumentEncoder,
    split: &SplitData,
    device: &B::Device,
) -> f64 {
    use burn::data::dataloader::batcher::Batcher;

    let encoded = encoder.encode_batch(&split.texts, &split.labels);
    let (mut scores, mut labels) = (Vec::new(), Vec::new());
    for chunk in encoded.chunks(512) {
        let batch: Batch<B> = batcher.batch(chunk.to_vec(), device);
        let output = model.forward_step(batch);
        scores.extend(output.logits.into_data().to_vec::<f32>().unwrap());
        labels.extend(output.targets.into_data().to_vec::<f32>().unwrap());
    }
    roc_auc(&labels, &scores)
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
    let dataloader_valid = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(2)
        .build(InMemDataset::new(encoded_valid));

    let artifact_dir = args.artifact_dir.to_str().expect("utf-8 artifact dir");
    config
        .save(format!("{artifact_dir}/training_config.json"))
        .expect("saving config");
    model_config
        .save(format!("{artifact_dir}/model_config.json"))
        .expect("saving config");

    let result = train::<B>(
        artifact_dir,
        model_config.init::<B>(device),
        &config,
        dataloader_train,
        dataloader_valid,
        device,
        None::<burn::train::renderer::CliMetricsRenderer>,
    );

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
    let val_auc = evaluate(
        &result.model,
        &batcher,
        &encoder,
        &valid_split,
        &inner_device,
    );
    println!("validation AUC (full split): {val_auc:.6}");
    let test_split = load(args, "test");
    let test_auc = evaluate(
        &result.model,
        &batcher,
        &encoder,
        &test_split,
        &inner_device,
    );
    println!("test AUC (full split): {test_auc:.6}");
    println!("artifacts in {artifact_dir}");
}
