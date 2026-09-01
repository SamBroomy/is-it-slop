//! Train the hierarchical slop classifier and inspect its artifacts.
//!
//! The backend is a compile-time feature, following Burn's text-classification example: build
//! with `--features metal` (Apple GPU), `wgpu`, or `cuda`; with no feature the Flex pure-Rust
//! CPU backend runs everywhere. `train` runs the frozen recipe with the terminal metrics
//! dashboard; `evaluate` re-scores a trained bundle's shrunk weight forms afterwards,
//! deliberately outside the training run so a backend limitation there can never cost a
//! finished run its outputs.
//!
//! ```sh
//! cargo run --release -p is-it-slop-training --features metal -- \
//!     train --data-dir data/3.0.0 --artifact-dir model_artifacts/burn/v16384-d64
//! cargo run --release -p is-it-slop-training --features metal -- \
//!     evaluate --data-dir data/3.0.0 --artifact-dir model_artifacts/burn/v16384-d64
//! ```

use std::path::{Path, PathBuf};

use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::InMemDataset;
use burn::module::Module;
use burn::tensor::backend::AutodiffBackend;
use clap::{Parser, Subcommand};
use is_it_slop_training::data::{
    DocumentEncoder, EncodedDocument, SlopBatcher, SplitData, Vocabulary, read_split,
};
use is_it_slop_training::evaluation::{
    ThresholdMetrics, confidence_ranking_auc, fit_temperature, roc_auc, score,
    threshold_at_best_f1, threshold_at_precision,
};
use is_it_slop_training::model::{HierarchicalModelConfig, HierarchicalSlopClassifier};
use is_it_slop_training::training::{Batch, TrainingConfig, train};
use tracing::info;

/// Train the slop classifier and inspect the resulting bundle.
#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Train with the frozen campaign recipe and write the bundle artifacts.
    Train(TrainArgs),
    /// Re-score a trained bundle's shrunk weight forms (f16 roundtrip, Q8) on validation.
    Evaluate(EvaluateArgs),
}

#[derive(clap::Args, Debug)]
struct TrainArgs {
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

#[derive(clap::Args, Debug)]
struct EvaluateArgs {
    /// The trained bundle to evaluate (weights, configs, vocabulary).
    #[arg(long)]
    artifact_dir: PathBuf,
    /// Directory holding the released split parquets.
    #[arg(long, default_value = "data/3.0.0")]
    data_dir: PathBuf,
    /// Rows from the validation split, for smoke runs.
    #[arg(long)]
    limit: Option<usize>,
}

mod backend;

/// The non-autodiff backend evaluation and checking run on: a forward-only loop under
/// `Autodiff` leaks until the OS kills it (EXPERIMENTS.md).
type Inner = <backend::Selected as AutodiffBackend>::InnerBackend;

fn main() {
    // Burn and CubeCL log per-iteration progress and autotune chatter at INFO; keep the console
    // for this crate's own reporting (RUST_LOG still overrides).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "info,burn_train=warn,burn_fusion=warn,cubecl=warn,cubecl_runtime=warn".into()
            }),
        )
        .init();

    match Cli::parse().command {
        Command::Train(args) => {
            launch::<backend::Selected>(&args, backend::NAME, &backend::device());
        }
        Command::Evaluate(args) => evaluate::<Inner>(&args, &backend::device()),
    }
}

fn load(data_dir: &Path, limit: Option<usize>, name: &str) -> SplitData {
    let mut split = read_split(&data_dir.join(format!("{name}.parquet")))
        .unwrap_or_else(|e| panic!("reading the {name} split: {e}"));
    if let Some(limit) = limit
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

/// Full-split model outputs over already-encoded documents, batched to bound memory: everything
/// the metrics, threshold fitting, and uncertainty signals consume, one entry per document.
struct SplitPredictions {
    labels: Vec<f32>,
    /// Raw document logits; `sigmoid(logit / T)` is the calibrated AI probability.
    logits: Vec<f32>,
    /// Population std of the real chunks' AI probabilities — the model's internal disagreement.
    chunk_std: Vec<f32>,
    /// Real chunks per document (dispersion over a single chunk is 0 by construction).
    real_chunks: Vec<f32>,
}

fn predictions<B: burn::tensor::backend::Backend>(
    model: &HierarchicalSlopClassifier<B>,
    batcher: &SlopBatcher,
    encoded: &[EncodedDocument],
    device: &B::Device,
) -> SplitPredictions {
    use burn::data::dataloader::batcher::Batcher;
    use burn::tensor::activation::sigmoid;

    let mut out = SplitPredictions {
        labels: Vec::with_capacity(encoded.len()),
        logits: Vec::with_capacity(encoded.len()),
        chunk_std: Vec::with_capacity(encoded.len()),
        real_chunks: Vec::with_capacity(encoded.len()),
    };
    for docs in encoded.chunks(512) {
        let batch: Batch<B> = batcher.batch(docs.to_vec(), device);
        let output = model.forward(batch.ids);

        let chunk_probs = sigmoid(output.chunk_logits);
        let mask = output.chunk_mask.float();
        let count = mask.clone().sum_dim(1);
        let mean = (chunk_probs.clone() * mask.clone()).sum_dim(1) / count.clone();
        let centered = (chunk_probs - mean) * mask;
        let std = (centered.clone() * centered)
            .sum_dim(1)
            .div(count.clone())
            .sqrt();

        out.labels.extend(docs.iter().map(|d| d.label));
        // convert() first: on an f16/flex32 element backend the data arrives in that type.
        out.logits.extend(
            output
                .doc_logits
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap(),
        );
        out.chunk_std
            .extend(std.into_data().convert::<f32>().to_vec::<f32>().unwrap());
        out.real_chunks
            .extend(count.into_data().convert::<f32>().to_vec::<f32>().unwrap());
    }
    out
}

/// `sigmoid(logit / T)` — the calibrated AI probabilities the bundle's threshold applies to.
fn calibrate(logits: &[f32], temperature: f64) -> Vec<f32> {
    let t = temperature as f32;
    logits
        .iter()
        .map(|z| 1.0 / (1.0 + (-z / t).exp()))
        .collect()
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
/// The `evaluate` subcommand later inserts an `artifact_checks` block.
#[derive(serde::Serialize)]
struct RunResults<'a> {
    model_config: &'a HierarchicalModelConfig,
    training_config: &'a TrainingConfig,
    parameters: usize,
    train_seconds: f64,
    /// Recovered from the newest checkpoint on disk; absent if none were written.
    epochs_trained: Option<usize>,
    /// Which confidence signals rank validation errors — see [`UncertaintySignals`].
    uncertainty: UncertaintySignals,
    validation: SplitResults,
    test: SplitResults,
}

/// Uncertainty candidates, each scored by how well it ranks correct validation predictions
/// above errors ([`confidence_ranking_auc`]; 0.5 = no signal). Everything lives in calibrated
/// probability space: correctness at the shipped best-F1 threshold, margin about 0.5.
#[derive(serde::Serialize)]
struct UncertaintySignals {
    /// Temperature-scaling calibration fit on validation; ships with the bundle.
    temperature: f64,
    /// `|sigmoid(logit / T) - 0.5|` — confidence as calibrated margin, the primary signal.
    calibrated_margin_auc: Option<f64>,
    /// `-std(chunk_probs)` — internal disagreement across chunks, the mixed-content signal.
    /// Restricted to documents with >= 2 real chunks: a single chunk's dispersion is 0 by
    /// construction and would read as false confidence.
    chunk_dispersion_auc: Option<f64>,
    /// Documents the dispersion signal was measured on.
    n_multichunk: usize,
}

fn uncertainty_signals(
    val: &SplitPredictions,
    cal_probs: &[f32],
    threshold: f32,
    temperature: f64,
) -> UncertaintySignals {
    let correct: Vec<f32> = val
        .labels
        .iter()
        .zip(cal_probs)
        .map(|(&y, &p)| f32::from(u8::from((p >= threshold) == (y > 0.5))))
        .collect();
    let margin: Vec<f32> = cal_probs.iter().map(|&p| (p - 0.5).abs()).collect();

    let multi: Vec<usize> = (0..val.labels.len())
        .filter(|&i| val.real_chunks[i] >= 2.0)
        .collect();
    let multi_correct: Vec<f32> = multi.iter().map(|&i| correct[i]).collect();
    let multi_dispersion: Vec<f32> = multi.iter().map(|&i| -val.chunk_std[i]).collect();

    UncertaintySignals {
        temperature,
        calibrated_margin_auc: confidence_ranking_auc(&correct, &margin),
        chunk_dispersion_auc: confidence_ranking_auc(&multi_correct, &multi_dispersion),
        n_multichunk: multi.len(),
    }
}

/// Save the trained classifier in the bundle's weight formats.
///
/// `model.safetensors` is the canonical shipped artifact (inspectable, ecosystem-standard, and
/// the format the vocabulary already uses); `model.f16.safetensors` is the half-precision
/// candidate, ~50% smaller; `model.bpk` is Burn's native store, a local artifact only (zero-copy
/// embedding, and the checkpoint format once the Burn version allows it — until then checkpoints
/// stay `MessagePack` because the `Learner`'s checkpointer takes a `Recorder`, which burn-store
/// does not implement).
fn save_model_artifacts<B: burn::tensor::backend::Backend>(
    model: &HierarchicalSlopClassifier<B>,
    artifact_dir: &Path,
) {
    use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot, SafetensorsStore};

    let version = env!("CARGO_PKG_VERSION");
    let mut bpk = BurnpackStore::from_file(artifact_dir.join("model.bpk"))
        .metadata("preprocessing_version", version)
        .overwrite(true);
    model.save_into(&mut bpk).expect("saving model.bpk");

    let mut full = SafetensorsStore::from_file(artifact_dir.join("model.safetensors"))
        .metadata("preprocessing_version", version)
        .overwrite(true);
    model
        .save_into(&mut full)
        .expect("saving model.safetensors");

    let mut half = SafetensorsStore::from_file(artifact_dir.join("model.f16.safetensors"))
        .metadata("preprocessing_version", version)
        .with_to_adapter(HalfPrecisionAdapter::new())
        .overwrite(true);
    model
        .save_into(&mut half)
        .expect("saving model.f16.safetensors");
}

/// The last epoch that wrote a checkpoint — the number of epochs actually run.
fn epochs_trained(artifact_dir: &Path) -> Option<usize> {
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

fn launch<B: AutodiffBackend>(args: &TrainArgs, backend: &str, device: &B::Device) {
    info!(backend, ?device, "starting");

    let train_split = load(&args.data_dir, args.limit, "train");
    let valid_split = load(&args.data_dir, args.limit, "validation");

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
    // Training splits long documents into several examples so no text is truncated away;
    // evaluation keeps one grid per document so metrics stay document-level.
    let encoded_train = encoder.encode_batch_split(&train_split.texts, &train_split.labels);
    let encoded_valid = encoder.encode_batch(&valid_split.texts, &valid_split.labels);
    info!(
        documents = train_split.len(),
        examples = encoded_train.len(),
        "training examples after long-document splitting"
    );

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

    // Interactive runs keep Burn's TUI dashboard; redirected/nohup'd runs get one quiet
    // progress line per interval instead of the fallback renderer's per-iteration dump.
    let renderer = if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        None
    } else {
        Some(is_it_slop_training::training::ThrottledRenderer::new())
    };

    let started = std::time::Instant::now();
    let result = train::<B>(
        artifact_dir,
        model_config.init::<B>(device),
        &config,
        dataloader_train,
        dataloader_valid,
        device,
        renderer,
    );
    let train_seconds = started.elapsed().as_secs_f64();

    save_model_artifacts(&result.model.model, &args.artifact_dir);

    // The parity-gate numbers: full-split AUC, the statistic the campaign tables report
    // (combo band: validation 0.9960-0.9961, test 0.9968). AUC is invariant under the monotone
    // calibration, so it is computed on the raw logits.
    let inner_device = Default::default();
    let val = predictions(&result.model.model, &batcher, &encoded_valid, &inner_device);
    let val_auc = roc_auc(&val.labels, &val.logits);
    println!("validation AUC (full split): {val_auc:.6}");
    let test_split = load(&args.data_dir, args.limit, "test");
    let encoded_test = encoder.encode_batch(&test_split.texts, &test_split.labels);
    let test = predictions(&result.model.model, &batcher, &encoded_test, &inner_device);
    println!(
        "test AUC (full split): {:.6}",
        roc_auc(&test.labels, &test.logits)
    );

    // The bundle's decision contract lives in calibrated probability space: temperature scaling
    // is fit on validation, thresholds are fit on the calibrated probabilities, and both ship.
    // Same operating points as fitting on raw probabilities (the calibration is monotone), but
    // the threshold value is meaningful next to `sigmoid(logit / T)`. Python fit on raw sigmoid;
    // this supersedes it.
    let temperature = fit_temperature(&val.labels, &val.logits);
    let val_cal = calibrate(&val.logits, temperature);
    let test_cal = calibrate(&test.logits, temperature);
    let thr_precision = threshold_at_precision(&val.labels, &val_cal, 0.99);
    let thr_f1 = threshold_at_best_f1(&val.labels, &val_cal);
    println!(
        "temperature {temperature:.4}; calibrated thresholds: precision>=0.99 at \
         {thr_precision}, best F1 at {thr_f1}"
    );
    std::fs::write(
        args.artifact_dir.join("classification_threshold.txt"),
        thr_f1.to_string(),
    )
    .expect("saving the threshold");
    std::fs::write(
        args.artifact_dir.join("calibration_temperature.txt"),
        temperature.to_string(),
    )
    .expect("saving the temperature");

    let results = RunResults {
        model_config: &model_config,
        training_config: &config,
        parameters: result.model.num_params(),
        train_seconds,
        epochs_trained: epochs_trained(&args.artifact_dir),
        uncertainty: uncertainty_signals(&val, &val_cal, thr_f1, temperature),
        validation: SplitResults::new(&val.labels, &val_cal, thr_precision, thr_f1),
        test: SplitResults::new(&test.labels, &test_cal, thr_precision, thr_f1),
    };
    std::fs::write(
        args.artifact_dir.join("results.json"),
        serde_json::to_string_pretty(&results).expect("serializing results"),
    )
    .expect("saving results");
    println!("artifacts in {artifact_dir}; run the `evaluate` subcommand for the f16/Q8 costs");
}

// --------------------------------------------------------------------------------------------
// The `evaluate` subcommand
// --------------------------------------------------------------------------------------------

/// Every shrunk form of the weights re-scored on validation, so a smaller artifact ships with
/// its measured cost next to the full-precision number. Runs outside the training process on
/// purpose: a backend without quantized ops fails here, not in a finished training run.
#[derive(serde::Serialize)]
struct ArtifactChecks {
    /// The reference: `model.safetensors` loaded back and scored.
    #[serde(rename = "full_validation_auc")]
    full: f64,
    /// AUC after loading the f16 artifact — the cost of halving the artifact, and a roundtrip
    /// of the exact burn-store path inference will use.
    #[serde(rename = "f16_safetensors_validation_auc")]
    f16_safetensors: f64,
    /// AUC after in-memory Q8 min-max quantization of the matrix weights (rank >= 2, as with
    /// weight decay — biases and norms are tiny and precision-sensitive). Measurement only: the
    /// quantized form is not yet emitted as an artifact.
    #[serde(rename = "q8_minmax_validation_auc")]
    q8_minmax: f64,
}

/// [`Quantizer`](burn::module::Quantizer) restricted to matrices, leaving rank-1 parameters
/// (biases, `LayerNorm` scales) in float.
struct MatrixQuantizer(burn::module::Quantizer);

impl<B: burn::tensor::backend::Backend> burn::module::ModuleMapper<B> for MatrixQuantizer {
    fn map_float<const D: usize>(
        &mut self,
        param: burn::module::Param<burn::tensor::Tensor<B, D>>,
    ) -> burn::module::Param<burn::tensor::Tensor<B, D>> {
        if D >= 2 {
            burn::module::ModuleMapper::<B>::map_float(&mut self.0, param)
        } else {
            param
        }
    }
}

fn evaluate<B: burn::tensor::backend::Backend>(args: &EvaluateArgs, device: &B::Device) {
    use burn::module::Quantizer;
    use burn::tensor::quantization::{Calibration, QuantScheme, QuantStore, QuantValue};
    use burn_store::{HalfPrecisionAdapter, ModuleSnapshot, SafetensorsStore};

    let dir = &args.artifact_dir;
    let model_config = HierarchicalModelConfig::load(dir.join("model_config.json"))
        .expect("model_config.json in the artifact dir");
    let vocabulary =
        Vocabulary::load(&dir.join("token_vocabulary.safetensors")).expect("bundle vocabulary");
    let encoder = DocumentEncoder::new(vocabulary);
    let batcher = SlopBatcher {
        max_chunks: encoder.max_chunks(),
        chunk_size: encoder.chunk_size(),
    };
    let valid = load(&args.data_dir, args.limit, "validation");
    let encoded = encoder.encode_batch(&valid.texts, &valid.labels);

    let mut model = model_config.init::<B>(device);
    let mut store = SafetensorsStore::from_file(dir.join("model.safetensors"));
    model.load_from(&mut store).expect("loading the weights");
    let p = predictions(&model, &batcher, &encoded, device);
    let full = roc_auc(&p.labels, &p.logits);

    let mut f16_model = model_config.init::<B>(device);
    let mut store = SafetensorsStore::from_file(dir.join("model.f16.safetensors"))
        .with_from_adapter(HalfPrecisionAdapter::new());
    f16_model
        .load_from(&mut store)
        .expect("loading the f16 weights");
    let p = predictions(&f16_model, &batcher, &encoded, device);
    let f16 = roc_auc(&p.labels, &p.logits);

    // Native store: the packed-u32 default requires last dims divisible by 4, which the
    // [hidden, 1] head weights are not — and its failure poisons the device stream.
    let mut quantizer = MatrixQuantizer(Quantizer {
        calibration: Calibration::MinMax,
        scheme: QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native),
    });
    let q8_model = model.map(&mut quantizer);
    let p = predictions(&q8_model, &batcher, &encoded, device);
    let q8 = roc_auc(&p.labels, &p.logits);

    println!("validation AUC — full: {full:.6}");
    println!("             f16: {f16:.6} (cost {:+.6})", f16 - full);
    println!("             q8:  {q8:.6} (cost {:+.6})", q8 - full);

    // Fold the numbers into the run summary when one exists; stand alone otherwise.
    let checks = ArtifactChecks {
        full,
        f16_safetensors: f16,
        q8_minmax: q8,
    };
    let results_path = dir.join("results.json");
    let mut doc: serde_json::Value = std::fs::read_to_string(&results_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    doc["artifact_checks"] = serde_json::to_value(&checks).expect("serializing checks");
    std::fs::write(
        &results_path,
        serde_json::to_string_pretty(&doc).expect("serializing results"),
    )
    .expect("saving results");
    println!("artifact_checks written to {}", results_path.display());
}
