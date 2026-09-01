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
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use is_it_slop_training::data::{
    DocumentEncoder, EncodedDocument, SlopBatcher, SplitData, Vocabulary, read_split,
};
use is_it_slop_training::evaluation::{
    ThresholdMetrics, confidence_ranking_auc, fit_temperature, roc_auc, score,
    threshold_at_best_f1, threshold_at_precision,
};
use is_it_slop_training::model::{HierarchicalModelConfig, HierarchicalSlopClassifier};
use is_it_slop_training::training::{Batch, TrainableClassifier, TrainingConfig, train};
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

/// Full-split model outputs over already-encoded documents, batched to bound memory: everything
/// the metrics, threshold fitting, and uncertainty signals consume, one entry per document.
struct SplitPredictions {
    labels: Vec<f32>,
    /// Raw document logits.
    logits: Vec<f32>,
    /// `sigmoid` of the document logits — the AI probabilities.
    probs: Vec<f32>,
    /// Mean of the real chunks' AI probabilities.
    chunk_mean: Vec<f32>,
    /// Population std of the real chunks' AI probabilities — the model's internal disagreement.
    chunk_std: Vec<f32>,
    /// Real chunks per document (dispersion over a single chunk is 0 by construction).
    real_chunks: Vec<f32>,
}

fn predictions<B: burn::tensor::backend::Backend>(
    model: &TrainableClassifier<B>,
    batcher: &SlopBatcher,
    encoded: &[EncodedDocument],
    device: &B::Device,
) -> SplitPredictions {
    use burn::data::dataloader::batcher::Batcher;
    use burn::tensor::activation::sigmoid;

    let mut out = SplitPredictions {
        labels: Vec::with_capacity(encoded.len()),
        logits: Vec::with_capacity(encoded.len()),
        probs: Vec::with_capacity(encoded.len()),
        chunk_mean: Vec::with_capacity(encoded.len()),
        chunk_std: Vec::with_capacity(encoded.len()),
        real_chunks: Vec::with_capacity(encoded.len()),
    };
    for docs in encoded.chunks(512) {
        let batch: Batch<B> = batcher.batch(docs.to_vec(), device);
        let output = model.model.forward(batch.ids);

        let chunk_probs = sigmoid(output.chunk_logits);
        let mask = output.chunk_mask.float();
        let count = mask.clone().sum_dim(1);
        let mean = (chunk_probs.clone() * mask.clone()).sum_dim(1) / count.clone();
        let centered = (chunk_probs - mean.clone()) * mask;
        let std = (centered.clone() * centered)
            .sum_dim(1)
            .div(count.clone())
            .sqrt();

        out.labels.extend(docs.iter().map(|d| d.label));
        let logits: Vec<f32> = output.doc_logits.into_data().to_vec().unwrap();
        out.probs
            .extend(logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())));
        out.logits.extend(logits);
        out.chunk_mean
            .extend(mean.into_data().to_vec::<f32>().unwrap());
        out.chunk_std
            .extend(std.into_data().to_vec::<f32>().unwrap());
        out.real_chunks
            .extend(count.into_data().to_vec::<f32>().unwrap());
    }
    out
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
    /// Which confidence signals rank validation errors — see [`UncertaintySignals`].
    uncertainty: UncertaintySignals,
    validation: SplitResults,
    test: SplitResults,
    artifact_checks: ArtifactChecks,
}

/// Uncertainty candidates, each scored by how well it ranks correct validation predictions
/// above errors ([`confidence_ranking_auc`]; 0.5 = no signal). Correctness is at the best-F1
/// threshold, while the margin is measured about the calibrated 0.5 point — a deliberate
/// exploratory pairing, since calibrated distance-from-half is the signal inference would ship.
#[derive(serde::Serialize)]
struct UncertaintySignals {
    /// Temperature-scaling calibration fit on validation.
    temperature: f64,
    /// `|sigmoid(logit / T) - 0.5|` — confidence as calibrated margin.
    calibrated_margin_auc: Option<f64>,
    /// `-|doc_prob - mean(chunk_probs)|` — the "does the document head agree with a naive
    /// chunk aggregation" signal.
    doc_chunk_agreement_auc: Option<f64>,
    /// `-std(chunk_probs)` — internal disagreement across chunks, the mixed-content signal.
    /// Restricted to documents with >= 2 real chunks: a single chunk's dispersion is 0 by
    /// construction and would read as false confidence.
    chunk_dispersion_auc: Option<f64>,
    /// Documents the dispersion signal was measured on.
    n_multichunk: usize,
}

fn uncertainty_signals(val: &SplitPredictions, threshold: f32) -> UncertaintySignals {
    let temperature = fit_temperature(&val.labels, &val.logits);
    let correct: Vec<f32> = val
        .labels
        .iter()
        .zip(&val.probs)
        .map(|(&y, &p)| f32::from(u8::from((p >= threshold) == (y > 0.5))))
        .collect();

    let t = temperature as f32;
    let margin: Vec<f32> = val
        .logits
        .iter()
        .map(|&z| (1.0 / (1.0 + (-z / t).exp()) - 0.5).abs())
        .collect();
    let agreement: Vec<f32> = val
        .probs
        .iter()
        .zip(&val.chunk_mean)
        .map(|(&p, &m)| -(p - m).abs())
        .collect();

    let multi: Vec<usize> = (0..val.labels.len())
        .filter(|&i| val.real_chunks[i] >= 2.0)
        .collect();
    let multi_correct: Vec<f32> = multi.iter().map(|&i| correct[i]).collect();
    let multi_dispersion: Vec<f32> = multi.iter().map(|&i| -val.chunk_std[i]).collect();

    UncertaintySignals {
        temperature,
        calibrated_margin_auc: confidence_ranking_auc(&correct, &margin),
        doc_chunk_agreement_auc: confidence_ranking_auc(&correct, &agreement),
        chunk_dispersion_auc: confidence_ranking_auc(&multi_correct, &multi_dispersion),
        n_multichunk: multi.len(),
    }
}

/// Every shrunk form of the weights re-scored on validation, so a smaller artifact ships with
/// its measured cost next to the full-precision number.
#[derive(serde::Serialize)]
struct ArtifactChecks {
    /// AUC after saving to f16 safetensors and loading back — also roundtrips the exact
    /// burn-store path inference will use.
    f16_safetensors_validation_auc: f64,
    /// AUC after in-memory Q8 min-max quantization of the matrix weights (rank >= 2, as with
    /// weight decay — biases and norms are tiny and precision-sensitive). Measurement only: the
    /// quantized form is not yet emitted as an artifact. `None` when the selected backend cannot
    /// run quantized ops — the check is panic-isolated so it can never cost a run its results.
    q8_minmax_validation_auc: Option<f64>,
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

/// Save, reload, and quantize the trained weights, scoring each form on the validation split.
fn check_artifacts<B: burn::tensor::backend::Backend>(
    model: &TrainableClassifier<B>,
    model_config: &HierarchicalModelConfig,
    batcher: &SlopBatcher,
    encoded_valid: &[EncodedDocument],
    artifact_dir: &std::path::Path,
    device: &B::Device,
) -> ArtifactChecks {
    use burn::module::Quantizer;
    use burn::tensor::quantization::{Calibration, QuantScheme, QuantStore, QuantValue};
    use burn_store::{HalfPrecisionAdapter, ModuleSnapshot, SafetensorsStore};

    let mut f16_model = model_config.init::<B>(device);
    let mut store = SafetensorsStore::from_file(artifact_dir.join("model.f16.safetensors"))
        .with_from_adapter(HalfPrecisionAdapter::new());
    f16_model
        .load_from(&mut store)
        .expect("loading the f16 artifact back");
    let f16 = TrainableClassifier {
        model: f16_model,
        chunk_loss_weight: model.chunk_loss_weight,
    };
    let p = predictions(&f16, batcher, encoded_valid, device);
    let f16_auc = roc_auc(&p.labels, &p.probs);

    // Native store: the packed-u32 default requires last dims divisible by 4, which the
    // [hidden, 1] head weights are not.
    let q8_auc = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut quantizer = MatrixQuantizer(Quantizer {
            calibration: Calibration::MinMax,
            scheme: QuantScheme::default()
                .with_value(QuantValue::Q8S)
                .with_store(QuantStore::Native),
        });
        let q8 = TrainableClassifier {
            model: model.model.clone().map(&mut quantizer),
            chunk_loss_weight: model.chunk_loss_weight,
        };
        let p = predictions(&q8, batcher, encoded_valid, device);
        roc_auc(&p.labels, &p.probs)
    }));
    if q8_auc.is_err() {
        tracing::warn!("Q8 quantized evaluation panicked; the backend lacks quantized ops");
    }
    ArtifactChecks {
        f16_safetensors_validation_auc: f16_auc,
        q8_minmax_validation_auc: q8_auc.ok(),
    }
}

/// Save the trained classifier in the bundle's weight formats.
///
/// `model.safetensors` is the canonical shipped artifact (inspectable, ecosystem-standard, and
/// the format the vocabulary already uses); `model.f16.safetensors` is the half-precision
/// candidate, ~50% smaller; `model.bpk` is Burn's native store, kept for zero-copy embedding and
/// as the pre-quantization reference. Checkpoints stay `MessagePack` — the `Learner`'s checkpointer
/// takes a `Recorder`, which burn-store does not implement.
fn save_model_artifacts<B: burn::tensor::backend::Backend>(
    model: &HierarchicalSlopClassifier<B>,
    artifact_dir: &std::path::Path,
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

    save_model_artifacts(&result.model.model, &args.artifact_dir);

    // The parity-gate numbers: full-split AUC, the statistic the campaign tables report
    // (combo band: validation 0.9960-0.9961, test 0.9968).
    let inner_device = Default::default();
    let val = predictions(&result.model, &batcher, &encoded_valid, &inner_device);
    let val_auc = roc_auc(&val.labels, &val.probs);
    println!("validation AUC (full split): {val_auc:.6}");
    let test_split = load(args, "test");
    let encoded_test = encoder.encode_batch(&test_split.texts, &test_split.labels);
    let test = predictions(&result.model, &batcher, &encoded_test, &inner_device);
    println!(
        "test AUC (full split): {:.6}",
        roc_auc(&test.labels, &test.probs)
    );

    // Thresholds are fit on validation, never test, and are per-artifact: the bundle ships the
    // best-F1 one as `classification_threshold.txt`, exactly as the Python trainer did.
    let thr_precision = threshold_at_precision(&val.labels, &val.probs, 0.99);
    let thr_f1 = threshold_at_best_f1(&val.labels, &val.probs);
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
        uncertainty: uncertainty_signals(&val, thr_f1),
        validation: SplitResults::new(&val.labels, &val.probs, thr_precision, thr_f1),
        test: SplitResults::new(&test.labels, &test.probs, thr_precision, thr_f1),
        artifact_checks: check_artifacts(
            &result.model,
            &model_config,
            &batcher,
            &encoded_valid,
            &args.artifact_dir,
            &inner_device,
        ),
    };
    std::fs::write(
        args.artifact_dir.join("results.json"),
        serde_json::to_string_pretty(&results).expect("serializing results"),
    )
    .expect("saving results");
    println!("artifacts in {artifact_dir}");
}
