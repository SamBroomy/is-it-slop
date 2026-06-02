//! JNI bindings for Android integration.
//!
//! Maps to Kotlin class `ai.isitlop.SlopDetector`.
//! Build with `--features jni` to produce a `cdylib` for `aarch64-linux-android`.
//!
//! # Kotlin API
//!
//! ```kotlin
//! import ai.isitlop.SlopDetector
//!
//! val result = SlopDetector.predict("Some text")
//! // {"aiProbability":0.92,"humanProbability":0.08,"classification":"AI","numChunks":1,"chunkAgreement":1.0}
//!
//! val label = SlopDetector.classify("Some text")
//! // "Human" or "AI"
//!
//! val results = SlopDetector.predictBatch("""["text 1","text 2"]""")
//! // [{"aiProbability":...}, ...]
//!
//! SlopDetector.setThreshold(0.7f)
//! val t = SlopDetector.getThreshold()
//! val v = SlopDetector.getVersion()
//! ```

use std::sync::{Mutex, OnceLock};

use jni::{
    JNIEnv,
    objects::{JClass, JString},
    sys::{jfloat, jstring},
};

use crate::{MODEL_VERSION, Predictor, Threshold, pipeline::PipelineError};

static PREDICTOR: OnceLock<Mutex<Predictor>> = OnceLock::new();

fn predictor_mutex() -> &'static Mutex<Predictor> {
    PREDICTOR.get_or_init(|| Mutex::new(Predictor::new()))
}

fn with_predictor<T>(f: impl FnOnce(&Predictor) -> Result<T, PipelineError>) -> Result<T, String> {
    let guard = predictor_mutex()
        .lock()
        .map_err(|e| format!("Lock error: {e}"))?;
    f(&guard).map_err(|e| e.to_string())
}

fn return_string(env: &mut JNIEnv<'_>, output: &str) -> jstring {
    env.new_string(output)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

fn error_json(msg: &str) -> String {
    serde_json::json!({"error": msg}).to_string()
}

fn read_input<'a>(env: &mut JNIEnv<'a>, text: &JString<'a>) -> Result<String, String> {
    env.get_string(text)
        .map(|s| s.into())
        .map_err(|e| format!("Failed to read input: {e}"))
}

fn prediction_to_json(pred: &crate::UnifiedPrediction, class: &str) -> serde_json::Value {
    let info = pred.chunk_info();
    serde_json::json!({
        "aiProbability": pred.prediction.ai_probability(),
        "humanProbability": pred.prediction.human_probability(),
        "classification": class,
        "numChunks": info.num_chunks,
        "chunkAgreement": info.chunk_agreement,
    })
}

/// Predict probabilities for a single text.
/// Returns JSON with `aiProbability`, `humanProbability`, `classification`,
/// `numChunks`, and `chunkAgreement` fields.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_ai_isitlop_SlopDetector_predict<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'_>,
    text: JString<'local>,
) -> jstring {
    let result = (|| -> Result<String, String> {
        let text = read_input(&mut env, &text)?;
        with_predictor(|p| {
            let pred = p.predict(&text)?;
            let class = pred.classification(p.threshold()).to_string();
            let json = prediction_to_json(&pred, &class);
            Ok(json.to_string())
        })
    })();

    match result {
        Ok(json) => return_string(&mut env, &json),
        Err(err) => return_string(&mut env, &error_json(&err)),
    }
}

/// Classify a single text as `"Human"` or `"AI"`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_ai_isitlop_SlopDetector_classify<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'_>,
    text: JString<'local>,
) -> jstring {
    let result = (|| -> Result<String, String> {
        let text = read_input(&mut env, &text)?;
        with_predictor(|p| p.classify(&text).map(|c| c.to_string()))
    })();

    match result {
        Ok(class) => return_string(&mut env, &class),
        Err(err) => return_string(&mut env, &error_json(&err)),
    }
}

/// Batch prediction. Accepts a JSON array of strings and returns a JSON array of
/// prediction objects.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_ai_isitlop_SlopDetector_predictBatch<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'_>,
    texts_json: JString<'local>,
) -> jstring {
    let result = (|| -> Result<String, String> {
        let json = read_input(&mut env, &texts_json)?;
        let texts: Vec<String> =
            serde_json::from_str(&json).map_err(|e| format!("Invalid JSON array: {e}"))?;

        let results = with_predictor(|p| {
            let preds = p.predict_batch(&texts)?;
            let results: Vec<serde_json::Value> = preds
                .iter()
                .map(|pred| {
                    let class = pred.classification(p.threshold()).to_string();
                    prediction_to_json(pred, &class)
                })
                .collect();
            Ok(results)
        })?;

        serde_json::to_string(&results).map_err(|e| format!("Serialization error: {e}"))
    })();

    match result {
        Ok(json) => return_string(&mut env, &json),
        Err(err) => return_string(&mut env, &error_json(&err)),
    }
}

/// Batch classification. Accepts a JSON array of strings and returns a JSON array
/// of `"Human"` / `"AI"` labels.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_ai_isitlop_SlopDetector_classifyBatch<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'_>,
    texts_json: JString<'local>,
) -> jstring {
    let result = (|| -> Result<String, String> {
        let json = read_input(&mut env, &texts_json)?;
        let texts: Vec<String> =
            serde_json::from_str(&json).map_err(|e| format!("Invalid JSON array: {e}"))?;

        let labels: Vec<String> = with_predictor(|p| {
            let classes = p.classify_batch(&texts)?;
            Ok(classes.iter().map(|c| c.to_string()).collect())
        })?;

        serde_json::to_string(&labels).map_err(|e| format!("Serialization error: {e}"))
    })();

    match result {
        Ok(json) => return_string(&mut env, &json),
        Err(err) => return_string(&mut env, &error_json(&err)),
    }
}

/// Set the classification threshold. Can be called at any time.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_ai_isitlop_SlopDetector_setThreshold(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    value: jfloat,
) {
    if let Ok(threshold) = Threshold::try_new(value) {
        if let Ok(mut guard) = predictor_mutex().lock() {
            *guard = Predictor::new().with_threshold(threshold);
        }
    }
}

/// Get the current classification threshold.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_ai_isitlop_SlopDetector_getThreshold(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jfloat {
    predictor_mutex()
        .lock()
        .map(|guard| *guard.threshold())
        .unwrap_or(0.5)
}

/// Get the model version string (e.g. `"3.0.0"`).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_ai_isitlop_SlopDetector_getVersion(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jstring {
    return_string(&mut env, MODEL_VERSION)
}
