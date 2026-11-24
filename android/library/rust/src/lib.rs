//! JNI glue for the Android library.
//!
//! Deliberately dumb: no state, no JSON, no thresholds. Kotlin hands over an array of texts and
//! gets back one flat `float[]` of [`STRIDE`] values per text; labels and thresholds are applied
//! on the Kotlin side. Every entry point catches panics and turns them into a Java exception,
//! because unwinding across the JNI boundary would abort the app.
//!
//! Row layout per text: `[ai_probability, confidence, num_chunks, chunk_dispersion]`.
//! `chunk_dispersion` is `NaN` for single-chunk texts. A blank text yields a row of all `NaN`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use is_it_slop::{CLASSIFICATION_THRESHOLD, MODEL_VERSION, Predictor};
use jni::{
    JNIEnv,
    objects::{JClass, JObjectArray, JString},
    sys::{jfloat, jfloatArray, jstring},
};

/// Floats per text in the array returned by `predictEach`.
pub const STRIDE: usize = 4;

/// Predict every text (concurrently, one result per text) into the flat row layout above.
#[must_use]
pub fn predict_rows<T: AsRef<str>>(texts: &[T]) -> Vec<f32> {
    Predictor::new()
        .predict_each(texts)
        .into_iter()
        .flat_map(|result| {
            result.map_or([f32::NAN; STRIDE], |p| {
                [
                    p.ai_probability(),
                    p.confidence(),
                    p.num_chunks() as f32,
                    p.chunk_dispersion().unwrap_or(f32::NAN),
                ]
            })
        })
        .collect()
}

/// Run `body`, converting an `Err` or a panic into a thrown `IllegalStateException` and a null
/// return.
fn guarded<R>(
    env: &mut JNIEnv<'_>,
    null: R,
    body: impl FnOnce(&mut JNIEnv<'_>) -> Result<R, String>,
) -> R {
    let outcome = catch_unwind(AssertUnwindSafe(|| body(env)));
    let message = match outcome {
        Ok(Ok(value)) => return value,
        Ok(Err(message)) => message,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .map_or_else(
                || "native panic".to_owned(),
                |m| format!("native panic: {m}"),
            ),
    };
    // If throwing itself fails there is nothing more we can do.
    let _ = env.throw_new("java/lang/IllegalStateException", message);
    null
}

fn read_texts(env: &mut JNIEnv<'_>, texts: &JObjectArray<'_>) -> Result<Vec<String>, String> {
    let len = env.get_array_length(texts).map_err(|e| e.to_string())?;
    (0..len)
        .map(|i| {
            let element = env
                .get_object_array_element(texts, i)
                .map_err(|e| e.to_string())?;
            env.get_string(&JString::from(element))
                .map(String::from)
                .map_err(|e| e.to_string())
        })
        .collect()
}

/// `NativeBridge.predictEach(texts: Array<String>): FloatArray`
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_sambroomy_isitslop_NativeBridge_predictEach<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    texts: JObjectArray<'local>,
) -> jfloatArray {
    guarded(&mut env, std::ptr::null_mut(), |env| {
        let texts = read_texts(env, &texts)?;
        let rows = predict_rows(&texts);
        let array = env
            .new_float_array(i32::try_from(rows.len()).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        env.set_float_array_region(&array, 0, &rows)
            .map_err(|e| e.to_string())?;
        Ok(array.into_raw())
    })
}

/// `NativeBridge.modelVersion(): String`
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_sambroomy_isitslop_NativeBridge_modelVersion<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
) -> jstring {
    guarded(&mut env, std::ptr::null_mut(), |env| {
        env.new_string(MODEL_VERSION)
            .map(JString::into_raw)
            .map_err(|e| e.to_string())
    })
}

/// `NativeBridge.defaultThreshold(): Float`
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_sambroomy_isitslop_NativeBridge_defaultThreshold<'local>(
    _env: JNIEnv<'local>,
    _class: JClass<'local>,
) -> jfloat {
    CLASSIFICATION_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_one_per_text_with_nan_for_blank() {
        let rows = predict_rows(&["Some reasonable text to look at.", "", "Another sentence."]);

        assert_eq!(rows.len(), 3 * STRIDE);
        let (first, rest) = rows.split_at(STRIDE);
        let (blank, last) = rest.split_at(STRIDE);

        assert!((0.0..=1.0).contains(&first[0]));
        assert!(first[2] >= 1.0, "at least one chunk");
        assert!(blank.iter().all(|v| v.is_nan()));
        assert!((0.0..=1.0).contains(&last[0]));
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert_eq!(predict_rows::<&str>(&[]).len(), 0);
    }
}
