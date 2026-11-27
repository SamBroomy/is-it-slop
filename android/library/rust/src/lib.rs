//! JNI glue for the Android library.
//!
//! Deliberately dumb: no state, no JSON, no thresholds. Kotlin hands over an array of texts and
//! gets back one flat `float[]` of [`STRIDE`] values per text; labels and thresholds are applied
//! on the Kotlin side. Every entry point catches panics and turns them into a Java exception,
//! because unwinding across the JNI boundary would abort the app.
//!
//! Row layout per text: `[ai_probability, confidence, num_chunks, chunk_dispersion]`.
//! `chunk_dispersion` is `NaN` for single-chunk texts. A blank text yields a row of all `NaN`.

#![expect(
    clippy::missing_const_for_fn,
    reason = "JNI entry points are called by the JVM, never const-evaluated"
)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use is_it_slop::{CLASSIFICATION_THRESHOLD, MODEL_VERSION, Predictor};
use jni::{
    Env, EnvUnowned,
    errors::ThrowRuntimeExAndDefault,
    jni_str,
    objects::{JClass, JObjectArray, JString},
    strings::JNIString,
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
                    f32::from(u16::try_from(p.num_chunks()).expect("chunk count fits u16")),
                    p.chunk_dispersion().unwrap_or(f32::NAN),
                ]
            })
        })
        .collect()
}

/// Run `body`, converting an `Err` or a panic into a thrown `IllegalStateException` and a null
/// return.
fn guarded<R>(
    env: &mut Env<'_>,
    null: R,
    body: impl FnOnce(&mut Env<'_>) -> Result<R, String>,
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
    let _ = env.throw_new(
        jni_str!("java/lang/IllegalStateException"),
        JNIString::new(message),
    );
    null
}

fn read_texts(
    env: &mut Env<'_>,
    texts: &JObjectArray<'_, JString<'_>>,
) -> Result<Vec<String>, String> {
    let len = texts.len(env).map_err(|e| e.to_string())?;
    (0..len)
        .map(|i| {
            let element = texts.get_element(env, i).map_err(|e| e.to_string())?;
            element.try_to_string(env).map_err(|e| e.to_string())
        })
        .collect()
}

/// `NativeBridge.predictEach(texts: Array<String>): FloatArray`
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_sambroomy_isitslop_NativeBridge_predictEach<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    texts: JObjectArray<'local, JString<'local>>,
) -> jfloatArray {
    env.with_env(|env| -> jni::errors::Result<jfloatArray> {
        Ok(guarded(env, std::ptr::null_mut(), |env| {
            let texts = read_texts(env, &texts)?;
            let rows = predict_rows(&texts);
            let array = env.new_float_array(rows.len()).map_err(|e| e.to_string())?;
            array.set_region(env, 0, &rows).map_err(|e| e.to_string())?;
            Ok(array.into_raw())
        }))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `NativeBridge.modelVersion(): String`
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_sambroomy_isitslop_NativeBridge_modelVersion<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
) -> jstring {
    env.with_env(|env| -> jni::errors::Result<jstring> {
        Ok(guarded(env, std::ptr::null_mut(), |env| {
            env.new_string(MODEL_VERSION)
                .map(JString::into_raw)
                .map_err(|e| e.to_string())
        }))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// `NativeBridge.defaultThreshold(): Float`
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_sambroomy_isitslop_NativeBridge_defaultThreshold<'local>(
    _env: EnvUnowned<'local>,
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
