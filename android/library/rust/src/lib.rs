use jni::objects::{JClass, JString};
use jni::sys::{jfloat, jstring};
use jni::JNIEnv;
use is_it_slop::{Predictor, Threshold};

fn new_predictor() -> Predictor {
    Predictor::new()
}

fn new_predictor_with_threshold(threshold: f32) -> Predictor {
    let t = Threshold::try_new(threshold).unwrap_or_default();
    Predictor::new().with_threshold(t)
}

fn run_predict(predictor: &Predictor, text: &str) -> String {
    match predictor.predict(text) {
        Ok(unified) => {
            let class = unified.classification(predictor.threshold());
            serde_json::json!({
                "human": unified.prediction.human_probability(),
                "ai": unified.prediction.ai_probability(),
                "class": class.to_string(),
            })
            .to_string()
        }
        Err(e) => serde_json::json!({"error": e.to_string()}).to_string(),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_codewithtamim_IsItSlop_nativePredict(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    text: JString<'_>,
) -> jstring {
    let text: String = match env.get_string(&text) {
        Ok(s) => s.into(),
        Err(_) => {
            return env
                .new_string(r#"{"error":"invalid string argument"}"#)
                .expect("Failed to create Java string")
                .into_raw();
        }
    };

    let predictor = new_predictor();
    let json = run_predict(&predictor, &text);

    env.new_string(json)
        .expect("Failed to create Java string")
        .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_codewithtamim_IsItSlop_nativePredictWithThreshold(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    text: JString<'_>,
    threshold: jfloat,
) -> jstring {
    let text: String = match env.get_string(&text) {
        Ok(s) => s.into(),
        Err(_) => {
            return env
                .new_string(r#"{"error":"invalid string argument"}"#)
                .expect("Failed to create Java string")
                .into_raw();
        }
    };

    let predictor = new_predictor_with_threshold(threshold);
    let json = run_predict(&predictor, &text);

    env.new_string(json)
        .expect("Failed to create Java string")
        .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_codewithtamim_IsItSlop_nativeClassify(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    text: JString<'_>,
) -> jstring {
    let text: String = match env.get_string(&text) {
        Ok(s) => s.into(),
        Err(_) => {
            return env
                .new_string("error")
                .expect("Failed to create Java string")
                .into_raw();
        }
    };

    let predictor = new_predictor();
    let label = match predictor.classify(&text) {
        Ok(class) => class.to_string(),
        Err(_) => "error".to_string(),
    };

    env.new_string(label)
        .expect("Failed to create Java string")
        .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_github_codewithtamim_IsItSlop_nativePredictBatch(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    texts_json: JString<'_>,
) -> jstring {
    let json_input: String = match env.get_string(&texts_json) {
        Ok(s) => s.into(),
        Err(_) => {
            return env
                .new_string(r#"{"error":"invalid string argument"}"#)
                .expect("Failed to create Java string")
                .into_raw();
        }
    };

    let texts: Vec<String> = match serde_json::from_str(&json_input) {
        Ok(t) => t,
        Err(e) => {
            let err =
                serde_json::json!({"error": format!("Invalid JSON array: {e}")}).to_string();
            return env
                .new_string(err)
                .expect("Failed to create Java string")
                .into_raw();
        }
    };

    if texts.is_empty() {
        let err = serde_json::json!({"error": "empty input array"}).to_string();
        return env
            .new_string(err)
            .expect("Failed to create Java string")
            .into_raw();
    }

    let predictor = new_predictor();
    let results: Vec<String> = texts.iter().map(|t| run_predict(&predictor, t)).collect();

    let json = format!("[{}]", results.join(","));

    env.new_string(json)
        .expect("Failed to create Java string")
        .into_raw()
}
