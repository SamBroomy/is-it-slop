package io.github.sambroomy.isitslop

/**
 * The JNI surface. `external` functions are implemented in Rust (`android/library/rust`), found by
 * name: `Java_io_github_sambroomy_isitslop_NativeBridge_<function>`. Renaming anything here means renaming
 * the Rust function too. Not public API.
 *
 * `@JvmStatic` makes these true static methods, which is what the Rust side expects (a `JClass`
 * argument rather than an instance).
 */
internal object NativeBridge {
    /** One row of `Prediction.STRIDE` floats per text; see the Rust crate for the layout. */
    @JvmStatic external fun predictEach(texts: Array<String>): FloatArray

    @JvmStatic external fun modelVersion(): String

    @JvmStatic external fun defaultThreshold(): Float
}
