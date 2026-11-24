package com.github.sambroomy

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Detect AI-generated text, entirely on device.
 *
 * ```
 * val prediction = IsItSlop.predict("Some text")      // suspend; safe from the main thread
 * if (prediction.classification == Classification.Ai) { ... }
 * ```
 *
 * Inference is CPU work. The `suspend` functions run on [Dispatchers.Default]; the `*Blocking`
 * variants run on the calling thread, so never call those from the main thread. The first call
 * loads the native library and the model, so it is slower; call [warmUp] early if that matters.
 *
 * All functions are thread safe and may run concurrently.
 */
object IsItSlop {
    /**
     * Probability of AI at or above which a text is classified as [Classification.Ai].
     * Fitted by the training process on calibrated validation probabilities to maximise overall F1;
     * you should almost always use it as is.
     */
    val defaultThreshold: Float by lazy { guarded { NativeBridge.defaultThreshold() } }

    /** Version of the embedded model. */
    val modelVersion: String by lazy { guarded { NativeBridge.modelVersion() } }

    /** Load the native library and model now, so the first real call is fast. */
    suspend fun warmUp() = withContext(Dispatchers.Default) { warmUpBlocking() }

    /** Blocking [warmUp]. */
    fun warmUpBlocking() {
        predictBlocking("warm up")
    }

    /**
     * Predict a single [text].
     *
     * @throws IsItSlopException.EmptyInput if [text] is blank.
     * @throws IsItSlopException.Native if the native layer fails.
     */
    suspend fun predict(text: String): Prediction = withContext(Dispatchers.Default) { predictBlocking(text) }

    /** Blocking [predict]. Do not call from the main thread. */
    fun predictBlocking(text: String): Prediction =
        predictEachBlocking(listOf(text)).single() ?: throw IsItSlopException.EmptyInput()

    /**
     * Predict many texts at once. All or nothing: a blank text fails the whole call.
     * Use [predictEach] to get a result per text instead.
     *
     * @throws IsItSlopException.EmptyInput if [texts] is empty or any text is blank.
     */
    suspend fun predictAll(texts: List<String>): List<Prediction> =
        withContext(Dispatchers.Default) { predictAllBlocking(texts) }

    /** Blocking [predictAll]. Do not call from the main thread. */
    fun predictAllBlocking(texts: List<String>): List<Prediction> {
        if (texts.isEmpty()) throw IsItSlopException.EmptyInput()
        return predictEachBlocking(texts).map { it ?: throw IsItSlopException.EmptyInput() }
    }

    /**
     * Predict many texts at once, one result per text, in order. A blank text gives `null` at
     * its position instead of failing the others.
     */
    suspend fun predictEach(texts: List<String>): List<Prediction?> =
        withContext(Dispatchers.Default) { predictEachBlocking(texts) }

    /** Blocking [predictEach]. Do not call from the main thread. */
    fun predictEachBlocking(texts: List<String>): List<Prediction?> {
        if (texts.isEmpty()) return emptyList()
        val rows = guarded { NativeBridge.predictEach(texts.toTypedArray()) }
        return Prediction.fromRows(rows)
    }

    private inline fun <T> guarded(block: () -> T): T {
        NativeLibrary.ensureLoaded()
        return try {
            block()
        } catch (e: UnsatisfiedLinkError) {
            throw IsItSlopException.Native("native library does not match this version of the library", e)
        } catch (e: IllegalStateException) {
            // Thrown from native code, including caught Rust panics.
            throw IsItSlopException.Native(e.message ?: "native failure", e)
        }
    }
}

/** Whether a text was judged human-written or AI-generated. */
enum class Classification { Human, Ai }

/**
 * Result of analysing one text.
 *
 * @property aiProbability Calibrated probability the text is AI-generated, in `0.0..1.0`.
 * @property confidence Margin from the 50% coin-flip point, in `0.0..1.0`: 0 means the model is
 *   guessing, 1 means it is certain either way.
 * @property numChunks Number of 192-token chunks the text produced (1 for short texts).
 * @property chunkDispersion Standard deviation of per-chunk AI probabilities; high values suggest
 *   mixed human and AI content. `null` for single-chunk texts.
 */
data class Prediction(
    val aiProbability: Float,
    val confidence: Float,
    val numChunks: Int,
    val chunkDispersion: Float?,
) {
    /** Probability the text is human-written. */
    val humanProbability: Float get() = 1f - aiProbability

    /** Classification at [IsItSlop.defaultThreshold]. */
    val classification: Classification get() = classify()

    /**
     * Classify, AI if [aiProbability] is at or above [threshold] (in `0.0..1.0`).
     *
     * Use the default. [IsItSlop.defaultThreshold] was chosen by the training process to maximise
     * overall F1, so a custom threshold is an escape hatch for a deliberate precision/recall trade
     * (a higher one flags fewer texts as AI, a lower one flags more), not a tuning knob.
     */
    fun classify(threshold: Float = IsItSlop.defaultThreshold): Classification {
        require(threshold in 0f..1f) { "threshold must be in 0.0..1.0, got $threshold" }
        return if (aiProbability >= threshold) Classification.Ai else Classification.Human
    }

    internal companion object {
        /** Values per text in the native result; keep in sync with `STRIDE` in the Rust crate. */
        const val STRIDE = 4

        /** Decode the flat native result. A row of `NaN`s is a blank text. */
        fun fromRows(rows: FloatArray): List<Prediction?> {
            require(rows.size % STRIDE == 0) { "native result has unexpected length ${rows.size}" }
            return List(rows.size / STRIDE) { i ->
                val o = i * STRIDE
                if (rows[o].isNaN()) {
                    null
                } else {
                    Prediction(
                        aiProbability = rows[o],
                        confidence = rows[o + 1],
                        numChunks = rows[o + 2].toInt(),
                        chunkDispersion = rows[o + 3].takeUnless { it.isNaN() },
                    )
                }
            }
        }
    }
}

/** Errors from [IsItSlop]. */
sealed class IsItSlopException(message: String, cause: Throwable? = null) : RuntimeException(message, cause) {
    /** The input was empty or blank. */
    class EmptyInput : IsItSlopException("Input text is empty")

    /** The native library failed to load or run. */
    class Native(message: String, cause: Throwable? = null) : IsItSlopException(message, cause)
}

internal object NativeLibrary {
    private val loadError: Throwable? by lazy {
        try {
            System.loadLibrary("is_it_slop_android")
            null
        } catch (e: UnsatisfiedLinkError) {
            e
        }
    }

    fun ensureLoaded() {
        loadError?.let { throw IsItSlopException.Native("could not load libis_it_slop_android.so", it) }
    }
}
