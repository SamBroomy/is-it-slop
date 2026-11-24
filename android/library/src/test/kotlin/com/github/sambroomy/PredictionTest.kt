package com.github.sambroomy

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

/** Pure JVM tests: no native library is loaded because every threshold here is explicit. */
class PredictionTest {
    private fun prediction(ai: Float) = Prediction(ai, confidence = 0f, numChunks = 1, chunkDispersion = null)

    @Test
    fun classifyIsInclusiveAtTheThreshold() {
        assertEquals(Classification.Ai, prediction(0.5f).classify(0.5f))
        assertEquals(Classification.Human, prediction(0.49f).classify(0.5f))
    }

    @Test
    fun invalidThresholdIsRejected() {
        assertFailsWith<IllegalArgumentException> { prediction(0.5f).classify(1.5f) }
        assertFailsWith<IllegalArgumentException> { prediction(0.5f).classify(-0.1f) }
    }

    @Test
    fun nanRowsDecodeToNullAndRowsToPredictions() {
        val nan = Float.NaN
        val rows = floatArrayOf(0.9f, 0.8f, 2f, 0.1f, nan, nan, nan, nan, 0.2f, 0.6f, 1f, nan)

        val decoded = Prediction.fromRows(rows)

        assertEquals(3, decoded.size)
        assertEquals(Prediction(0.9f, 0.8f, 2, 0.1f), decoded[0])
        assertNull(decoded[1])
        assertEquals(Prediction(0.2f, 0.6f, 1, null), decoded[2])
    }

    @Test
    fun malformedNativeResultIsRejected() {
        assertFailsWith<IllegalArgumentException> { Prediction.fromRows(floatArrayOf(0.1f, 0.2f)) }
    }
}
