package com.github.sambroomy

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking

/**
 * End to end through JNI on the host: Gradle builds the Rust crate for the build machine
 * (`cargoBuildHost`) and points `java.library.path` at it, so this exercises the real native
 * code and checks that the JNI function names match [NativeBridge].
 */
class IsItSlopNativeTest {
    private val text = "The quick brown fox jumps over the lazy dog, then writes a short note about it."

    @Test
    fun predictsOneText() {
        val prediction = IsItSlop.predictBlocking(text)

        assertTrue(prediction.aiProbability in 0f..1f)
        assertTrue(prediction.numChunks >= 1)
        assertEquals(1f, prediction.aiProbability + prediction.humanProbability, 1e-6f)
    }

    @Test
    fun defaultClassificationUsesTheDefaultThreshold() {
        val prediction = IsItSlop.predictBlocking(text)

        assertEquals(prediction.classify(IsItSlop.defaultThreshold), prediction.classification)
        assertEquals(Classification.Ai, prediction.classify(0f))
    }

    @Test
    fun eachGivesNullForBlankTexts() {
        val results = IsItSlop.predictEachBlocking(listOf(text, "  ", text))

        assertEquals(3, results.size)
        assertNotNull(results[0])
        assertNull(results[1])
        assertNotNull(results[2])
    }

    @Test
    fun allFailsOnBlankTextInsteadOfCrashing() {
        assertFailsWith<IsItSlopException.EmptyInput> { IsItSlop.predictAllBlocking(listOf(text, "")) }
        assertFailsWith<IsItSlopException.EmptyInput> { IsItSlop.predictAllBlocking(emptyList()) }
        assertFailsWith<IsItSlopException.EmptyInput> { IsItSlop.predictBlocking("") }
    }

    @Test
    fun suspendVariantAgreesWithBlocking() = runBlocking {
        val suspended = IsItSlop.predict(text)

        assertEquals(IsItSlop.predictBlocking(text).aiProbability, suspended.aiProbability, 1e-5f)
    }

    @Test
    fun exposesModelMetadata() {
        assertTrue(IsItSlop.modelVersion.isNotBlank())
        assertTrue(IsItSlop.defaultThreshold in 0f..1f)
    }
}
