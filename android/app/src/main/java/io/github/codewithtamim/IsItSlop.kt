package io.github.codewithtamim

import org.json.JSONArray
import org.json.JSONObject

object IsItSlop {

    init {
        System.loadLibrary("onnxruntime")
        System.loadLibrary("is_it_slop")
    }

    private external fun nativePredict(text: String): String
    private external fun nativePredictWithThreshold(text: String, threshold: Float): String
    private external fun nativeClassify(text: String): String
    private external fun nativePredictBatch(textsJson: String): String

    data class Result(
        val humanProbability: Float,
        val aiProbability: Float,
        val classification: String,
        val error: String? = null,
    ) {
        val isError: Boolean get() = error != null
        val isAi: Boolean get() = classification == "AI"
        val isHuman: Boolean get() = classification == "Human"
    }

    fun predict(text: String): Result {
        val json = nativePredict(text)
        return parseResult(json)
    }

    fun predictWithThreshold(text: String, threshold: Float): Result {
        val json = nativePredictWithThreshold(text, threshold)
        return parseResult(json)
    }

    fun classify(text: String): String = nativeClassify(text)

    fun predictBatch(texts: List<String>): List<Result> {
        val jsonInput = JSONArray(texts).toString()
        val jsonOutput = nativePredictBatch(jsonInput)
        val arr = JSONArray(jsonOutput)
        return (0 until arr.length()).map { i ->
            parseResult(arr.getJSONObject(i).toString())
        }
    }

    private fun parseResult(json: String): Result {
        val obj = JSONObject(json)
        if (obj.has("error")) {
            return Result(
                humanProbability = 0f,
                aiProbability = 0f,
                classification = "Error",
                error = obj.getString("error"),
            )
        }
        return Result(
            humanProbability = obj.getDouble("human").toFloat(),
            aiProbability = obj.getDouble("ai").toFloat(),
            classification = obj.getString("class"),
        )
    }
}
