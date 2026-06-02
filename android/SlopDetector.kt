package ai.isitlop

object SlopDetector {
    private var loaded = false

    fun load() {
        if (!loaded) {
            System.loadLibrary("is_it_slop")
            loaded = true
        }
    }

    init { load() }

    external fun predict(text: String): String
    external fun classify(text: String): String
    external fun predictBatch(textsJson: String): String
    external fun classifyBatch(textsJson: String): String
    external fun setThreshold(value: Float)
    external fun getThreshold(): Float
    external fun getVersion(): String
}
