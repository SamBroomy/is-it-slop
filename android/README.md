# is-it-slop for Android

Detect AI-generated text on device. A thin Kotlin library over the same Rust model as the CLI and
Python package: the model is embedded in a native `.so`, no network and no extra runtime.

Everything Android-specific (Gradle, Kotlin, the JNI glue crate) lives in this folder.

## Usage

```kotlin
// suspend functions run on Dispatchers.Default, so they are safe from the main thread
val prediction = IsItSlop.predict("Some text to check")
println("${prediction.classification}: ${prediction.aiProbability}")

// Custom threshold (the default is IsItSlop.defaultThreshold)
prediction.classify(threshold = 0.7f)

// Many texts: all or nothing, or one nullable result per text (null = blank text)
IsItSlop.predictAll(listOf("a", "b"))
IsItSlop.predictEach(listOf("a", ""))
```

Blocking variants (`predictBlocking`, ...) exist for non-coroutine code; never call them on the main
thread. Errors are thrown as `IsItSlopException` (`EmptyInput`, `Native`).

## Layout

| Path | What |
| --- | --- |
| `library/src/main/kotlin` | Public Kotlin API (`IsItSlop.kt`) and the JNI surface (`NativeBridge.kt`) |
| `library/rust` | The JNI glue crate, `is-it-slop-android`, a member of the root Cargo workspace |
| `library/build.gradle.kts` | Builds the Rust crate with cargo-ndk and packages the AAR; the version comes from the workspace `Cargo.toml` |
| `justfile` | `just android::build-aar`, `test`, `publish-local` from the repo root |

The native layer is stateless and returns one row of floats per text. Labels and thresholds are
applied in Kotlin.

## Building

You need JDK 17, the Android SDK with NDK 27 (`ANDROID_NDK_HOME` set), Gradle, and:

```sh
cargo install cargo-ndk
rustup target add aarch64-linux-android x86_64-linux-android
just android::build-aar   # AAR in android/library/build/outputs/aar/
```

Only `arm64-v8a` and `x86_64` are built (the model is embedded in each `.so`). The Rust glue tests
run on the host with `cargo test -p is-it-slop-android`.

Not done yet: a release pipeline that attaches the AAR to a GitHub release and publishes it.
