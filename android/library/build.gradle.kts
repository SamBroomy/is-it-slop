plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    id("maven-publish")
}

// One version for everything: the Rust workspace version in the repo root Cargo.toml.
val repoRoot = rootDir.parentFile
val libraryVersion: String = Regex("""\[workspace\.package][^\[]*?\nversion\s*=\s*"([^"]+)"""")
    .find(repoRoot.resolve("Cargo.toml").readText())
    ?.groupValues?.get(1)
    ?: error("could not find [workspace.package] version in Cargo.toml")

// Only 64-bit ABIs: ~all current devices, and the model is embedded in every .so.
val abis = listOf("arm64-v8a", "x86_64")
val rustJniLibs = layout.buildDirectory.dir("rustJniLibs")

android {
    namespace = "com.github.sambroomy"
    compileSdk = 35

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    sourceSets["main"].jniLibs.srcDir(rustJniLibs)

    publishing {
        singleVariant("release")
    }
}

// Build the Rust JNI crate for each ABI with cargo-ndk (`cargo install cargo-ndk`;
// needs ANDROID_NDK_HOME and `rustup target add aarch64-linux-android x86_64-linux-android`).
val cargoNdk by tasks.registering(Exec::class) {
    workingDir = repoRoot
    val args = abis.flatMap { listOf("-t", it) } +
        listOf("-o", rustJniLibs.get().asFile.path, "build", "--release", "-p", "is-it-slop-android")
    commandLine(listOf("cargo", "ndk") + args)
    inputs.files(fileTree(repoRoot.resolve("crates")) { include("**/*.rs", "**/Cargo.toml") })
    inputs.files(fileTree(projectDir.resolve("rust")) { include("**/*.rs", "Cargo.toml") })
    outputs.dir(rustJniLibs)
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("JniLibFolders") }
    .configureEach { dependsOn(cargoNdk) }

dependencies {
    implementation(libs.kotlinx.coroutines.android)
    testImplementation(libs.kotlin.test)
}

afterEvaluate {
    publishing {
        publications {
            register<MavenPublication>("release") {
                groupId = "com.github.sambroomy"
                artifactId = "is-it-slop"
                version = libraryVersion
                from(components["release"])
            }
        }
    }
}
