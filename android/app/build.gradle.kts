plugins {
    alias(libs.plugins.android.library)
    id("maven-publish")
}

val libraryVersion = "1.0.0"

android {
    namespace = "io.github.codewithtamim"
    compileSdk = 36

    defaultConfig {
        minSdk = 24
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    publishing {
        singleVariant("release")
    }
}

val rustJniLibsDir = file("${project.projectDir}/src/main/jniLibs")
val workspaceRoot = file("${project.projectDir}/../..")
val cargoHome = System.getenv("CARGO_HOME") ?: "${System.getProperty("user.home")}/.cargo"
val cargoPath = "$cargoHome/bin/cargo"
val ndkHome = System.getenv("ANDROID_NDK_HOME")

tasks.register<Exec>("buildRustLibrary") {
    description = "Build is-it-slop Rust library for Android using cargo-ndk"
    group = "rust"

    inputs.dir("$workspaceRoot/crates/is-it-slop/src")
    inputs.file("$workspaceRoot/crates/is-it-slop/Cargo.toml")
    inputs.file("$workspaceRoot/Cargo.toml")
    inputs.file("$workspaceRoot/Cargo.lock")
    outputs.dir(rustJniLibsDir)

    workingDir = workspaceRoot
    commandLine(
        cargoPath, "ndk",
        "-t", "arm64-v8a",
        "-t", "armeabi-v7a",
        "-t", "x86_64",
        "-t", "x86",
        "-o", rustJniLibsDir,
        "build", "--release",
        "-p", "is-it-slop",
        "--no-default-features",
        "--features", "android"
    )

    environment("PATH", System.getenv("PATH") ?: "/usr/bin:/bin")
    environment("CARGO_HOME", cargoHome)
    environment("ANDROID_NDK_HOME", ndkHome)
}

dependencies {
    implementation(libs.onnxruntime.android)
    implementation(libs.androidx.core.ktx)
}

afterEvaluate {
    publishing {
        publications {
            register<MavenPublication>("release") {
                groupId = "io.github.codewithtamim"
                artifactId = "is-it-slop"
                version = libraryVersion
                from(components["release"])
            }
        }
    }
}
