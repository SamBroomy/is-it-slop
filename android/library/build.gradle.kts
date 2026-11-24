import com.vanniktech.maven.publish.AndroidSingleVariantLibrary

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.maven.publish)
}

// One version for everything: the Rust workspace version in the repo root Cargo.toml.
val repoRoot = rootDir.parentFile
val libraryVersion: String = Regex("""\[workspace\.package][^\[]*?\nversion\s*=\s*"([^"]+)"""")
    .find(repoRoot.resolve("Cargo.toml").readText())
    ?.groupValues?.get(1)
    ?: error("could not find [workspace.package] version in Cargo.toml")

// All four Android ABIs. Apps ship only the ones they target (ABI splits / app bundles), so the
// big AAR does not reach end users; 32-bit ones are for old phones and old emulators.
val abis = listOf("arm64-v8a", "armeabi-v7a", "x86_64", "x86")
val rustJniLibs = layout.buildDirectory.dir("rustJniLibs")

android {
    namespace = "io.github.sambroomy.isitslop"
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

    packaging {
        // cargo-ndk also copies the core crate's own cdylib (used by the Python bindings); not needed here.
        jniLibs.excludes += "**/libis_it_slop.so"
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

// JVM tests load the real Rust library built for this machine (see IsItSlopNativeTest).
val cargoBuildHost by tasks.registering(Exec::class) {
    workingDir = repoRoot
    commandLine("cargo", "build", "-p", "is-it-slop-android")
}

tasks.withType<Test>().configureEach {
    dependsOn(cargoBuildHost)
    systemProperty("java.library.path", repoRoot.resolve("target/debug").path)
}

dependencies {
    implementation(libs.kotlinx.coroutines.android)
    testImplementation(libs.kotlin.test)
}

// Publishing to Maven Central (Central Portal). Signing and credentials come from CI secrets
// (ORG_GRADLE_PROJECT_mavenCentralUsername / mavenCentralPassword / signingInMemoryKey /
// signingInMemoryKeyPassword); without them only `publishToMavenLocal` works.
mavenPublishing {
    configure(AndroidSingleVariantLibrary(variant = "release", sourcesJar = true, publishJavadocJar = false))
    coordinates("io.github.sambroomy", "is-it-slop", libraryVersion)
    if (providers.gradleProperty("signingInMemoryKey").isPresent) {
        publishToMavenCentral()
        signAllPublications()
    }
    pom {
        name.set("is-it-slop")
        description.set("Detect AI-generated text on device. Small Rust model, embedded, no network.")
        url.set("https://github.com/SamBroomy/is-it-slop")
        licenses {
            license {
                name.set("MIT License")
                url.set("https://opensource.org/licenses/MIT")
            }
        }
        developers {
            developer {
                id.set("SamBroomy")
                name.set("SamBroomy")
                url.set("https://github.com/SamBroomy")
            }
        }
        scm {
            url.set("https://github.com/SamBroomy/is-it-slop")
            connection.set("scm:git:https://github.com/SamBroomy/is-it-slop.git")
            developerConnection.set("scm:git:ssh://git@github.com/SamBroomy/is-it-slop.git")
        }
    }
}
