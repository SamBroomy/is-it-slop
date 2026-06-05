plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    id("maven-publish")
    id("io.github.andrefigas.rustjni") version "0.0.27"
}

val libraryVersion = "1.0.0"

android {
    namespace = "io.github.codewithtamim"
    compileSdk = 35

    defaultConfig {
        minSdk = 24
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    publishing {
        singleVariant("release")
    }
}

rustJni {
    rustPath = "./android/library/rust"
    ndkVersion = "27.0.12077973"
    architectures {
        aarch64_linux_android("aarch64-linux-android24-clang")
        armv7_linux_androideabi("armv7a-linux-androideabi24-clang")
        i686_linux_android("i686-linux-android24-clang")
        x86_64_linux_android("x86_64-linux-android24-clang")
    }
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
