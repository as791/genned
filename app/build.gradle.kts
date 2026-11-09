import com.android.build.api.variant.FilterConfiguration

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ksp)
}

// versionCode = base * 10 + ABI offset (universal + 0), so every APK of a build outranks
// every APK of the previous one and Android installs any of them over any other.
val ABI_VERSION_OFFSETS = mapOf("armeabi-v7a" to 1, "arm64-v8a" to 2, "x86" to 3, "x86_64" to 4)

android {
    namespace = "com.genned.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.genned.app"
        minSdk = 26
        targetSdk = 35
        // CI sets GENNED_VERSION_CODE (1000 + workflow run number) so every CI build can be
        // uploaded to Play as an update; local builds stay at 1. Each APK's own versionCode is
        // this x 10 + its ABI offset (androidComponents below).
        versionCode = System.getenv("GENNED_VERSION_CODE")?.toIntOrNull() ?: 1
        // CI sets GENNED_VERSION_NAME from the release tag (v1.2.3 -> 1.2.3).
        versionName = System.getenv("GENNED_VERSION_NAME") ?: "0.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        // Fixed debug key (committed, non-secret) so each CI-built debug APK has the same
        // signature and installs as an update. Release builds must use a private upload key.
        getByName("debug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
        // Private upload key for Play, never committed: CI writes it from the
        // UPLOAD_KEYSTORE_BASE64 secret (see README "Release builds"). Without these
        // variables the release build is simply unsigned (still built and R8-checked in CI).
        val uploadStore = System.getenv("GENNED_UPLOAD_STORE_FILE")
        if (uploadStore != null && file(uploadStore).exists()) {
            create("upload") {
                storeFile = file(uploadStore)
                storePassword = System.getenv("GENNED_UPLOAD_STORE_PASSWORD")
                keyAlias = System.getenv("GENNED_UPLOAD_KEY_ALIAS")
                keyPassword = System.getenv("GENNED_UPLOAD_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.findByName("upload")
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            isMinifyEnabled = false
            applicationIdSuffix = ".debug"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        // ONNX Runtime ships native .so libraries per ABI. The universal APK keeps all of
        // them; the per-ABI APKs below carry one each.
        jniLibs {
            useLegacyPackaging = false
        }
    }

    // One APK per CPU type next to the universal one: most phones need only arm64-v8a, and
    // each per-ABI APK skips the other three ABIs' native libraries. Off when building an App
    // Bundle (AGP can't bundle with ABI splits on; Play splits bundles by itself), so CI
    // builds the bundle in its own Gradle invocation.
    splits {
        abi {
            isEnable = gradle.startParameter.taskNames.none { it.contains("bundle", ignoreCase = true) }
            reset()
            include(*ABI_VERSION_OFFSETS.keys.toTypedArray())
            isUniversalApk = true
        }
    }

    testOptions {
        unitTests {
            isIncludeAndroidResources = true
            isReturnDefaultValues = true
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

androidComponents {
    onVariants { variant ->
        variant.outputs.forEach { output ->
            val abi = output.filters.find { it.filterType == FilterConfiguration.FilterType.ABI }?.identifier
            val base = output.versionCode.orNull ?: 1
            output.versionCode.set(base * 10 + (ABI_VERSION_OFFSETS[abi] ?: 0))
        }
    }
}

dependencies {
    implementation(project(":domain"))

    implementation(libs.core.ktx)
    implementation(libs.core.splashscreen)
    implementation(libs.lifecycle.runtime.ktx)
    implementation(libs.lifecycle.viewmodel.ktx)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.activity.compose)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons.extended)
    debugImplementation(libs.compose.ui.tooling)

    implementation(libs.navigation.compose)

    implementation(libs.room.runtime)
    implementation(libs.room.ktx)
    ksp(libs.room.compiler)

    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)

    implementation(libs.exifinterface)

    // On-device AI classifier runtime. See domain classifier docs at
    // app/src/main/kotlin/com/genned/app/data/detection/classifier and internal-docs/MODEL.md
    // for the bundled model this loads.
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit)
    testImplementation(libs.truth)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)
    testImplementation(libs.androidx.test.ext.junit)
    testImplementation(libs.room.testing)

    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.espresso.core)
}

ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}
