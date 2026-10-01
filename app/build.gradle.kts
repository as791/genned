plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.genned.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.genned.app"
        minSdk = 26
        targetSdk = 35
        // CI sets GENNED_VERSION_CODE (1000 + workflow run number) so every CI build can be
        // uploaded to Play as an update; local builds stay at 1.
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

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        // ONNX Runtime ships native .so libraries per ABI; keep all supported ABIs for
        // the MVP rather than splitting APKs, to keep the release process simple.
        jniLibs {
            useLegacyPackaging = false
        }
    }

    testOptions {
        unitTests {
            isIncludeAndroidResources = true
            isReturnDefaultValues = true
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
