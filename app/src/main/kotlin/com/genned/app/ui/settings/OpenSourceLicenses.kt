package com.genned.app.ui.settings

import androidx.annotation.RawRes
import com.genned.app.R

/** One attributed component shown on the open-source licenses screen. */
data class OpenSourceComponent(
    val name: String,
    val license: String,
    val copyright: String,
    val url: String,
    @RawRes val licenseText: Int,
)

/**
 * Everything the app ships that requires attribution: the libraries in the APK and the two
 * bundled detector models (internal-docs/MODEL.md). License texts are verbatim copies in
 * res/raw. When adding a dependency or model, add it here (OpenSourceLicensesTest checks the
 * bundled models are listed).
 */
object OpenSourceLicenses {
    private const val APACHE_2 = "Apache License 2.0"
    private const val MIT = "MIT License"

    val components: List<OpenSourceComponent> = listOf(
        OpenSourceComponent(
            name = "Dafilab/ai-image-detector (bundled AI-image detector model)",
            license = APACHE_2,
            copyright = "Copyright Dafilab",
            url = "https://huggingface.co/Dafilab/ai-image-detector",
            licenseText = R.raw.license_apache_2_0,
        ),
        OpenSourceComponent(
            name = "Community Forensics ViT-S 224 (bundled AI-image detector model, as published and fine-tuned)",
            license = MIT,
            copyright = "Copyright (c) 2025 Jeongsoo Park",
            url = "https://huggingface.co/OwensLab/commfor-model-224",
            licenseText = R.raw.license_mit_community_forensics,
        ),
        OpenSourceComponent(
            name = "ONNX Runtime",
            license = MIT,
            copyright = "Copyright (c) Microsoft Corporation",
            url = "https://github.com/microsoft/onnxruntime",
            licenseText = R.raw.license_mit_onnxruntime,
        ),
        OpenSourceComponent(
            name = "Android Jetpack (AndroidX Core, Activity, Lifecycle, Navigation, Room, " +
                "ExifInterface, SplashScreen)",
            license = APACHE_2,
            copyright = "Copyright The Android Open Source Project",
            url = "https://developer.android.com/jetpack/androidx",
            licenseText = R.raw.license_apache_2_0,
        ),
        OpenSourceComponent(
            name = "Jetpack Compose and Material Icons",
            license = APACHE_2,
            copyright = "Copyright The Android Open Source Project",
            url = "https://developer.android.com/jetpack/compose",
            licenseText = R.raw.license_apache_2_0,
        ),
        OpenSourceComponent(
            name = "Kotlin standard library, kotlinx.coroutines, kotlinx.serialization",
            license = APACHE_2,
            copyright = "Copyright JetBrains s.r.o. and Kotlin Programming Language contributors",
            url = "https://kotlinlang.org",
            licenseText = R.raw.license_apache_2_0,
        ),
    )
}
