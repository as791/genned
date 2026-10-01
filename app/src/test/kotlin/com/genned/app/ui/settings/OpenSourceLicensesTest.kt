package com.genned.app.ui.settings

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class OpenSourceLicensesTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()

    @Test
    fun `every component has its full license text bundled`() {
        for (component in OpenSourceLicenses.components) {
            val text = context.resources.openRawResource(component.licenseText).bufferedReader().use { it.readText() }
            val expected = if (component.license.startsWith("MIT")) "Permission is hereby granted" else "Apache License"
            assertThat(text).contains(expected)
        }
    }

    @Test
    fun `both bundled detector models are attributed`() {
        val names = OpenSourceLicenses.components.joinToString("\n") { it.name }
        assertThat(names).contains("Dafilab/ai-image-detector")
        assertThat(names).contains("Community Forensics")
    }

    @Test
    fun `the ONNX Runtime attribution keeps Microsoft's copyright line`() {
        val ort = OpenSourceLicenses.components.single { it.name == "ONNX Runtime" }
        val text = context.resources.openRawResource(ort.licenseText).bufferedReader().use { it.readText() }
        assertThat(text).contains("Copyright (c) Microsoft Corporation")
    }
}
