package com.genned.app.text

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.genned.app.R
import com.genned.domain.evidence.EvidenceTexts
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class TextsTest {

    private val texts = ResourceTexts(ApplicationProvider.getApplicationContext<Context>().resources)

    @Test
    fun `resource limitation texts match the domain's English defaults`() {
        val fromResources = ResourceEvidenceTexts(texts)
        val english = EvidenceTexts.English
        assertThat(fromResources.probabilistic).isEqualTo(english.probabilistic)
        assertThat(fromResources.classifierDidNotRun).isEqualTo(english.classifierDidNotRun)
        assertThat(fromResources.noVerifiedCredentials).isEqualTo(english.noVerifiedCredentials)
        assertThat(fromResources.noGeneratorMetadata).isEqualTo(english.noGeneratorMetadata)
    }

    @Test
    fun `format arguments and plurals are filled in`() {
        assertThat(texts.get(R.string.signal_summary_ai_probability, 73)).contains("73%")
        assertThat(texts.quantity(R.plurals.video_classifier_estimate, 1, 40, 1)).contains("1 sampled frame")
        assertThat(texts.quantity(R.plurals.video_classifier_estimate, 1, 40, 1)).doesNotContain("frames")
        assertThat(texts.quantity(R.plurals.video_classifier_estimate, 6, 40, 6)).contains("6 sampled frames")
    }
}
