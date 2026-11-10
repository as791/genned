package com.genned.app.data.detection.provenance

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.genned.app.text.ResourceTexts
import com.genned.domain.evidence.EvidenceEngine
import com.genned.domain.model.Classification
import com.genned.domain.model.SignalAvailability
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/** The read-result -> signal mapping (no native code: the reader is not called here). */
@RunWith(RobolectricTestRunner::class)
class C2PAProviderTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val provider = C2PAProvider(context, ResourceTexts(context.resources), reader = { _, _ -> error("not used") })

    private fun report(name: String) = C2paReadResult.Report(
        checkNotNull(javaClass.getResource("/c2pa/$name.json")).readText(),
    )

    @Test
    fun `only trusted manifests that say how the content was made are available`() {
        val ai = provider.toSignal(report("ai_trusted"))
        assertThat(ai.availability).isEqualTo(SignalAvailability.AVAILABLE)
        assertThat(ai.score).isEqualTo(C2PAProvider.AI_SCORE)
        assertThat(ai.description).contains("Genned Test")
        assertThat(ai.evidence).contains("TestGen")

        val capture = provider.toSignal(report("capture_trusted"))
        assertThat(capture.availability).isEqualTo(SignalAvailability.AVAILABLE)
        assertThat(capture.score).isEqualTo(C2PAProvider.CAPTURE_SCORE)

        for (name in listOf("ai_untrusted", "capture_untrusted", "composite_untrusted", "tampered_trusted", "tampered_untrusted")) {
            val signal = provider.toSignal(report(name))
            assertThat(signal.availability).isEqualTo(SignalAvailability.UNAVAILABLE)
            assertThat(signal.score).isNull()
        }
    }

    @Test
    fun `an untrusted AI claim is mentioned but not counted`() {
        val signal = provider.toSignal(report("ai_untrusted"))
        assertThat(signal.description).contains("not on the trusted list")
        assertThat(signal.description).contains("generative AI")
    }

    @Test
    fun `a trusted AI manifest decides the result, a forged one cannot`() {
        val engine = EvidenceEngine()
        assertThat(engine.aggregate(listOf(provider.toSignal(report("ai_trusted")))).classification)
            .isEqualTo(Classification.HIGH)
        val forged = engine.aggregate(listOf(provider.toSignal(report("tampered_trusted"))))
        assertThat(forged.classification).isNotEqualTo(Classification.HIGH)
    }

    @Test
    fun `credentials that exist but are not counted are flagged to stay visible`() {
        for (name in listOf("ai_untrusted", "capture_untrusted", "tampered_trusted")) {
            assertThat(provider.toSignal(report(name)).notable).isTrue()
        }
        assertThat(provider.toSignal(C2paReadResult.RemoteManifest).notable).isTrue()
        // Nothing there, or nothing we could read: folded away with the other empty checks.
        assertThat(provider.toSignal(C2paReadResult.NoManifest).notable).isFalse()
        assertThat(provider.toSignal(C2paReadResult.UnsupportedFormat).notable).isFalse()
        assertThat(provider.toSignal(report("ai_trusted")).notable).isFalse()
    }

    @Test
    fun `nothing found is never evidence`() {
        for (result in listOf(
            C2paReadResult.NoManifest, C2paReadResult.RemoteManifest, C2paReadResult.UnsupportedFormat,
            C2paReadResult.UnsupportedDevice, C2paReadResult.Failed("x"), C2paReadResult.Report("{}"),
        )) {
            val signal = provider.toSignal(result)
            assertThat(signal.availability).isEqualTo(SignalAvailability.UNAVAILABLE)
            assertThat(signal.description).isNotEmpty()
        }
    }
}
