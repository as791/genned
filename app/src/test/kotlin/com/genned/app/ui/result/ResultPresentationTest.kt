package com.genned.app.ui.result

import com.genned.domain.model.DetectionSignal
import com.genned.domain.model.SignalAvailability
import com.genned.domain.model.SignalType
import com.google.common.truth.Truth.assertThat
import org.junit.Test

class ResultPresentationTest {

    private fun available(type: SignalType, score: Float) =
        DetectionSignal(type, SignalAvailability.AVAILABLE, score, 1f, "d")

    private fun unavailable(type: SignalType, notable: Boolean = false) =
        DetectionSignal.unavailable(type, "d").copy(notable = notable)

    private fun types(signals: List<DetectionSignal>) = signals.map { it.type }

    @Test
    fun `a normal photo shows the detector first and folds what had nothing to report`() {
        // The order analysis produces: credentials, camera details, AI tool name, detector, watermark.
        val arranged = ResultPresentation.arrange(
            listOf(
                unavailable(SignalType.CONTENT_CREDENTIALS),
                available(SignalType.EXIF_METADATA, 0f),
                available(SignalType.GENERATOR_METADATA, 0f),
                available(SignalType.AI_CLASSIFIER, 0.8f),
                unavailable(SignalType.WATERMARK),
            ),
        )
        assertThat(types(arranged.shown)).containsExactly(
            SignalType.AI_CLASSIFIER, SignalType.EXIF_METADATA, SignalType.GENERATOR_METADATA,
        ).inOrder()
        assertThat(types(arranged.notAvailable)).containsExactly(SignalType.CONTENT_CREDENTIALS, SignalType.WATERMARK)
    }

    @Test
    fun `verified credentials come first, then a found AI tool name, then the detector`() {
        val arranged = ResultPresentation.arrange(
            listOf(
                available(SignalType.CONTENT_CREDENTIALS, 0.98f),
                available(SignalType.EXIF_METADATA, 1f),
                available(SignalType.GENERATOR_METADATA, 1f),
                available(SignalType.AI_CLASSIFIER, 0.4f),
            ),
        )
        assertThat(types(arranged.shown)).containsExactly(
            SignalType.CONTENT_CREDENTIALS, SignalType.GENERATOR_METADATA, SignalType.AI_CLASSIFIER, SignalType.EXIF_METADATA,
        ).inOrder()
        assertThat(arranged.notAvailable).isEmpty()
    }

    @Test
    fun `credentials that were found but not counted stay visible, after the detector`() {
        val arranged = ResultPresentation.arrange(
            listOf(
                unavailable(SignalType.CONTENT_CREDENTIALS, notable = true),
                available(SignalType.AI_CLASSIFIER, 0.3f),
                unavailable(SignalType.WATERMARK),
            ),
        )
        assertThat(types(arranged.shown)).containsExactly(SignalType.AI_CLASSIFIER, SignalType.CONTENT_CREDENTIALS).inOrder()
        assertThat(types(arranged.notAvailable)).containsExactly(SignalType.WATERMARK)
    }

    @Test
    fun `a video keeps the detector visible and folds the checks video does not have`() {
        val arranged = ResultPresentation.arrange(
            listOf(
                available(SignalType.AI_CLASSIFIER, 0.6f),
                unavailable(SignalType.WATERMARK),
                unavailable(SignalType.GENERATOR_METADATA),
                unavailable(SignalType.EXIF_METADATA),
                unavailable(SignalType.CONTENT_CREDENTIALS),
            ),
        )
        assertThat(types(arranged.shown)).containsExactly(SignalType.AI_CLASSIFIER)
        assertThat(arranged.notAvailable).hasSize(4)
    }

    @Test
    fun `the detector is shown even when it could not run`() {
        val failed = DetectionSignal.error(SignalType.AI_CLASSIFIER, "failed")
        val arranged = ResultPresentation.arrange(listOf(failed, unavailable(SignalType.WATERMARK)))
        assertThat(arranged.shown).containsExactly(failed)
    }
}
