package com.genned.app.data.detection.classifier

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import kotlin.math.exp

class EnsembleConfigTest {

    private fun sigmoid(x: Double) = (1.0 / (1.0 + exp(-x))).toFloat()

    @Test
    fun `both models at their photo means combine to zero`() {
        val photo = EnsembleConfig.combinePhoto(EnsembleConfig.Photo.MEAN_PRIMARY, EnsembleConfig.Photo.MEAN_CF)
        val video = EnsembleConfig.combineVideo(EnsembleConfig.Video.MEAN_PRIMARY, EnsembleConfig.Video.MEAN_CF)
        assertThat(photo).isWithin(1e-9).of(0.0)
        assertThat(video).isWithin(1e-9).of(0.0)
    }

    @Test
    fun `each model contributes its standardized logit with equal weight`() {
        with(EnsembleConfig.Photo) {
            assertThat(EnsembleConfig.combinePhoto(MEAN_PRIMARY + STD_PRIMARY, MEAN_CF)).isWithin(1e-9).of(0.5)
            assertThat(EnsembleConfig.combinePhoto(MEAN_PRIMARY, MEAN_CF + STD_CF)).isWithin(1e-9).of(0.5)
        }
        with(EnsembleConfig.Video) {
            assertThat(EnsembleConfig.combineVideo(MEAN_PRIMARY + STD_PRIMARY, MEAN_CF)).isWithin(1e-9).of(0.5)
            assertThat(EnsembleConfig.combineVideo(MEAN_PRIMARY, MEAN_CF + STD_CF)).isWithin(1e-9).of(0.5)
        }
    }

    @Test
    fun `combined score rises with either model's evidence`() {
        val photo = EnsembleConfig.combinePhoto(0.0, 0.0)
        assertThat(EnsembleConfig.combinePhoto(1.0, 0.0)).isGreaterThan(photo)
        assertThat(EnsembleConfig.combinePhoto(0.0, 1.0)).isGreaterThan(photo)
        val video = EnsembleConfig.combineVideo(0.0, 0.0)
        assertThat(EnsembleConfig.combineVideo(1.0, 0.0)).isGreaterThan(video)
        assertThat(EnsembleConfig.combineVideo(0.0, 1.0)).isGreaterThan(video)
    }

    @Test
    fun `photo and video use their own Community Forensics standardization`() {
        // The fine-tuned photo copy and the published video copy have different logit scales,
        // so the same raw logits must not combine to the same score.
        assertThat(EnsembleConfig.combinePhoto(2.0, 3.0)).isNotEqualTo(EnsembleConfig.combineVideo(2.0, 3.0))
    }

    @Test
    fun `photo and video probabilities are their fitted Platt calibrations`() {
        for (s in listOf(-2.0, -0.5, 0.0, 0.5, 2.0)) {
            assertThat(EnsembleConfig.photoProbability(s))
                .isWithin(1e-6f).of(sigmoid(EnsembleConfig.Photo.SLOPE * s + EnsembleConfig.Photo.INTERCEPT))
            assertThat(EnsembleConfig.videoProbability(s))
                .isWithin(1e-6f).of(sigmoid(EnsembleConfig.Video.SLOPE * s + EnsembleConfig.Video.INTERCEPT))
        }
    }

    @Test
    fun `an average video frame score reads lower than an average photo score`() {
        // Real video frames look more AI-like to both models, so the video calibration is
        // shifted down: at each path's mean evidence (s = 0) a video needs more to read AI.
        assertThat(EnsembleConfig.videoProbability(0.0)).isLessThan(EnsembleConfig.photoProbability(0.0))
    }

    @Test
    fun `probabilities stay in range for extreme inputs`() {
        assertThat(EnsembleConfig.photoProbability(1e6)).isAtMost(1f)
        assertThat(EnsembleConfig.photoProbability(-1e6)).isAtLeast(0f)
        assertThat(EnsembleConfig.videoProbability(1e6)).isAtMost(1f)
        assertThat(EnsembleConfig.videoProbability(-1e6)).isAtLeast(0f)
    }
}
