package com.genned.app.data.detection.classifier

import kotlin.math.exp

/**
 * How the bundled detectors are combined. Photos and video pair the primary model with
 * different Community Forensics weights (see internal-docs/MODEL.md "Phase 3"):
 * - **photos:** the Phase 3 fine-tuned copy (`commfor-224.onnx`, train run 37747738638,
 *   open-weights generators only). It catches far more AI photos but is worse on video;
 * - **video:** the published weights (`commfor-224-video.onnx`), unchanged.
 *
 * Each path has its own constants, fit by tools/ensemble_calibrate.py on the exact files
 * the app bundles:
 *
 *     s = ((primaryGap - MEAN_PRIMARY) / STD_PRIMARY + (cfLogit - MEAN_CF) / STD_CF) / 2
 *     photo:  P(ai) = sigmoid(SLOPE * s + INTERCEPT)
 *     video:  P(ai) = sigmoid(SLOPE * mean(s over frames) + INTERCEPT)
 *
 * The standardization is fit on photo scores (label-free); the video calibration is fit
 * separately because video frames look different to both models. The Python mirror is
 * tools/evaluate.py APP_ENSEMBLE (photos) and APP_VIDEO_ENSEMBLE.
 */
object EnsembleConfig {
    const val DISPLAY_NAME = "${ModelConfig.DISPLAY_NAME} + ${CommunityForensicsConfig.DISPLAY_NAME} (ensemble)"

    /** Photos, with the fine-tuned Community Forensics: Ensemble build run 37787974107. */
    object Photo {
        const val MEAN_PRIMARY = -0.40238
        const val STD_PRIMARY = 7.98345
        const val MEAN_CF = -1.26874
        const val STD_CF = 6.00447
        const val SLOPE = 3.26446
        const val INTERCEPT = 0.02569
    }

    /** Video, with the published Community Forensics: Ensemble build run 36196623887. */
    object Video {
        const val MEAN_PRIMARY = -0.38708
        const val STD_PRIMARY = 8.00114
        const val MEAN_CF = -3.96926
        const val STD_CF = 4.37424
        const val SLOPE = 3.49213
        const val INTERCEPT = -1.85881
    }

    /** The photo ensemble's uncalibrated score. */
    fun combinePhoto(primaryGap: Double, communityForensicsLogit: Double): Double =
        ((primaryGap - Photo.MEAN_PRIMARY) / Photo.STD_PRIMARY +
            (communityForensicsLogit - Photo.MEAN_CF) / Photo.STD_CF) / 2.0

    /** The video ensemble's uncalibrated score for one frame. */
    fun combineVideo(primaryGap: Double, communityForensicsLogit: Double): Double =
        ((primaryGap - Video.MEAN_PRIMARY) / Video.STD_PRIMARY +
            (communityForensicsLogit - Video.MEAN_CF) / Video.STD_CF) / 2.0

    fun photoProbability(combined: Double): Float = sigmoid(Photo.SLOPE * combined + Photo.INTERCEPT)

    fun videoProbability(meanCombined: Double): Float = sigmoid(Video.SLOPE * meanCombined + Video.INTERCEPT)

    private fun sigmoid(x: Double): Float = (1.0 / (1.0 + exp(-x))).toFloat().coerceIn(0f, 1f)
}
