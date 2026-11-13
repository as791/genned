package com.genned.app.data.detection.classifier

import kotlin.math.exp

/**
 * How the bundled detectors are combined. Photos and video each use their own pair of
 * model files (see internal-docs/MODEL.md "Accuracy round 4"):
 * - **photos:** both models retrained (`ai-image-detector.onnx`, train run 43242952132;
 *   `commfor-224.onnx`, train run 43083106402; open-weights generators only). They read
 *   phone photos and newer generators better but are worse on video;
 * - **video:** both models with their published weights (`ai-image-detector-video.onnx`,
 *   `commfor-224-video.onnx`), unchanged.
 *
 * Each path has its own constants, fit by tools/ensemble_calibrate.py on the exact files
 * the app bundles:
 *
 *     s = ((primaryGap - MEAN_PRIMARY) / STD_PRIMARY + (cfLogit - MEAN_CF) / STD_CF) / 2
 *     photo:  P(ai) = sigmoid(SLOPE * s + INTERCEPT)
 *     video:  P(ai) = sigmoid(SLOPE * mean(s over frames) + INTERCEPT)
 *
 * Each standardization is fit on photo scores of that path's own files (label-free); the
 * video calibration is fit separately because video frames look different to both models.
 * The Python mirror is
 * tools/evaluate.py APP_ENSEMBLE (photos) and APP_VIDEO_ENSEMBLE.
 */
object EnsembleConfig {
    const val DISPLAY_NAME = "${ModelConfig.DISPLAY_NAME} + ${CommunityForensicsConfig.DISPLAY_NAME} (ensemble)"

    /** Photos, with both retrained models: Ensemble build run 43339026079. */
    object Photo {
        const val MEAN_PRIMARY = -0.09054
        const val STD_PRIMARY = 3.41626
        const val MEAN_CF = -1.48263
        const val STD_CF = 6.01668
        const val SLOPE = 3.94365
        const val INTERCEPT = 0.27254
    }

    /** Video, with both published models: Ensemble build run 36196623887. */
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
