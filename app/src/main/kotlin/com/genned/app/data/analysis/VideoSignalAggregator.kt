package com.genned.app.data.analysis

import com.genned.app.R
import com.genned.app.data.detection.classifier.ModelConfig
import com.genned.app.text.Texts
import com.genned.app.text.displayPercent
import com.genned.domain.model.DetectionSignal
import com.genned.domain.model.SignalAvailability
import com.genned.domain.model.SignalType

/**
 * Combines one [DetectionSignal] per sampled video frame (all [SignalType.AI_CLASSIFIER])
 * into a single classifier signal for the video as a whole. Pure logic apart from its
 * [Texts], so it's unit-testable without a real video file.
 *
 * Frames are combined in logit space (each frame's image-calibrated score is mapped
 * back to its logit gap, and the gaps are averaged), then the *video* calibration is
 * applied. The image calibration alone overstated video: real clips read AI-like.
 * See [ModelConfig.VIDEO_CALIBRATION_SLOPE].
 */
object VideoSignalAggregator {
    /**
     * @param videoProbability Maps the mean per-frame raw evidence to a video-calibrated
     *   probability. It comes from the classifier that produced the frames
     *   (AIImageClassifierProvider.videoProbability), since single-model and ensemble use
     *   different calibrations. Frames without a rawScore fall back to inverting the
     *   single-model photo calibration.
     */
    fun aggregateFrameSignals(
        perFrameSignals: List<DetectionSignal>,
        texts: Texts,
        videoProbability: (Double) -> Float = ModelConfig::videoCalibratedProbability,
    ): DetectionSignal {
        val available = perFrameSignals.filter {
            it.availability == SignalAvailability.AVAILABLE && it.score != null
        }
        if (available.isEmpty()) {
            // Frames can fail with the model bundled; only blame bundling when a frame said so.
            if (perFrameSignals.isNotEmpty() &&
                perFrameSignals.none { it.availability == SignalAvailability.UNAVAILABLE }
            ) {
                return DetectionSignal.error(SignalType.AI_CLASSIFIER, texts.get(R.string.video_classifier_failed))
            }
            return DetectionSignal.unavailable(SignalType.AI_CLASSIFIER, texts.get(R.string.video_classifier_not_bundled))
        }

        val scores = available.map { it.score!! }
        val meanRaw = available.map {
            it.rawScore ?: ModelConfig.logitDifferenceFromCalibratedProbability(it.score!!)
        }.average()
        val videoScore = videoProbability(meanRaw)
        val meanConfidence = available.map { it.confidence }.average().toFloat()

        return DetectionSignal(
            type = SignalType.AI_CLASSIFIER,
            availability = SignalAvailability.AVAILABLE,
            score = videoScore,
            confidence = meanConfidence,
            description = texts.quantity(
                R.plurals.video_classifier_estimate, available.size, displayPercent(videoScore), available.size,
            ),
            evidence = texts.get(
                R.string.video_per_frame_evidence,
                scores.joinToString(", ") { texts.get(R.string.card_value_percent, displayPercent(it)) },
            ),
        )
    }
}
