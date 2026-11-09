package com.genned.app.data.analysis

import android.net.Uri
import com.genned.app.R
import com.genned.app.data.storage.HistoryRepository
import com.genned.app.data.video.SampledFrame
import com.genned.app.data.video.VideoFrameSampler
import com.genned.app.text.Texts
import com.genned.domain.evidence.EvidenceEngine
import com.genned.domain.model.AnalysisInput
import com.genned.domain.model.AnalysisResult
import com.genned.domain.model.DetectionSignal
import com.genned.domain.model.SignalType
import com.genned.domain.provider.DetectionProvider
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope

/**
 * Video support: sample a handful of still frames (see [VideoFrameSampler]) and run
 * the *same* on-device image classifier used for photos on each one, then combine
 * the per-frame scores ([VideoSignalAggregator]) into one signal fed through the
 * same [EvidenceEngine] as the image path.
 *
 * This is deliberately labeled and limited to what it actually is: frame-sampled
 * still-image classification, not video, audio, or motion/temporal analysis — no
 * video-specific model exists here or is implied. Metadata/provenance inspection
 * (EXIF, generator signatures, Content Credentials) is not implemented for video
 * in this version; that is stated explicitly in the result's signals and
 * limitations rather than silently skipped.
 */
class AnalyzeVideoUseCase(
    private val frameSampler: VideoFrameSampler,
    private val classifierProvider: DetectionProvider,
    private val watermarkProvider: DetectionProvider,
    private val evidenceEngine: EvidenceEngine,
    private val historyRepository: HistoryRepository,
    /** Video calibration of the classifier that scores the frames (see VideoSignalAggregator). */
    private val videoProbability: (Double) -> Float,
    private val texts: Texts,
) {
    suspend fun run(
        videoUri: Uri,
        onPreview: suspend (String) -> Unit,
        onStage: suspend (AnalysisStage) -> Unit,
    ): Pair<Long, AnalysisResult> {
        onStage(AnalysisStage.SAMPLING_FRAMES)
        val frames = frameSampler.sampleFrames(videoUri)
        try {
            val middleFrame = frames[frames.size / 2]
            onPreview(middleFrame.file.absolutePath)

            onStage(AnalysisStage.VISUAL)
            val perFrameSignals = coroutineScope {
                frames.map { frame -> async { analyzeFrame(frame) } }.awaitAll()
            }
            val classifierSignal = VideoSignalAggregator.aggregateFrameSignals(perFrameSignals, texts, videoProbability)
            val watermarkSignal = runCatching { watermarkProvider.analyze(frameInput(middleFrame)) }
                .getOrElse { DetectionSignal.error(watermarkProvider.signalType, texts.get(R.string.signal_failed_unexpectedly)) }

            val signals = listOf(
                classifierSignal,
                watermarkSignal,
                DetectionSignal.unavailable(SignalType.GENERATOR_METADATA, texts.get(R.string.video_metadata_not_implemented)),
                DetectionSignal.unavailable(SignalType.EXIF_METADATA, texts.get(R.string.video_metadata_not_implemented)),
                DetectionSignal.unavailable(SignalType.CONTENT_CREDENTIALS, texts.get(R.string.credentials_unavailable)),
            )

            val baseResult = evidenceEngine.aggregate(signals)
            val result = baseResult.copy(limitations = baseResult.limitations + videoLimitation(frames.size))

            val analysisId = historyRepository.save(result, middleFrame.file)
            return analysisId to result
        } finally {
            frames.forEach { it.file.delete() }
        }
    }

    private suspend fun analyzeFrame(frame: SampledFrame): DetectionSignal =
        runCatching { classifierProvider.analyze(frameInput(frame)) }
            .getOrElse { DetectionSignal.error(classifierProvider.signalType, texts.get(R.string.signal_failed_unexpectedly)) }

    private fun frameInput(frame: SampledFrame) = AnalysisInput(
        originalFilePath = frame.file.absolutePath,
        normalizedFilePath = frame.file.absolutePath,
        originalMimeType = "image/jpeg",
        widthPx = frame.widthPx,
        heightPx = frame.heightPx,
        fileSizeBytes = frame.file.length(),
    )

    private fun videoLimitation(frameCount: Int): String =
        texts.quantity(R.plurals.video_limitation, frameCount, frameCount)
}
