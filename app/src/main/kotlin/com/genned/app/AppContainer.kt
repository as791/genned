package com.genned.app

import android.content.Context
import com.genned.app.data.analysis.AnalyzeImageUseCase
import com.genned.app.data.analysis.AnalyzeVideoUseCase
import com.genned.app.data.detection.classifier.AIImageClassifierProvider
import com.genned.app.data.detection.metadata.ExifMetadataProvider
import com.genned.app.data.detection.metadata.KnownGeneratorMetadataProvider
import com.genned.app.data.detection.provenance.C2PAProvider
import com.genned.app.data.detection.watermark.WatermarkProvider
import com.genned.app.data.image.ImageLoader
import com.genned.app.data.sharing.ResultCardRenderer
import com.genned.app.data.storage.AnalysisDatabase
import com.genned.app.data.storage.HistoryRepository
import com.genned.app.data.storage.ThumbnailStore
import com.genned.app.data.video.VideoFrameSampler
import com.genned.app.text.ResourceEvidenceTexts
import com.genned.app.text.ResourceTexts
import com.genned.app.text.Texts
import com.genned.domain.evidence.EvidenceEngine
import com.genned.domain.evidence.EvidenceWeights

/**
 * Hand-written composition root. No DI framework: the object graph here is small and
 * static for the lifetime of the process, and one engineer can read this file top to
 * bottom faster than tracing generated Hilt/Koin wiring — see internal-docs/ARCHITECTURE.md
 * "Why no DI framework".
 */
class AppContainer(context: Context) {
    private val appContext = context.applicationContext

    private val database by lazy { AnalysisDatabase.getInstance(appContext) }
    private val thumbnailStore by lazy { ThumbnailStore(appContext) }

    val imageLoader by lazy { ImageLoader(appContext) }
    private val videoFrameSampler by lazy { VideoFrameSampler(appContext) }
    val historyRepository by lazy { HistoryRepository(database.analysisDao(), thumbnailStore) }
    val resultCardRenderer by lazy { ResultCardRenderer(appContext) }

    /** User-facing text for signals and results, from string resources (translatable). */
    private val texts: Texts by lazy { ResourceTexts(appContext.resources) }
    private val evidenceTexts by lazy { ResourceEvidenceTexts(texts) }

    private val c2paProvider by lazy { C2PAProvider(appContext, texts) }
    private val exifMetadataProvider by lazy { ExifMetadataProvider(texts) }
    private val generatorMetadataProvider by lazy { KnownGeneratorMetadataProvider(texts) }
    private val classifierProvider by lazy { AIImageClassifierProvider(appContext) }
    private val watermarkProvider by lazy { WatermarkProvider(texts) }

    /** Video keeps the default LOW cut-off; photos and screen captures use the photo one. */
    private val evidenceEngine by lazy { EvidenceEngine(EvidenceWeights.LOW_THRESHOLD, evidenceTexts) }
    private val photoEvidenceEngine by lazy { EvidenceEngine(EvidenceWeights.PHOTO_LOW_THRESHOLD, evidenceTexts) }

    val analyzeImageUseCase by lazy {
        AnalyzeImageUseCase(
            provenanceProviders = listOf(c2paProvider),
            metadataProviders = listOf(exifMetadataProvider, generatorMetadataProvider),
            visualProviders = listOf(classifierProvider, watermarkProvider),
            evidenceEngine = photoEvidenceEngine,
            historyRepository = historyRepository,
            texts = texts,
        )
    }

    val analyzeVideoUseCase by lazy {
        AnalyzeVideoUseCase(
            frameSampler = videoFrameSampler,
            classifierProvider = classifierProvider.videoFrames,
            watermarkProvider = watermarkProvider,
            evidenceEngine = evidenceEngine,
            historyRepository = historyRepository,
            videoProbability = classifierProvider::videoProbability,
            contentCredentials = c2paProvider::analyzeVideo,
            texts = texts,
        )
    }
}
