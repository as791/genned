package com.genned.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onRoot
import androidx.lifecycle.SavedStateHandle
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.genned.app.data.settings.AppPreferences
import com.genned.app.data.sharing.ResultCardRenderer
import com.genned.app.data.storage.AnalysisDatabase
import com.genned.app.data.storage.HistoryRepository
import com.genned.app.data.storage.MediaKind
import com.genned.app.data.storage.ThumbnailStore
import com.genned.app.ui.guide.GuideScreen
import com.genned.app.ui.history.HistoryScreen
import com.genned.app.ui.history.HistoryViewModel
import com.genned.app.ui.home.HomeScreen
import com.genned.app.ui.home.HomeViewModel
import com.genned.app.ui.overlay.OverlaySetupScreen
import com.genned.app.ui.result.ResultScreen
import com.genned.app.ui.result.ResultViewModel
import com.genned.app.ui.theme.GennedTheme
import com.genned.domain.evidence.EvidenceEngine
import com.genned.domain.evidence.EvidenceWeights
import com.genned.domain.model.AnalysisResult
import com.genned.domain.model.DetectionSignal
import com.genned.domain.model.SignalAvailability
import com.genned.domain.model.SignalType
import com.github.takahirom.roborazzi.captureRoboImage
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File

/**
 * Renders the main screens with sample data and saves them as PNGs in app/build/screens,
 * which CI uploads as the "screens" artifact. It lets the design be looked at without a
 * phone; it asserts nothing about pixels.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [33], qualifiers = PHONE)
class ScreenshotTest {

    @get:Rule
    val compose = createComposeRule()

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private lateinit var database: AnalysisDatabase
    private lateinit var repository: HistoryRepository

    @Before
    fun setUp() {
        database = Room.inMemoryDatabaseBuilder(context, AnalysisDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        repository = HistoryRepository(database.analysisDao(), ThumbnailStore(context))
    }

    @After
    fun tearDown() {
        database.close()
    }

    // --- Sample data ---

    private fun picture(): File {
        val bitmap = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        Canvas(bitmap).drawPaint(
            Paint().apply {
                shader = LinearGradient(0f, 0f, 512f, 512f, 0xFF4A50C4.toInt(), 0xFFE8A04C.toInt(), Shader.TileMode.CLAMP)
            },
        )
        return File.createTempFile("sample", ".jpg").also { file ->
            file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        }
    }

    private fun available(
        type: SignalType,
        score: Float,
        description: String,
        evidence: String? = null,
        confidence: Float = 1f,
    ) = DetectionSignal(type, SignalAvailability.AVAILABLE, score, confidence, description, evidence)

    private fun photoSignals(aiProbability: Float) = listOf(
        DetectionSignal.unavailable(SignalType.CONTENT_CREDENTIALS, context.getString(R.string.credentials_none)),
        // Same score and confidence as ExifMetadataProvider gives "no camera details".
        available(SignalType.EXIF_METADATA, 0.3f, context.getString(R.string.exif_no_camera_metadata), confidence = 0.3f),
        available(SignalType.GENERATOR_METADATA, 0f, context.getString(R.string.generator_none_found)),
        available(
            SignalType.AI_CLASSIFIER,
            aiProbability,
            context.getString(R.string.classifier_estimate, (aiProbability * 100).toInt()),
        ),
        DetectionSignal.unavailable(SignalType.WATERMARK, context.getString(R.string.watermark_unavailable)),
    )

    private fun verifiedSignals() = listOf(
        available(
            SignalType.CONTENT_CREDENTIALS,
            0.98f,
            context.getString(R.string.credentials_trusted_ai, "OpenAI"),
            context.getString(R.string.credentials_evidence_tool, "ChatGPT"),
        ),
    ) + photoSignals(0.41f).drop(1)

    private fun videoSignals() = listOf(
        available(
            SignalType.AI_CLASSIFIER,
            0.47f,
            context.resources.getQuantityString(R.plurals.video_classifier_estimate, 6, 47, 6),
        ),
        DetectionSignal.unavailable(SignalType.WATERMARK, context.getString(R.string.watermark_unavailable)),
        DetectionSignal.unavailable(SignalType.GENERATOR_METADATA, context.getString(R.string.video_metadata_not_implemented)),
        DetectionSignal.unavailable(SignalType.EXIF_METADATA, context.getString(R.string.video_metadata_not_implemented)),
        DetectionSignal.unavailable(SignalType.CONTENT_CREDENTIALS, context.getString(R.string.credentials_none)),
    )

    private fun save(signals: List<DetectionSignal>, kind: MediaKind = MediaKind.PHOTO): Long = runBlocking {
        // The same LOW cut-offs the app uses (photos and screen captures, and video).
        val lowCutOff = if (kind == MediaKind.VIDEO) EvidenceWeights.LOW_THRESHOLD else EvidenceWeights.PHOTO_LOW_THRESHOLD
        val result: AnalysisResult = EvidenceEngine(lowCutOff).aggregate(signals)
        repository.save(result, picture(), kind)
    }

    // --- Rendering ---

    private fun shoot(name: String, waitForText: String, content: @Composable () -> Unit) {
        compose.setContent {
            GennedTheme(dynamicColor = false) {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    content()
                }
            }
        }
        // History comes from the database on a background thread.
        compose.waitUntil(timeoutMillis = 15_000) {
            compose.onAllNodesWithText(waitForText, substring = true).fetchSemanticsNodes().isNotEmpty()
        }
        compose.onRoot().captureRoboImage("build/screens/$name.png")
    }

    @Composable
    private fun Home() {
        HomeScreen(
            onMediaSelected = { _, _ -> },
            onOpenHistory = {},
            onOpenSettings = {},
            onOpenGuide = {},
            onOpenOverlaySetup = {},
            onOpenHistoryItem = {},
            viewModel = HomeViewModel(repository, AppPreferences(context)),
        )
    }

    private fun result(id: Long, name: String) = shoot(name, context.getString(R.string.result_why_heading)) {
        ResultScreen(
            onCheckAnother = {},
            onBack = {},
            onOpenGuide = {},
            viewModel = ResultViewModel(
                SavedStateHandle(mapOf("analysisId" to id.toString())),
                repository,
                ResultCardRenderer(context),
            ),
        )
    }

    // --- Screens ---

    @Test
    fun homeFirstRun() = shoot("01_home_first_run", context.getString(R.string.home_guide_heading)) { Home() }

    @Test
    fun homeWithHistory() {
        save(photoSignals(0.96f))
        save(videoSignals(), MediaKind.VIDEO)
        save(photoSignals(0.07f))
        shoot("02_home_with_history", context.getString(R.string.home_recent_checks)) { Home() }
    }

    @Test
    fun resultLikelyAi() = result(save(photoSignals(0.96f)), "03_result_likely_ai")

    @Test
    fun resultUncertainHighScore() = result(save(photoSignals(0.86f)), "03b_result_uncertain_high_score")

    @Test
    fun resultUncertainVideo() = result(save(videoSignals(), MediaKind.VIDEO), "04_result_uncertain_video")

    @Test
    fun resultLikelyNotAi() = result(save(photoSignals(0.07f)), "05_result_likely_not_ai")

    @Test
    fun resultVerified() = result(save(verifiedSignals()), "06_result_verified")

    @Test
    @Config(sdk = [33], qualifiers = PHONE_NIGHT)
    fun resultLikelyAiDark() = result(save(photoSignals(0.96f)), "07_result_likely_ai_dark")

    @Test
    fun history() {
        save(photoSignals(0.96f))
        save(videoSignals(), MediaKind.VIDEO)
        save(photoSignals(0.07f), MediaKind.SCREEN)
        shoot("08_history", context.getString(R.string.history_group_today)) {
            HistoryScreen(onBack = {}, onOpenItem = {}, viewModel = HistoryViewModel(repository))
        }
    }

    @Test
    fun guide() = shoot("09_guide", context.getString(R.string.guide_results_heading)) { GuideScreen(onBack = {}) }

    @Test
    fun overlaySetup() = shoot("10_overlay_setup", context.getString(R.string.overlay_setup_steps_heading)) {
        OverlaySetupScreen(onBack = {})
    }
}

private const val PHONE = "w411dp-h891dp-xxhdpi"
private const val PHONE_NIGHT = "w411dp-h891dp-night-xxhdpi"
