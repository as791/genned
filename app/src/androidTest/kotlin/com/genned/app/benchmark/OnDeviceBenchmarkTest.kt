package com.genned.app.benchmark

import android.net.Uri
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.genned.app.AppContainer
import com.genned.domain.model.AnalysisInput
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/**
 * What a check costs on a device: time for a photo check (the first, which loads the
 * models, and the median of later ones), a video check, and peak memory while checking.
 * Runs the app's own pipeline ([AppContainer]) on the fixtures that
 * .github/workflows/c2pa-device-test.yml makes. On the CI emulator the numbers are only
 * relative (a regression check), not phone speeds. Results go to logcat under [TAG] as
 * `key=value` lines, which the workflow copies into its summary.
 */
@RunWith(AndroidJUnit4::class)
class OnDeviceBenchmarkTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext

    private fun fixture(name: String): File {
        val out = File(app.cacheDir, "benchmark-$name")
        instrumentation.context.assets.open("c2pa-fixtures/$name").use { input ->
            out.outputStream().use { input.copyTo(it) }
        }
        return out
    }

    private fun usedBytes(): Long =
        Debug.getNativeHeapAllocatedSize() + Runtime.getRuntime().let { it.totalMemory() - it.freeMemory() }

    /** Samples Java + native heap every 20 ms while [block] runs; returns (result, peak bytes). */
    private inline fun <T> withPeakMemory(block: () -> T): Pair<T, Long> {
        val peak = AtomicLong(usedBytes())
        val running = AtomicBoolean(true)
        val sampler = thread(name = "peak-memory") {
            while (running.get()) {
                peak.accumulateAndGet(usedBytes()) { a, b -> maxOf(a, b) }
                Thread.sleep(20)
            }
        }
        try {
            val result = block()
            return result to peak.accumulateAndGet(usedBytes()) { a, b -> maxOf(a, b) }
        } finally {
            running.set(false)
            sampler.join()
        }
    }

    private inline fun millis(block: () -> Unit): Long {
        val start = System.nanoTime()
        block()
        return (System.nanoTime() - start) / 1_000_000
    }

    private fun report(key: String, value: Any) = Log.i(TAG, "$key=$value")

    @Test
    fun photoAndVideoCheckCost() = runBlocking {
        val container = AppContainer(app)
        val photo = Uri.fromFile(fixture("base.jpg"))
        val video = Uri.fromFile(fixture("base.mp4"))

        suspend fun checkPhoto() {
            val normalized = container.imageLoader.normalize(photo)
            val input = AnalysisInput(
                originalFilePath = normalized.originalFile.absolutePath,
                normalizedFilePath = normalized.normalizedFile.absolutePath,
                originalMimeType = normalized.originalMimeType,
                widthPx = normalized.widthPx,
                heightPx = normalized.heightPx,
                fileSizeBytes = normalized.fileSizeBytes,
                classifierCrop = normalized.classifierCrop,
            )
            val (_, result) = container.analyzeImageUseCase.run(input, normalized.normalizedFile) {}
            assertThat(result.signals).isNotEmpty()
        }

        suspend fun checkVideo() {
            val (_, result) = container.analyzeVideoUseCase.run(video, onPreview = {}, onStage = {})
            assertThat(result.signals).isNotEmpty()
        }

        val baseline = usedBytes()
        val (photoTimes, photoPeak) = withPeakMemory {
            val first = millis { checkPhoto() }
            first to List(PHOTO_REPEATS) { millis { checkPhoto() } }
        }
        val (videoTimes, videoPeak) = withPeakMemory {
            val first = millis { checkVideo() }
            first to List(VIDEO_REPEATS) { millis { checkVideo() } }
        }

        report("device", "${android.os.Build.MODEL} (API ${android.os.Build.VERSION.SDK_INT}, ${android.os.Build.SUPPORTED_ABIS.first()})")
        report("photo_first_ms", photoTimes.first)
        report("photo_median_ms", photoTimes.second.sorted()[PHOTO_REPEATS / 2])
        report("video_first_ms", videoTimes.first)
        report("video_median_ms", videoTimes.second.sorted()[VIDEO_REPEATS / 2])
        report("memory_before_mb", baseline / MB)
        report("photo_peak_mb", photoPeak / MB)
        report("video_peak_mb", videoPeak / MB)
    }

    private companion object {
        const val TAG = "GennedBenchmark"
        const val PHOTO_REPEATS = 5
        const val VIDEO_REPEATS = 3
        const val MB = 1024 * 1024
    }
}
