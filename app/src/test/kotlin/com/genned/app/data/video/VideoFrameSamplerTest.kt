package com.genned.app.data.video

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Bad videos must end as an honest [VideoLoadException] or as a check on the frames that
 * could be read, never as a crash: a corrupt stream can make the platform decoder throw for
 * one frame, or run out of memory on a huge one.
 */
@RunWith(RobolectricTestRunner::class)
class VideoFrameSamplerTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val sampler = VideoFrameSampler(context)

    private fun source(durationMs: Long?, frame: (Int) -> Bitmap?) = object : FrameSource {
        var calls = 0
        override fun durationMs(): Long? = durationMs
        override fun frameAt(timeUs: Long): Bitmap? = frame(calls++)
    }

    private fun bitmap() = Bitmap.createBitmap(64, 48, Bitmap.Config.ARGB_8888)

    @Test
    fun `a readable clip gives one frame per sample point`() {
        val frames = sampler.sampleFrom(source(30_000) { bitmap() }, mutableListOf())
        assertThat(frames).hasSize(VideoFrameSampler.FRAME_COUNT)
        frames.forEach { it.file.delete() }
    }

    @Test
    fun `no duration or zero length is unsupported`() {
        for (duration in listOf(null, 0L, -5L)) {
            val e = runCatching { sampler.sampleFrom(source(duration) { bitmap() }, mutableListOf()) }.exceptionOrNull()
            assertThat(e).isInstanceOf(VideoLoadException.Unsupported::class.java)
        }
    }

    @Test
    fun `a clip over three minutes is too long`() {
        val e = runCatching { sampler.sampleFrom(source(3 * 60 * 1000L + 1) { bitmap() }, mutableListOf()) }.exceptionOrNull()
        assertThat(e).isInstanceOf(VideoLoadException.TooLong::class.java)
    }

    @Test
    fun `a frame the decoder throws on is skipped, not fatal`() {
        val frames = sampler.sampleFrom(
            source(30_000) { i -> if (i == 1) throw IllegalStateException("corrupt NAL unit") else bitmap() },
            mutableListOf(),
        )
        assertThat(frames).hasSize(VideoFrameSampler.FRAME_COUNT - 1)
        frames.forEach { it.file.delete() }
    }

    @Test
    fun `running out of memory keeps the frames already read`() {
        val frames = sampler.sampleFrom(
            source(30_000) { i -> if (i >= 2) throw OutOfMemoryError("huge frame") else bitmap() },
            mutableListOf(),
        )
        assertThat(frames).hasSize(2)
        frames.forEach { it.file.delete() }
    }

    @Test
    fun `a clip where every frame fails is unsupported`() {
        val e = runCatching {
            sampler.sampleFrom(source(30_000) { throw IllegalStateException("corrupt") }, mutableListOf())
        }.exceptionOrNull()
        assertThat(e).isInstanceOf(VideoLoadException.Unsupported::class.java)
    }
}
