package com.genned.app.data.image

import android.content.Context
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.test.runTest
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.GraphicsMode
import java.io.File

/**
 * Covers the failure paths a share-sheet URI can hit: not an image at all, and a
 * URI that can't be opened — both must fail with a specific, honest
 * [ImageLoadException] rather than crashing the analysis pipeline.
 *
 * [GraphicsMode.Mode.NATIVE] is required here: Robolectric's default ("legacy")
 * BitmapFactory shadow doesn't actually validate image bytes — it fakes a
 * successful decode for arbitrary input, which would make these malformed-image
 * tests pass vacuously (no exception ever thrown) regardless of what ImageLoader
 * does. NATIVE mode routes through Robolectric's real Skia-backed decoder, so
 * decode failures here reflect what a real device would actually do.
 *
 * These assert [ImageLoadException] broadly rather than a specific subtype:
 * whether malformed bytes are classified as [ImageLoadException.Unsupported] (failed
 * bounds decode) or [ImageLoadException.Corrupt] (decode itself throws) depends on
 * decoder-internal behavior that legitimately differs between real Android and
 * Robolectric's simulated BitmapFactory. Both are equally correct, honest outcomes
 * for malformed input — the real contract under test is "never crash, always a
 * specific ImageLoadException," not which exact subtype.
 */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class ImageLoaderTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val loader = ImageLoader(context)

    @Test
    fun `garbage bytes that are not an image fail with a specific exception, not a crash`() = runTest {
        val file = File.createTempFile("garbage", ".jpg")
        file.writeBytes(ByteArray(256) { it.toByte() })

        val exception = runCatching { loader.normalize(Uri.fromFile(file)) }.exceptionOrNull()

        assertThat(exception).isInstanceOf(ImageLoadException::class.java)
    }

    @Test
    fun `an empty file fails with a specific exception, not a crash`() = runTest {
        val file = File.createTempFile("empty", ".jpg")

        val exception = runCatching { loader.normalize(Uri.fromFile(file)) }.exceptionOrNull()

        assertThat(exception).isInstanceOf(ImageLoadException::class.java)
    }

    @Test
    fun `an image claiming a huge size is refused before decoding`() = runTest {
        // A valid PNG header claiming 30000 x 30000 pixels (900 megapixels) with almost no data.
        val file = File.createTempFile("bomb", ".png")
        file.writeBytes(pngHeader(30000, 30000))

        val exception = runCatching { loader.normalize(Uri.fromFile(file)) }.exceptionOrNull()

        assertThat(exception).isInstanceOf(ImageLoadException.TooLarge::class.java)
    }

    @Test
    fun `garbage in the EXIF block does not stop the check`() = runTest {
        val jpeg = smallJpeg()
        // Splice an APP1 "Exif" segment full of nonsense right after the SOI marker.
        val junk = "Exif".toByteArray() + byteArrayOf(0, 0, 0x4D, 0x4D, 0, 0x2A) + ByteArray(200) { (it * 37).toByte() }
        val segment = byteArrayOf(0xFF.toByte(), 0xE1.toByte(), ((junk.size + 2) shr 8).toByte(), (junk.size + 2).toByte())
        val file = File.createTempFile("bad_exif", ".jpg")
        file.writeBytes(jpeg.copyOfRange(0, 2) + segment + junk + jpeg.copyOfRange(2, jpeg.size))

        val result = runCatching { loader.normalize(Uri.fromFile(file)) }

        assertThat(result.isSuccess).isTrue()
    }

    @Test
    fun `mutated images either load or fail with an ImageLoadException`() = runTest {
        val base = smallJpeg()
        val random = java.util.Random(7)
        repeat(40) {
            val bytes = base.copyOf(if (random.nextInt(3) == 0) 2 + random.nextInt(base.size - 2) else base.size)
            repeat(1 + random.nextInt(6)) { bytes[2 + random.nextInt(bytes.size - 2)] = random.nextInt(256).toByte() }
            val file = File.createTempFile("fuzz", ".jpg").apply { writeBytes(bytes); deleteOnExit() }

            val exception = runCatching { loader.normalize(Uri.fromFile(file)) }.exceptionOrNull()

            if (exception != null) assertThat(exception).isInstanceOf(ImageLoadException::class.java)
        }
    }

    private fun smallJpeg(): ByteArray {
        val bitmap = android.graphics.Bitmap.createBitmap(48, 32, android.graphics.Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(0xFF3366AA.toInt())
        val out = java.io.ByteArrayOutputStream()
        bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
        return out.toByteArray()
    }

    /** PNG signature + IHDR (with a correct CRC) claiming width x height, then IEND. */
    private fun pngHeader(width: Int, height: Int): ByteArray {
        fun int32(v: Int) = byteArrayOf((v ushr 24).toByte(), (v ushr 16).toByte(), (v ushr 8).toByte(), v.toByte())
        fun chunk(type: String, data: ByteArray): ByteArray {
            val crc = java.util.zip.CRC32().apply { update(type.toByteArray()); update(data) }
            return int32(data.size) + type.toByteArray() + data + int32(crc.value.toInt())
        }
        val ihdr = int32(width) + int32(height) + byteArrayOf(8, 2, 0, 0, 0)
        val signature = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
        return signature + chunk("IHDR", ihdr) + chunk("IEND", ByteArray(0))
    }

    @Test
    fun `a URI pointing at nothing fails rather than crashing`() = runTest {
        val missingFileUri = Uri.fromFile(File(context.cacheDir, "does_not_exist_${System.nanoTime()}.jpg"))

        val exception = runCatching { loader.normalize(missingFileUri) }.exceptionOrNull()

        assertThat(exception).isInstanceOf(ImageLoadException::class.java)
    }
}
