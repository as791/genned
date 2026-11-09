package com.genned.app.data.detection.provenance

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class C2paReaderHelpersTest {

    private fun bytes(vararg parts: Any): ByteArray = parts.flatMap { p ->
        when (p) {
            is String -> p.map { it.code.toByte() }
            is Int -> listOf(p.toByte())
            else -> error("bad part")
        }
    }.toByteArray()

    @Test
    fun `file type comes from the bytes, not the declared type`() {
        assertThat(C2paMimeTypes.sniff(bytes(0xFF, 0xD8, 0xFF, 0xE0), "image/png")).isEqualTo("image/jpeg")
        assertThat(C2paMimeTypes.sniff(bytes(0x89, "PNG", 0x0D, 0x0A), null)).isEqualTo("image/png")
        assertThat(C2paMimeTypes.sniff(bytes("RIFF", 0, 0, 0, 0, "WEBPVP8 "), null)).isEqualTo("image/webp")
        assertThat(C2paMimeTypes.sniff(bytes(0, 0, 0, 0x18, "ftypheic"), null)).isEqualTo("image/heic")
        assertThat(C2paMimeTypes.sniff(bytes(0, 0, 0, 0x18, "ftypavif"), null)).isEqualTo("image/avif")
        assertThat(C2paMimeTypes.sniff(bytes(0, 0, 0, 0x18, "ftypisom"), null)).isEqualTo("video/mp4")
        assertThat(C2paMimeTypes.sniff(bytes(0, 0, 0, 0x14, "ftypqt  "), null)).isEqualTo("video/quicktime")
    }

    @Test
    fun `unknown bytes fall back to a concrete declared type only`() {
        assertThat(C2paMimeTypes.sniff(bytes("abcd"), "video/webm")).isEqualTo("video/webm")
        assertThat(C2paMimeTypes.sniff(bytes("abcd"), "image/*")).isNull()
        assertThat(C2paMimeTypes.sniff(bytes("abcd"), null)).isNull()
        assertThat(C2paMimeTypes.sniff(ByteArray(0), null)).isNull()
    }

    @Test
    fun `reader errors are classified`() {
        assertThat(NativeC2paReader.classify("ManifestNotFound: no JUMBF data found")).isEqualTo(C2paReadResult.NoManifest)
        assertThat(NativeC2paReader.classify("No claim found")).isEqualTo(C2paReadResult.NoManifest)
        assertThat(NativeC2paReader.classify("RemoteManifestUrl: https://example.com/m.c2pa")).isEqualTo(C2paReadResult.RemoteManifest)
        assertThat(NativeC2paReader.classify("UnsupportedType")).isEqualTo(C2paReadResult.UnsupportedFormat)
        assertThat(NativeC2paReader.classify("something else")).isInstanceOf(C2paReadResult.Failed::class.java)
    }
}
