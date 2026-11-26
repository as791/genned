package com.genned.app.data.detection.provenance

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.genned.app.data.detection.provenance.C2paManifestInterpreter.Trust
import com.google.common.truth.Truth.assertThat
import com.google.common.truth.Truth.assertWithMessage
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile

/**
 * Conformance with the reference tool: every sample file in the C2PA SDK's own test set
 * (c2pa-rs fixtures) is read the way the app reads it (sniffed file type, bundled trust
 * lists) and compared with what c2patool says about the same file. The workflow writes
 * c2patool's verdicts to assets/c2pa-fixtures/conformance/expected.json.
 *
 * Verdicts: "none" (no Content Credentials), "valid" (signature and hashes check out,
 * whether or not the signer is trusted), "invalid" (altered or broken), "remote",
 * "error", and on Genned's side "unsupported" (a file type the app can't read).
 * A file where both sides give a verdict among none / valid / invalid and they differ
 * fails the test. Everything else is reported (logcat tag [TAG]) but doesn't fail:
 * c2patool errors are mostly deliberately malformed samples, and "unsupported" is a
 * coverage gap, not a wrong answer.
 */
@RunWith(AndroidJUnit4::class)
class C2paConformanceTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val reader = NativeC2paReader { C2paTrustSettings.fromAssets(app.assets) }

    private fun fixture(name: String): File {
        val out = File(app.cacheDir, "conformance-$name")
        instrumentation.context.assets.open("$DIR/$name").use { input ->
            out.outputStream().use { input.copyTo(it) }
        }
        return out
    }

    /** Genned's verdict on one file, read exactly as C2PAProvider does. */
    private fun gennedVerdict(file: File): String {
        val head = file.inputStream().use { input -> ByteArray(32).let { it.copyOf(input.read(it).coerceAtLeast(0)) } }
        val mime = C2paMimeTypes.sniff(head, null) ?: return "unsupported"
        val result = RandomAccessFile(file, "r").channel.use { reader.read(it, mime) }
        return when (result) {
            is C2paReadResult.Report -> runCatching { C2paManifestInterpreter.interpret(result.json).trust }
                .map { if (it == Trust.INVALID) "invalid" else "valid" }
                .getOrDefault("error")
            C2paReadResult.NoManifest -> "none"
            C2paReadResult.RemoteManifest -> "remote"
            C2paReadResult.UnsupportedFormat -> "unsupported"
            C2paReadResult.UnsupportedDevice -> throw AssertionError("c2pa native library unavailable on this device")
            is C2paReadResult.Failed -> "error"
        }
    }

    @Test
    fun gennedAgreesWithC2patoolOnTheSdkSamples() {
        val expected = instrumentation.context.assets.open("$DIR/expected.json").bufferedReader()
            .use { JSONObject(it.readText()) }
        assertWithMessage("no conformance samples").that(expected.length()).isGreaterThan(0)

        val decisive = setOf("none", "valid", "invalid")
        val mismatches = mutableListOf<String>()
        val counts = sortedMapOf<String, Int>()
        for (name in expected.keys()) {
            val reference = expected.getString(name)
            val genned = gennedVerdict(fixture(name))
            val agree = genned == reference
            val kind = when {
                agree -> "agree"
                genned in decisive && reference in decisive -> "MISMATCH"
                else -> "differs (not counted)"
            }
            counts.merge(kind, 1, Int::plus)
            Log.i(TAG, "$name | c2patool=$reference | genned=$genned | $kind")
            if (kind == "MISMATCH") mismatches += "$name: c2patool says $reference, Genned says $genned"
        }
        Log.i(TAG, "summary | ${counts.entries.joinToString { "${it.key}: ${it.value}" }} | of ${expected.length()}")
        assertWithMessage(mismatches.joinToString("\n")).that(mismatches).isEmpty()
        assertThat(counts["agree"] ?: 0).isGreaterThan(0)
    }

    private companion object {
        const val DIR = "c2pa-fixtures/conformance"
        const val TAG = "GennedConformance"
    }
}
