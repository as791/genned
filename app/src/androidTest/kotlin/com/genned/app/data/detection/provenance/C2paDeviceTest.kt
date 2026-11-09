package com.genned.app.data.detection.provenance

import android.content.Context
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.genned.app.data.detection.provenance.C2paManifestInterpreter.DeclaredSource
import com.genned.app.data.detection.provenance.C2paManifestInterpreter.Trust
import com.genned.app.text.ResourceTexts
import com.genned.domain.model.AnalysisInput
import com.genned.domain.model.SignalAvailability
import com.google.common.truth.Truth.assertThat
import com.google.common.truth.Truth.assertWithMessage
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile

/**
 * The real c2pa-android library on a device. Fixtures (assets/c2pa-fixtures) are made by
 * .github/workflows/c2pa-device-test.yml: an image signed by a throwaway test CA as
 * AI-generated, camera-captured, and AI-generated then altered, plus the unsigned image
 * and the CA certificate.
 */
@RunWith(AndroidJUnit4::class)
class C2paDeviceTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app: Context = instrumentation.targetContext

    private fun fixture(name: String): File {
        val out = File(app.cacheDir, "c2pa-fixture-$name")
        instrumentation.context.assets.open("c2pa-fixtures/$name").use { input ->
            out.outputStream().use { input.copyTo(it) }
        }
        return out
    }

    private val testAnchorSettings by lazy {
        val anchor = instrumentation.context.assets.open("c2pa-fixtures/anchor.pem").bufferedReader().use { it.readText() }
        val config = app.assets.open("c2pa/trust-config.cfg").bufferedReader().use { it.readText() }
        C2paTrustSettings.json(anchor, config)
    }

    private val testReader = NativeC2paReader { testAnchorSettings }
    private val bundledReader = NativeC2paReader { C2paTrustSettings.fromAssets(app.assets) }

    private fun read(reader: C2paReader, name: String, mime: String = "image/jpeg"): C2paReadResult =
        RandomAccessFile(fixture(name), "r").channel.use { reader.read(it, mime) }

    private fun findings(reader: C2paReader, name: String, mime: String = "image/jpeg"): C2paManifestInterpreter.Findings {
        val result = read(reader, name, mime)
        assertWithMessage("$name: $result").that(result).isInstanceOf(C2paReadResult.Report::class.java)
        return C2paManifestInterpreter.interpret((result as C2paReadResult.Report).json)
    }

    @Test
    fun trustedSignerAndDeclaredSourcesAreRead() {
        findings(testReader, "ai.jpg").let {
            assertThat(it.trust).isEqualTo(Trust.TRUSTED)
            assertThat(it.source).isEqualTo(DeclaredSource.GENERATIVE_AI)
            assertThat(it.signer).isEqualTo("Genned Test")
        }
        findings(testReader, "capture.jpg").let {
            assertThat(it.trust).isEqualTo(Trust.TRUSTED)
            assertThat(it.source).isEqualTo(DeclaredSource.CAPTURE)
        }
        findings(testReader, "ai.png", "image/png").let {
            assertThat(it.trust).isEqualTo(Trust.TRUSTED)
            assertThat(it.source).isEqualTo(DeclaredSource.GENERATIVE_AI)
        }
    }

    @Test
    fun alteredFileFailsVerification() {
        assertThat(findings(testReader, "tampered.jpg").trust).isEqualTo(Trust.INVALID)
    }

    @Test
    fun bundledTrustListsLoadAndDoNotTrustTheTestSigner() {
        assertThat(findings(bundledReader, "ai.jpg").trust).isEqualTo(Trust.UNTRUSTED)
    }

    @Test
    fun unsignedFileHasNoManifest() {
        assertThat(read(testReader, "base.jpg")).isEqualTo(C2paReadResult.NoManifest)
        assertThat(read(bundledReader, "base.jpg")).isEqualTo(C2paReadResult.NoManifest)
    }

    @Test
    fun providerEndToEnd() = runBlocking {
        val provider = C2PAProvider(app, ResourceTexts(app.resources), testReader)
        val file = fixture("ai.jpg")
        val signal = provider.analyze(
            AnalysisInput(
                originalFilePath = file.absolutePath,
                normalizedFilePath = file.absolutePath,
                originalMimeType = null,
                widthPx = 1,
                heightPx = 1,
                fileSizeBytes = file.length(),
            ),
        )
        assertWithMessage(signal.description).that(signal.availability).isEqualTo(SignalAvailability.AVAILABLE)
        assertThat(signal.score).isEqualTo(C2PAProvider.AI_SCORE)

        val video = provider.analyzeVideo(Uri.fromFile(fixture("ai.mp4")))
        assertWithMessage(video.description).that(video.availability).isEqualTo(SignalAvailability.AVAILABLE)
        assertThat(video.score).isEqualTo(C2PAProvider.AI_SCORE)

        val plain = provider.analyze(
            AnalysisInput(fixture("base.jpg").absolutePath, file.absolutePath, "image/jpeg", 1, 1, 1),
        )
        assertThat(plain.availability).isEqualTo(SignalAvailability.UNAVAILABLE)
    }
}
