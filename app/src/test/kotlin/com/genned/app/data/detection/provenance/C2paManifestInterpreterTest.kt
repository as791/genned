package com.genned.app.data.detection.provenance

import com.genned.app.data.detection.provenance.C2paManifestInterpreter.DeclaredSource
import com.genned.app.data.detection.provenance.C2paManifestInterpreter.Trust
import com.google.common.truth.Truth.assertThat
import org.junit.Assert.assertThrows
import org.junit.Test

/**
 * Fixtures in src/test/resources/c2pa are real c2patool reports (same c2pa-rs core as the
 * Android library) for a test image signed by a throwaway test CA, read with that CA trusted
 * ("_trusted") and not ("_untrusted"); "tampered" had its image bytes changed after signing.
 */
class C2paManifestInterpreterTest {

    private fun fixture(name: String) =
        checkNotNull(javaClass.getResource("/c2pa/$name.json")) { "missing fixture $name" }.readText()

    private fun interpret(name: String) = C2paManifestInterpreter.interpret(fixture(name))

    @Test
    fun `trusted AI-generated manifest`() {
        val f = interpret("ai_trusted")
        assertThat(f.trust).isEqualTo(Trust.TRUSTED)
        assertThat(f.source).isEqualTo(DeclaredSource.GENERATIVE_AI)
        assertThat(f.signer).isEqualTo("Genned Test")
        assertThat(f.aiTool).isEqualTo("TestGen")
        assertThat(f.failureCodes).isEmpty()
    }

    @Test
    fun `trusted composite with AI counts as AI`() {
        val f = interpret("composite_trusted")
        assertThat(f.trust).isEqualTo(Trust.TRUSTED)
        assertThat(f.source).isEqualTo(DeclaredSource.GENERATIVE_AI)
    }

    @Test
    fun `trusted camera capture`() {
        val f = interpret("capture_trusted")
        assertThat(f.trust).isEqualTo(Trust.TRUSTED)
        assertThat(f.source).isEqualTo(DeclaredSource.CAPTURE)
        assertThat(f.aiTool).isNull()
    }

    @Test
    fun `a valid signature from a signer not on the trust list is untrusted`() {
        for (name in listOf("ai_untrusted", "capture_untrusted", "composite_untrusted")) {
            assertThat(interpret(name).trust).isEqualTo(Trust.UNTRUSTED)
        }
        assertThat(interpret("ai_untrusted").source).isEqualTo(DeclaredSource.GENERATIVE_AI)
    }

    @Test
    fun `a file changed after signing is invalid even with a trusted signer`() {
        for (name in listOf("tampered_trusted", "tampered_untrusted")) {
            val f = interpret(name)
            assertThat(f.trust).isEqualTo(Trust.INVALID)
            assertThat(f.failureCodes).contains("assertion.dataHash.mismatch")
        }
    }

    @Test
    fun `a trusted manifest that declares neither AI nor capture`() {
        val json = fixture("ai_trusted").replace("trainedAlgorithmicMedia", "digitalArt")
        val f = C2paManifestInterpreter.interpret(json)
        assertThat(f.trust).isEqualTo(Trust.TRUSTED)
        assertThat(f.source).isEqualTo(DeclaredSource.NOT_DECLARED)
    }

    @Test
    fun `AI declared in an ingredient's manifest beats a capture claim in the active one`() {
        val capture = fixture("capture_trusted")
        // Add a second (ingredient) manifest that declares generative AI.
        val ingredient = """
            "urn:c2pa:ingredient": {"assertions": [{"label": "c2pa.actions", "data": {"actions": [
              {"action": "c2pa.created", "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}
            ]}}]},
        """.trimIndent()
        val json = capture.replaceFirst("\"manifests\": {", "\"manifests\": {$ingredient")
        assertThat(C2paManifestInterpreter.interpret(json).source).isEqualTo(DeclaredSource.GENERATIVE_AI)
    }

    @Test
    fun `older reports without validation_state fall back to the status codes`() {
        val untrusted = fixture("ai_untrusted").replace("\"validation_state\": \"Valid\"", "\"x\": 0")
        assertThat(C2paManifestInterpreter.interpret(untrusted).trust).isEqualTo(Trust.UNTRUSTED)
        val tampered = fixture("tampered_trusted").replace("\"validation_state\": \"Invalid\"", "\"x\": 0")
        assertThat(C2paManifestInterpreter.interpret(tampered).trust).isEqualTo(Trust.INVALID)
    }

    @Test
    fun `garbage is rejected, not guessed at`() {
        assertThrows(Exception::class.java) { C2paManifestInterpreter.interpret("[]") }
        assertThrows(Exception::class.java) { C2paManifestInterpreter.interpret("""{"manifests": {}}""") }
        assertThrows(Exception::class.java) { C2paManifestInterpreter.interpret("not json") }
    }
}
