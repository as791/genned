package com.genned.app.data.detection.provenance

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive

/**
 * Reads the JSON report a C2PA reader produces for a file (the c2pa-rs `Reader` format:
 * `active_manifest`, `manifests`, `validation_status`, `validation_state`) and boils it
 * down to what Genned needs: was the manifest verified and signed by a trusted signer, and
 * what does it say about how the content was made.
 *
 * Pure parsing, no Android or native code, so it is unit-tested against reports produced
 * by c2patool for signed, untrusted and tampered files (see C2paManifestInterpreterTest).
 */
object C2paManifestInterpreter {

    enum class Trust {
        /** Signature, content hashes and certificate chain all check out against the bundled trust list. */
        TRUSTED,

        /** Signature and hashes check out, but the signer is not on the trust list (anyone can make such a manifest). */
        UNTRUSTED,

        /** Verification failed: the file changed after signing, the signature is broken, or the certificate is bad. */
        INVALID,
    }

    enum class DeclaredSource {
        /** Some manifest declares generative-AI content (IPTC trainedAlgorithmicMedia or compositeWithTrainedAlgorithmicMedia). */
        GENERATIVE_AI,

        /** The active manifest declares a camera/film capture and nothing declares generative AI. */
        CAPTURE,

        /** No source type that answers the question either way. */
        NOT_DECLARED,
    }

    data class Findings(
        val trust: Trust,
        val source: DeclaredSource,
        /** Who signed the active manifest (certificate organization, else common name), if stated. */
        val signer: String?,
        /** The tool named on the generative-AI action, if any (e.g. the generator's name). */
        val aiTool: String?,
        /** Validation failure codes, for the "why" section. */
        val failureCodes: List<String>,
    )

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private const val IPTC = "digitalsourcetype/"
    private val AI_SOURCES = setOf("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia")
    private val CAPTURE_SOURCES = setOf("digitalCapture", "computationalCapture", "negativeFilm", "positiveFilm")

    /** @throws IllegalArgumentException if [readerJson] is not a reader report with an active manifest. */
    fun interpret(readerJson: String): Findings {
        val root = json.parseToJsonElement(readerJson) as? JsonObject
            ?: throw IllegalArgumentException("not a JSON object")
        val manifests = root["manifests"] as? JsonObject ?: JsonObject(emptyMap())
        val activeLabel = root.string("active_manifest")
            ?: throw IllegalArgumentException("no active manifest")
        val active = manifests[activeLabel] as? JsonObject
            ?: throw IllegalArgumentException("active manifest $activeLabel missing")

        val statusCodes = (root["validation_status"] as? JsonArray).orEmpty()
            .mapNotNull { (it as? JsonObject)?.string("code") }
        val trust = when {
            root.string("validation_state") == "Trusted" -> Trust.TRUSTED
            root.string("validation_state") == "Invalid" -> Trust.INVALID
            // "Valid" means the signature and hashes check out; the signer still isn't trusted.
            root.string("validation_state") == "Valid" -> Trust.UNTRUSTED
            // Older reports have no validation_state: any status other than an untrusted
            // signer is a failure.
            statusCodes.any { it != "signingCredential.untrusted" } -> Trust.INVALID
            else -> Trust.UNTRUSTED
        }

        val activeActions = actionsOf(active)
        val allActions = manifests.values.filterIsInstance<JsonObject>().flatMap(::actionsOf)
        val aiAction = allActions.firstOrNull { sourceOf(it) in AI_SOURCES }
        val source = when {
            aiAction != null -> DeclaredSource.GENERATIVE_AI
            activeActions.any { sourceOf(it) in CAPTURE_SOURCES } -> DeclaredSource.CAPTURE
            else -> DeclaredSource.NOT_DECLARED
        }

        val signature = active["signature_info"] as? JsonObject
        return Findings(
            trust = trust,
            source = source,
            signer = (signature?.string("issuer") ?: signature?.string("common_name"))?.takeIf { it.isNotBlank() },
            aiTool = aiAction?.let(::softwareAgentOf),
            failureCodes = statusCodes.filter { it != "signingCredential.untrusted" },
        )
    }

    /** Every action in a manifest's `c2pa.actions` / `c2pa.actions.v2` assertions. */
    private fun actionsOf(manifest: JsonObject): List<JsonObject> =
        (manifest["assertions"] as? JsonArray).orEmpty()
            .filterIsInstance<JsonObject>()
            .filter { it.string("label")?.let { l -> l == "c2pa.actions" || l.startsWith("c2pa.actions.") } == true }
            .flatMap { ((it["data"] as? JsonObject)?.get("actions") as? JsonArray).orEmpty() }
            .filterIsInstance<JsonObject>()

    /** The IPTC digital source type's last path segment, e.g. "trainedAlgorithmicMedia". */
    private fun sourceOf(action: JsonObject): String? =
        action.string("digitalSourceType")?.substringAfter(IPTC)?.substringAfterLast('/')

    private fun softwareAgentOf(action: JsonObject): String? = when (val agent = action["softwareAgent"]) {
        is JsonObject -> agent.string("name")
        null -> null
        else -> agent.primitiveOrNull()
    }?.takeIf { it.isNotBlank() }

    private fun JsonObject.string(key: String): String? = this[key]?.primitiveOrNull()

    private fun JsonElement.primitiveOrNull(): String? =
        runCatching { jsonPrimitive.contentOrNull }.getOrNull()

    private fun JsonArray?.orEmpty(): List<JsonElement> = this ?: emptyList()
}
