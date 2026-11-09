package com.genned.app.data.detection.provenance

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.ParcelFileDescriptor
import android.util.Log
import com.genned.app.R
import com.genned.app.data.detection.provenance.C2paManifestInterpreter.DeclaredSource
import com.genned.app.data.detection.provenance.C2paManifestInterpreter.Trust
import com.genned.app.text.Texts
import com.genned.domain.model.AnalysisInput
import com.genned.domain.model.DetectionSignal
import com.genned.domain.model.SignalAvailability
import com.genned.domain.model.SignalType
import com.genned.domain.provider.DetectionProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.withTimeoutOrNull
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * Content Credentials (C2PA): a signed manifest embedded in the file that records how it was
 * made, e.g. "created with generative AI" by an image generator or "captured" by a camera.
 * This is the one signal [com.genned.domain.evidence.EvidenceEngine] treats as authoritative
 * rather than probabilistic, so it only reports AVAILABLE when that is earned:
 *
 * | What the file has | Signal |
 * |---|---|
 * | Verified manifest from a trusted signer declaring generative AI | AVAILABLE, score [AI_SCORE] |
 * | Verified manifest from a trusted signer declaring a camera capture, no AI anywhere | AVAILABLE, score [CAPTURE_SCORE] |
 * | Trusted manifest that says neither | UNAVAILABLE |
 * | Manifest from a signer not on the trust list (anyone can make one) | UNAVAILABLE; an AI claim is mentioned, never counted |
 * | Manifest that fails verification (file changed after signing, bad signature) | UNAVAILABLE |
 * | No manifest, online-only manifest, unsupported type, Android < 9, x86 release build | UNAVAILABLE |
 *
 * Reads the byte-exact original ([AnalysisInput.originalFilePath]), never the re-encoded
 * copy, which has no manifest. Everything runs on the phone; nothing is fetched (see
 * [C2paTrustSettings], internal-docs/PRIVACY.md). Most apps and social networks strip
 * Content Credentials, so finding none says nothing either way.
 */
class C2PAProvider(
    private val context: Context,
    private val texts: Texts,
    private val reader: C2paReader = NativeC2paReader(cachedSettings(context)),
    private val timeoutMs: Long = TIMEOUT_MS,
) : DetectionProvider {
    override val signalType: SignalType = SignalType.CONTENT_CREDENTIALS

    /** Blocking native reads run here so a timeout can abandon one without blocking the analysis. */
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override suspend fun analyze(image: AnalysisInput): DetectionSignal =
        verify(image.originalMimeType) { RandomAccessFile(image.originalFilePath, "r").channel }

    /** The same check for a video, read straight from its [uri] (videos are not copied). */
    suspend fun analyzeVideo(uri: Uri): DetectionSignal =
        verify(context.contentResolver.getType(uri)) {
            val pfd = context.contentResolver.openFileDescriptor(uri, "r")
                ?: throw IllegalStateException("no file descriptor")
            ParcelFileDescriptor.AutoCloseInputStream(pfd).channel
        }

    private suspend fun verify(declaredMime: String?, open: () -> FileChannel): DetectionSignal {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return unavailable(R.string.credentials_needs_android_9)
        val work = scope.async {
            runCatching {
                open().use { channel ->
                    val head = ByteBuffer.allocate(16)
                    channel.read(head)
                    channel.position(0)
                    val mime = C2paMimeTypes.sniff(head.array().copyOf(head.position()), declaredMime)
                        ?: return@use C2paReadResult.UnsupportedFormat
                    reader.read(channel, mime)
                }
            }.getOrElse { C2paReadResult.Failed(it.toString()) }
        }
        val result = withTimeoutOrNull(timeoutMs) { work.await() }
            ?: return unavailable(R.string.credentials_timeout)
        return toSignal(result)
    }

    /** Maps a read result to the signal, per the table in the class doc. */
    fun toSignal(result: C2paReadResult): DetectionSignal = when (result) {
        is C2paReadResult.Report -> fromReport(result.json)
        C2paReadResult.NoManifest -> unavailable(R.string.credentials_none)
        C2paReadResult.RemoteManifest -> unavailable(R.string.credentials_remote)
        C2paReadResult.UnsupportedFormat -> unavailable(R.string.credentials_unsupported_format)
        C2paReadResult.UnsupportedDevice -> unavailable(R.string.credentials_unsupported_device)
        is C2paReadResult.Failed -> {
            Log.w(TAG, "Content Credentials read failed: ${result.reason}")
            unavailable(R.string.credentials_failed)
        }
    }

    private fun fromReport(json: String): DetectionSignal {
        val findings = runCatching { C2paManifestInterpreter.interpret(json) }.getOrElse {
            Log.w(TAG, "Content Credentials report not understood", it)
            return unavailable(R.string.credentials_failed)
        }
        val signer = findings.signer ?: texts.get(R.string.credentials_unknown_signer)
        val toolEvidence = findings.aiTool?.let { texts.get(R.string.credentials_evidence_tool, it) }
        return when (findings.trust) {
            Trust.TRUSTED -> when (findings.source) {
                DeclaredSource.GENERATIVE_AI -> available(AI_SCORE, texts.get(R.string.credentials_trusted_ai, signer), toolEvidence)
                DeclaredSource.CAPTURE -> available(CAPTURE_SCORE, texts.get(R.string.credentials_trusted_capture, signer), null)
                DeclaredSource.NOT_DECLARED -> unavailable(R.string.credentials_trusted_undeclared, signer)
            }
            Trust.UNTRUSTED -> if (findings.source == DeclaredSource.GENERATIVE_AI) {
                unavailable(R.string.credentials_untrusted_ai, signer).copy(evidence = toolEvidence)
            } else {
                unavailable(R.string.credentials_untrusted, signer)
            }
            Trust.INVALID -> unavailable(R.string.credentials_invalid).copy(
                evidence = findings.failureCodes.takeIf { it.isNotEmpty() }
                    ?.let { texts.get(R.string.credentials_evidence_failures, it.joinToString(", ")) },
            )
        }
    }

    private fun available(score: Float, description: String, evidence: String?) = DetectionSignal(
        type = signalType,
        availability = SignalAvailability.AVAILABLE,
        score = score,
        confidence = 1f,
        description = description,
        evidence = evidence,
    )

    private fun unavailable(id: Int, vararg args: Any) = DetectionSignal.unavailable(signalType, texts.get(id, *args))

    companion object {
        private const val TAG = "C2PAProvider"

        /** A trusted manifest declaring generative AI: as close to certain as this app gets. */
        const val AI_SCORE = 0.98f

        /** A trusted manifest declaring a camera capture and no generative AI anywhere. */
        const val CAPTURE_SCORE = 0.05f

        const val TIMEOUT_MS = 15_000L

        /** Trust settings are read from assets once, on first use. */
        private fun cachedSettings(context: Context): () -> String {
            val settings by lazy { C2paTrustSettings.fromAssets(context.assets) }
            return { settings }
        }
    }
}
