package com.genned.app.data.detection.provenance

import android.os.Build
import androidx.annotation.RequiresApi
import org.contentauth.c2pa.C2PAContext
import org.contentauth.c2pa.C2PAError
import org.contentauth.c2pa.C2PASettings
import org.contentauth.c2pa.CallbackStream
import org.contentauth.c2pa.Reader
import org.contentauth.c2pa.SeekMode
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/** What reading a file's Content Credentials produced. */
sealed interface C2paReadResult {
    /** A manifest was found and validated; [json] is the reader's report (see [C2paManifestInterpreter]). */
    data class Report(val json: String) : C2paReadResult

    /** The file has no embedded Content Credentials. */
    data object NoManifest : C2paReadResult

    /** The file points to Content Credentials stored online; Genned has no network access and does not fetch them. */
    data object RemoteManifest : C2paReadResult

    /** The C2PA reader does not handle this file type. */
    data object UnsupportedFormat : C2paReadResult

    /** Android older than 9, or no native C2PA library for this CPU (release builds leave out x86). */
    data object UnsupportedDevice : C2paReadResult

    /** Anything else; [reason] is for the debug log only. */
    data class Failed(val reason: String) : C2paReadResult
}

/** Reads Content Credentials from an open file. Behind an interface so the provider can be tested without native code. */
fun interface C2paReader {
    /** Blocking; call off the main thread. [mimeType] is the file's real type (see [C2paMimeTypes.sniff]). */
    fun read(channel: FileChannel, mimeType: String): C2paReadResult
}

/**
 * [C2paReader] backed by contentauth/c2pa-android (the C2PA Rust SDK over JNI).
 *
 * Verification runs fully on the device against [settingsJson] (trust anchors and settings,
 * see [C2paTrustSettings]). Remote manifests are never fetched: the settings turn that off
 * and the app has no INTERNET permission anyway.
 */
class NativeC2paReader(private val settingsJson: () -> String) : C2paReader {

    override fun read(channel: FileChannel, mimeType: String): C2paReadResult {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return C2paReadResult.UnsupportedDevice
        return try {
            Api28.read(channel, mimeType, settingsJson())
        } catch (e: UnsatisfiedLinkError) {
            C2paReadResult.UnsupportedDevice
        } catch (e: NoClassDefFoundError) {
            // The class initializer failed to load the native library earlier.
            C2paReadResult.UnsupportedDevice
        }
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private object Api28 {
        val supportedTypes: Set<String> by lazy { Reader.supportedMimeTypes().map { it.lowercase() }.toSet() }

        fun read(channel: FileChannel, mimeType: String, settingsJson: String): C2paReadResult {
            if (mimeType.lowercase() !in supportedTypes) return C2paReadResult.UnsupportedFormat
            return try {
                C2PASettings.create().use { settings ->
                    settings.updateFromString(settingsJson, "json")
                    C2PAContext.fromSettings(settings).use { context ->
                        channelStream(channel).use { stream ->
                            Reader.fromContext(context).withStream(mimeType, stream).use { reader ->
                                C2paReadResult.Report(reader.json())
                            }
                        }
                    }
                }
            } catch (e: C2PAError) {
                classify(e.message.orEmpty())
            }
        }

        /** A C2PA stream over [channel] (read-only; writes are refused). */
        private fun channelStream(channel: FileChannel) = CallbackStream(
            { buffer, length ->
                val read = channel.read(ByteBuffer.wrap(buffer, 0, minOf(length, buffer.size)))
                if (read < 0) 0 else read
            },
            { offset, mode ->
                val base = when (mode) {
                    SeekMode.START -> 0L
                    SeekMode.CURRENT -> channel.position()
                    SeekMode.END -> channel.size()
                }
                channel.position(base + offset)
                channel.position()
            },
            { _, _ -> -1 },
            { 0 },
        )
    }

    companion object {
        /** Maps the reader's error text to a result; c2pa-rs reports "no manifest" as an error. */
        fun classify(message: String): C2paReadResult {
            val m = message.lowercase()
            return when {
                "remote" in m -> C2paReadResult.RemoteManifest
                NO_MANIFEST.any { it in m } -> C2paReadResult.NoManifest
                "unsupported" in m || "notsupported" in m -> C2paReadResult.UnsupportedFormat
                else -> C2paReadResult.Failed(message)
            }
        }

        private val NO_MANIFEST = listOf(
            "jumbfnotfound", "manifestnotfound", "jumbf not found", "no jumbf", "could not find jumbf",
            "no claim", "claim missing", "claimmissing", "manifest not found", "no manifest",
        )
    }
}

/** File type detection from the first bytes, so a wrong or missing declared type doesn't skip the check. */
object C2paMimeTypes {
    fun sniff(head: ByteArray, declared: String?): String? {
        fun at(i: Int, s: String) = head.size >= i + s.length && s.indices.all { head[i + it] == s[it].code.toByte() }
        return when {
            head.size >= 3 && head[0] == 0xFF.toByte() && head[1] == 0xD8.toByte() && head[2] == 0xFF.toByte() -> "image/jpeg"
            head.size >= 4 && head[0] == 0x89.toByte() && at(1, "PNG") -> "image/png"
            at(0, "RIFF") && at(8, "WEBP") -> "image/webp"
            at(0, "GIF8") -> "image/gif"
            at(0, "II*\u0000") || at(0, "MM\u0000*") -> "image/tiff"
            at(4, "ftyp") -> when {
                at(8, "avif") || at(8, "avis") -> "image/avif"
                at(8, "heic") || at(8, "heix") || at(8, "mif1") || at(8, "msf1") -> "image/heic"
                at(8, "qt  ") -> "video/quicktime"
                else -> "video/mp4"
            }
            else -> declared?.substringBefore(';')?.trim()?.lowercase()?.takeIf { '/' in it && '*' !in it }
        }
    }
}
