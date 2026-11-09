package com.genned.app.data.detection.metadata

import androidx.exifinterface.media.ExifInterface
import com.genned.app.R
import com.genned.app.text.Texts
import com.genned.domain.model.AnalysisInput
import com.genned.domain.model.DetectionSignal
import com.genned.domain.model.SignalAvailability
import com.genned.domain.model.SignalType
import com.genned.domain.provider.DetectionProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * A narrow, honest EXIF signal: is standard *camera capture* metadata (make, model,
 * exposure) present? Its presence is real evidence a camera was involved somewhere in
 * this file's history (weak, since it doesn't rule out later AI editing); its
 * *absence* is deliberately treated as near-zero evidence rather than a sign of AI,
 * since social apps and screenshots strip EXIF constantly. This is why its weight in
 * EvidenceWeights is the smallest of all signals.
 *
 * Generator-tool string matching is a separate, stronger signal — see
 * [KnownGeneratorMetadataProvider].
 */
class ExifMetadataProvider(private val texts: Texts) : DetectionProvider {
    override val signalType: SignalType = SignalType.EXIF_METADATA

    override suspend fun analyze(image: AnalysisInput): DetectionSignal = withContext(Dispatchers.IO) {
        val file = File(image.originalFilePath)
        val exif = try {
            ExifInterface(file)
        } catch (e: Exception) {
            return@withContext DetectionSignal.error(signalType, texts.get(R.string.exif_unreadable))
        }

        val cameraFields = listOf(
            ExifInterface.TAG_MAKE,
            ExifInterface.TAG_MODEL,
            ExifInterface.TAG_F_NUMBER,
            ExifInterface.TAG_EXPOSURE_TIME,
            ExifInterface.TAG_ISO_SPEED_RATINGS,
            ExifInterface.TAG_DATETIME_ORIGINAL,
        )
        val presentFields = cameraFields.filter { !exif.getAttribute(it).isNullOrBlank() }

        if (presentFields.isNotEmpty()) {
            DetectionSignal(
                type = signalType,
                availability = SignalAvailability.AVAILABLE,
                score = 0f,
                confidence = 0.5f,
                description = texts.quantity(R.plurals.exif_camera_metadata_present, presentFields.size, presentFields.size),
                evidence = presentFields.joinToString(", "),
            )
        } else {
            DetectionSignal(
                type = signalType,
                availability = SignalAvailability.AVAILABLE,
                score = 0.3f,
                confidence = 0.3f,
                description = texts.get(R.string.exif_no_camera_metadata),
            )
        }
    }
}
