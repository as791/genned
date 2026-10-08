package com.genned.domain.model

/**
 * A normalized image ready for detection providers. The domain layer has no Android
 * dependency, so instead of a `Bitmap`/`Uri` this points at local files the app layer
 * prepared (see `ImageLoader` in the `app` module), stripped of the original
 * share-sheet URI.
 *
 * Two files are provided deliberately:
 * - [originalFilePath] is a byte-exact copy of the source as received — its EXIF
 *   fields, PNG text chunks, and any embedded C2PA manifest are all still intact.
 *   Metadata and provenance providers must read from *this* file.
 * - [normalizedFilePath] is orientation-corrected, downscaled, and re-encoded as
 *   JPEG for fast, memory-bounded preview and classifier input. Re-encoding strips
 *   metadata, so this file must never be used for metadata/provenance inspection.
 *
 * [classifierCrop], when set, is the part of the normalized image the visual classifier
 * should look at (the picture inside a shared screenshot); everything else still sees
 * the whole image.
 */
data class AnalysisInput(
    val originalFilePath: String,
    val normalizedFilePath: String,
    val originalMimeType: String?,
    val widthPx: Int,
    val heightPx: Int,
    val fileSizeBytes: Long,
    val classifierCrop: CropRect? = null,
)
