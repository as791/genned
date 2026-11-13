package com.genned.domain.evidence

/**
 * Transparent, fixed weights for [EvidenceEngine]'s weighted-average blend.
 *
 * The classification thresholds are set from benchmark data (see [HIGH_THRESHOLD]).
 * The blend weights are still a documented starting point, see
 * internal-docs/ARCHITECTURE.md "Recalibrating the evidence weights". They live in one
 * place, isolated from providers and UI, so they can be replaced from evaluation data.
 *
 * All weights below except [CLASSIFIER_WEIGHT] are *maximums*: [EvidenceEngine] scales
 * them by the signal's own score, so a corroborating signal that found nothing (score
 * 0) contributes zero weight instead of diluting the blend toward "human".
 */
object EvidenceWeights {
    /** The visual classifier is the strongest probabilistic signal we have. */
    const val CLASSIFIER_WEIGHT = 0.70f

    /** A matched generator signature (e.g. an SD "parameters" chunk) is strong but not proof. */
    const val GENERATOR_METADATA_WEIGHT = 0.22f

    /** A positive watermark match would be strong corroborating evidence. */
    const val WATERMARK_WEIGHT = 0.20f

    /** Generic EXIF anomalies (missing/stripped/inconsistent fields) are weak on their own. */
    const val EXIF_ANOMALY_WEIGHT = 0.06f

    /**
     * aiLikelihood >= this -> [com.genned.domain.model.Classification.HIGH].
     *
     * Chosen from data by tools/calibrate.py: the lowest threshold at which at most 5% of
     * real images show HIGH in every benchmark dataset x condition. For the shipped models
     * (photos: Ensemble build run 43339026079; video: run 36196623887) the worst case at
     * 0.90 is 3.2% of real photos and 2.0% of real videos. After everyday edits it is 5.3%
     * of real photos (Robustness eval run 43370045934); 0.85 would show up to 7.0%. With
     * 0.70, up to 39% of real images in the harder dataset showed HIGH (first model, run
     * 36185656205).
     */
    const val HIGH_THRESHOLD = 0.90f

    /**
     * aiLikelihood < this -> [com.genned.domain.model.Classification.LOW] for video (and the
     * [EvidenceEngine] default); between LOW and HIGH is UNCERTAIN. Chosen the same way:
     * the highest threshold at which at most 10% of AI content shows LOW. For video
     * (Ensemble build run 36196623887) that is 0.15 (9.0%); 0.20 would show 14% of AI
     * videos as LOW.
     */
    const val LOW_THRESHOLD = 0.15f

    /**
     * The LOW cut-off for photos and screen captures (see [LOW_THRESHOLD] for video). It is
     * chosen separately from the video one and happens to be equal today.
     * At 0.15, at most 7.3% of AI photos show LOW on unedited benchmark images (Robustness
     * eval run 43370045934, report run 43379280573), inside the 10% rule; under everyday
     * edits (grain, sharpening, screenshot-and-repost) it is up to 12.7%. Real phone photos
     * read UNCERTAIN 34% of the time. At 0.20 they would read UNCERTAIN 29% of the time,
     * but 11.3% of unedited and 14.3% of edited AI photos would show LOW. Chosen by the
     * maintainer with the accuracy round 4 models; it was 0.20 from v0.2.2 to v0.4.0.
     */
    const val PHOTO_LOW_THRESHOLD = 0.15f

    /** Likelihood reported when no signal produced usable evidence at all. */
    const val NO_EVIDENCE_LIKELIHOOD = 0.5f
}
