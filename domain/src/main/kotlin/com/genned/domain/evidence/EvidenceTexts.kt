package com.genned.domain.evidence

/**
 * The user-facing limitation texts [EvidenceEngine] adds to a result. The app supplies them
 * from its string resources so they can be translated; [English] keeps this module free of
 * Android and is what tests use.
 */
interface EvidenceTexts {
    /** Always shown: detection is probabilistic. */
    val probabilistic: String

    /** The visual classifier did not run, so only metadata signals count. */
    val classifierDidNotRun: String

    /** No verified Content Credentials, so no cryptographic provenance was established. */
    val noVerifiedCredentials: String

    /** Generator metadata was checked and nothing was found. */
    val noGeneratorMetadata: String

    object English : EvidenceTexts {
        override val probabilistic =
            "AI detection is probabilistic. It can produce false positives and false negatives, and newer " +
                "generation models may not be well represented in the classifier's training data."
        override val classifierDidNotRun =
            "The on-device visual classifier did not run for this image; the estimate above relies on " +
                "metadata signals only and is less reliable."
        override val noVerifiedCredentials =
            "No verified Content Credentials (C2PA) were found, so no cryptographic provenance could be " +
                "established."
        override val noGeneratorMetadata =
            "No generator metadata was found, but this does not indicate the image is authentic — social " +
                "platforms and editing tools frequently strip metadata."
    }
}
