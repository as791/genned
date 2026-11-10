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
            "AI detection is an estimate. It can flag real pictures and miss AI ones, especially from the newest AI tools."
        override val classifierDidNotRun =
            "The AI detector didn't run on this one, so the estimate above rests on file details only and is less reliable."
        override val noVerifiedCredentials =
            "No verified signed record (Content Credentials) was found, so there's no proof of where this came from."
        override val noGeneratorMetadata =
            "No AI tool's name was found in the file. That doesn't mean it's real: social and editing apps usually remove it."
    }
}
