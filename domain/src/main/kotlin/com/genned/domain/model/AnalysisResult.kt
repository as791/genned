package com.genned.domain.model

/**
 * The aggregated output of [com.genned.domain.evidence.EvidenceEngine], ready for the
 * result screen. [aiLikelihood] and [classification] are estimates, never proof.
 */
data class AnalysisResult(
    val aiLikelihood: Float,
    val classification: Classification,
    val signals: List<DetectionSignal>,
    val limitations: List<String>,
) {
    /**
     * The verified Content Credentials that decided this result, if any. When present the
     * result is what a trusted signer declared, not a probabilistic estimate.
     */
    val verifiedCredentials: DetectionSignal? get() = signals.verifiedCredentials()
}

/** A [SignalType.CONTENT_CREDENTIALS] signal that is AVAILABLE with a score, if any. */
fun List<DetectionSignal>.verifiedCredentials(): DetectionSignal? = firstOrNull {
    it.type == SignalType.CONTENT_CREDENTIALS &&
        it.availability == SignalAvailability.AVAILABLE &&
        it.score != null
}
