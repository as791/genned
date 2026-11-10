package com.genned.app.ui.result

import com.genned.app.data.storage.MediaKind
import com.genned.domain.evidence.EvidenceWeights
import com.genned.domain.model.DetectionSignal
import com.genned.domain.model.SignalAvailability
import com.genned.domain.model.SignalType
import com.genned.domain.model.verifiedCredentials

/** How the result screen lays out the checks: [shown] as cards in this order, [notAvailable] folded into one row. */
data class ArrangedSignals(val shown: List<DetectionSignal>, val notAvailable: List<DetectionSignal>)

/**
 * Orders the evidence so what decided the result comes first, and folds away checks that
 * had nothing to report. Display only: it changes no score.
 *
 * Shown, in order:
 * 1. verified Content Credentials (they decide the result on their own);
 * 2. an AI tool's name found in the file (strong evidence);
 * 3. the AI detector, always, even if it could not run, because it is the main check;
 * 4. Content Credentials that were found but not counted ([DetectionSignal.notable]);
 * 5. every other check that produced a reading.
 *
 * Everything else (unavailable or failed, nothing notable) goes to [ArrangedSignals.notAvailable].
 */
/** Why a result is "Uncertain", so a score like 87% next to "Uncertain" doesn't read as a contradiction. */
sealed interface UncertainHint {
    /** The score leans AI but is under the cut-off for saying so ([cutOffPercent]). */
    data class BelowHighCutOff(val cutOffPercent: Int) : UncertainHint

    /** The score leans away from AI but is not under the cut-off for saying so ([cutOffPercent]). */
    data class AboveLowCutOff(val cutOffPercent: Int) : UncertainHint

    /** An older check whose kind (and so its low cut-off) isn't known. */
    data object Generic : UncertainHint
}

object ResultPresentation {

    /**
     * Explains an Uncertain result with the cut-off it missed. Display only: the cut-offs are
     * the ones EvidenceEngine applied (EvidenceWeights), read here, never changed.
     */
    fun uncertainHint(aiLikelihood: Float, mediaKind: MediaKind?): UncertainHint = when {
        aiLikelihood >= 0.5f -> UncertainHint.BelowHighCutOff(percent(EvidenceWeights.HIGH_THRESHOLD))
        mediaKind == MediaKind.VIDEO -> UncertainHint.AboveLowCutOff(percent(EvidenceWeights.LOW_THRESHOLD))
        mediaKind != null -> UncertainHint.AboveLowCutOff(percent(EvidenceWeights.PHOTO_LOW_THRESHOLD))
        else -> UncertainHint.Generic
    }

    private fun percent(threshold: Float): Int = Math.round(threshold * 100)

    fun arrange(signals: List<DetectionSignal>): ArrangedSignals {
        val verified = signals.verifiedCredentials()
        val (shown, folded) = signals.partition { signal ->
            signal === verified ||
                signal.type == SignalType.AI_CLASSIFIER ||
                signal.notable ||
                signal.availability == SignalAvailability.AVAILABLE
        }
        return ArrangedSignals(
            shown = shown.sortedBy { rank(it, isVerified = it === verified) },
            notAvailable = folded,
        )
    }

    private fun rank(signal: DetectionSignal, isVerified: Boolean): Int = when {
        isVerified -> 0
        signal.type == SignalType.GENERATOR_METADATA && (signal.score ?: 0f) > 0f -> 1
        signal.type == SignalType.AI_CLASSIFIER -> 2
        signal.type == SignalType.CONTENT_CREDENTIALS -> 3
        else -> 4
    }
}
