package com.genned.app.ui.result

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
object ResultPresentation {

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
