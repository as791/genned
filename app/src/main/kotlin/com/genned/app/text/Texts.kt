package com.genned.app.text

import android.content.res.Resources
import androidx.annotation.PluralsRes
import androidx.annotation.StringRes
import com.genned.app.R
import com.genned.domain.evidence.EvidenceTexts

/**
 * User-facing text produced outside the UI (signal descriptions, the shared result card),
 * read from string resources so it can be translated. Analysis code takes this instead of a
 * Context.
 *
 * Results are saved with the text they were analyzed with (see SignalDto), so history keeps
 * the language it was created in.
 */
interface Texts {
    fun get(@StringRes id: Int, vararg args: Any): String
    fun quantity(@PluralsRes id: Int, count: Int, vararg args: Any): String
}

class ResourceTexts(private val resources: Resources) : Texts {
    override fun get(id: Int, vararg args: Any): String =
        if (args.isEmpty()) resources.getString(id) else resources.getString(id, *args)

    override fun quantity(id: Int, count: Int, vararg args: Any): String =
        resources.getQuantityString(id, count, *args)
}

/**
 * [EvidenceTexts] from string resources, for the domain's [com.genned.domain.evidence.EvidenceEngine].
 * Read on each use, so a language change applies to the next analysis.
 */
class ResourceEvidenceTexts(private val texts: Texts) : EvidenceTexts {
    override val probabilistic get() = texts.get(R.string.limitation_probabilistic)
    override val classifierDidNotRun get() = texts.get(R.string.limitation_classifier_did_not_run)
    override val noVerifiedCredentials get() = texts.get(R.string.limitation_no_verified_credentials)
    override val noGeneratorMetadata get() = texts.get(R.string.limitation_no_generator_metadata)
}
