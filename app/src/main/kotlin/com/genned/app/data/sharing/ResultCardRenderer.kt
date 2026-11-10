package com.genned.app.data.sharing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.text.StaticLayout
import android.text.TextPaint
import androidx.annotation.StringRes
import com.genned.app.R
import com.genned.app.text.displayPercent
import com.genned.domain.model.AnalysisResult
import com.genned.domain.model.Classification
import com.genned.domain.model.SignalAvailability
import com.genned.domain.model.SignalType
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Renders a shareable result-card PNG entirely from the aggregated result — never
 * the user's original image, to avoid unnecessary privacy/copyright exposure once
 * it's shared onward (see internal-docs/PRIVACY.md).
 */
class ResultCardRenderer(private val context: Context) {

    suspend fun render(result: AnalysisResult): File = withContext(Dispatchers.Default) {
        val width = 1080
        val height = 1350
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        val backgroundColor = Color.parseColor("#121318")
        val accentColor = classificationColor(result.classification)
        val onBackground = Color.parseColor("#F3F3FA")
        val mutedColor = Color.parseColor("#A6A6B8")
        val margin = 72f

        canvas.drawColor(backgroundColor)
        var y = 96f

        val headerPaint = TextPaint().apply {
            color = mutedColor
            textSize = 34f
            isAntiAlias = true
        }
        canvas.drawText("GENNED", margin, y, headerPaint)
        y += 110f

        val percentPaint = TextPaint().apply {
            color = onBackground
            textSize = 140f
            isAntiAlias = true
            isFakeBoldText = true
        }
        canvas.drawText("${displayPercent(result.aiLikelihood)}%", margin, y, percentPaint)
        y += 70f

        val labelPaint = TextPaint().apply {
            color = accentColor
            textSize = 44f
            isAntiAlias = true
            isFakeBoldText = true
        }
        canvas.drawText(classificationLabel(result.classification), margin, y, labelPaint)
        y += 90f

        canvas.drawLine(
            margin,
            y,
            width - margin,
            y,
            Paint().apply { color = Color.parseColor("#2B2C36"); strokeWidth = 2f },
        )
        y += 70f

        val rowLabelPaint = TextPaint().apply { color = mutedColor; textSize = 32f; isAntiAlias = true }
        val rowValuePaint = TextPaint().apply {
            color = onBackground
            textSize = 32f
            isAntiAlias = true
            isFakeBoldText = true
        }

        for ((label, value) in buildRows(result)) {
            canvas.drawText(label, margin, y, rowLabelPaint)
            val valueWidth = rowValuePaint.measureText(value)
            canvas.drawText(value, width - margin - valueWidth, y, rowValuePaint)
            y += 64f
        }

        y += 50f
        val disclaimerText = context.getString(R.string.card_disclaimer)
        val disclaimerPaint = TextPaint().apply {
            color = mutedColor
            textSize = 28f
            isAntiAlias = true
        }
        val disclaimerLayout = StaticLayout.Builder.obtain(
            disclaimerText,
            0,
            disclaimerText.length,
            disclaimerPaint,
            (width - 2 * margin).toInt(),
        ).setLineSpacing(4f, 1.1f).build()
        canvas.save()
        canvas.translate(margin, y)
        disclaimerLayout.draw(canvas)
        canvas.restore()

        val footerPaint = TextPaint().apply { color = mutedColor; textSize = 28f; isAntiAlias = true }
        canvas.drawText(context.getString(R.string.card_footer), margin, height - 80f, footerPaint)

        val outFile = File(sharedCardsDir(context), "result_${UUID.randomUUID()}.png")
        FileOutputStream(outFile).use { out -> bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) }
        bitmap.recycle()
        outFile
    }

    private fun buildRows(result: AnalysisResult): List<Pair<String, String>> {
        fun describe(type: SignalType, @StringRes label: Int): Pair<String, String> {
            val signal = result.signals.firstOrNull { it.type == type }
            val available = signal != null && signal.availability == SignalAvailability.AVAILABLE
            val score = signal?.score
            val value = when {
                !available || score == null -> context.getString(R.string.card_value_not_available)
                type == SignalType.AI_CLASSIFIER -> context.getString(R.string.card_value_percent, displayPercent(score))
                score > 0f -> context.getString(R.string.signal_summary_found)
                else -> context.getString(R.string.signal_summary_not_found)
            }
            return context.getString(label) to value
        }
        return listOf(
            describe(SignalType.AI_CLASSIFIER, R.string.card_row_visual),
            describe(SignalType.GENERATOR_METADATA, R.string.card_row_generator_metadata),
            describe(SignalType.CONTENT_CREDENTIALS, R.string.card_row_content_credentials),
        )
    }

    private fun classificationLabel(classification: Classification): String = context.getString(
        when (classification) {
            Classification.HIGH -> R.string.result_classification_high
            Classification.UNCERTAIN -> R.string.result_classification_uncertain
            Classification.LOW -> R.string.result_classification_low
        },
    )

    private fun classificationColor(classification: Classification): Int = when (classification) {
        Classification.HIGH -> Color.parseColor("#FF6B5E")
        Classification.UNCERTAIN -> Color.parseColor("#F0C24B")
        Classification.LOW -> Color.parseColor("#5FD98A")
    }

    companion object {
        fun sharedCardsDir(context: Context): File = File(context.cacheDir, "shared").apply { mkdirs() }
    }
}
