package com.genned.app.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import com.google.common.truth.Truth.assertWithMessage
import org.junit.Test

/**
 * The verdict is coloured text, so it has to stay readable: at least 4.5:1 (WCAG AA for
 * normal text) against the app's own light and dark backgrounds.
 */
class ContrastTest {

    private fun contrast(a: Color, b: Color): Float {
        val (light, dark) = listOf(a.luminance(), b.luminance()).sortedDescending()
        return (light + 0.05f) / (dark + 0.05f)
    }

    @Test
    fun `verdict colours are readable on light backgrounds`() {
        for ((name, text) in listOf("high" to HighRed, "uncertain" to UncertainAmber, "low" to LowGreen)) {
            for (background in listOf(NeutralBackgroundLight, NeutralSurfaceLight)) {
                assertWithMessage("$name on light").that(contrast(text, background)).isAtLeast(4.5f)
            }
        }
    }

    @Test
    fun `verdict colours are readable on dark backgrounds`() {
        // Dark theme uses the pale "container" tones as text (see ClassificationColors).
        for ((name, text) in listOf("high" to HighRedContainer, "uncertain" to UncertainAmberContainer, "low" to LowGreenContainer)) {
            for (background in listOf(NeutralBackgroundDark, NeutralSurfaceDark)) {
                assertWithMessage("$name on dark").that(contrast(text, background)).isAtLeast(4.5f)
            }
        }
    }
}
