package com.genned.app.text

import kotlin.math.roundToInt

/**
 * The whole-number percentage shown for an AI likelihood or probability, the same on every
 * screen, the bubble and the shared card. Rounded, and kept within 1-99: detection is an
 * estimate, so the app never shows a flat 0% or 100%.
 */
fun displayPercent(likelihood: Float): Int = (likelihood * 100).roundToInt().coerceIn(1, 99)
