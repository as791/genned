package com.genned.app.overlay

import kotlin.math.hypot

/** Where the bubble may sit and when a drag counts as "dropped on the ✕". Pure, so it is unit-tested. */
object BubbleGeometry {

    /** Keeps the whole bubble on screen, e.g. after the saved position came from another screen size or rotation. */
    fun clampToScreen(x: Int, y: Int, bubbleSize: Int, screenWidth: Int, screenHeight: Int): Pair<Int, Int> =
        x.coerceIn(0, maxOf(0, screenWidth - bubbleSize)) to y.coerceIn(0, maxOf(0, screenHeight - bubbleSize))

    /** True when the finger is within [radius] of the dismiss target's centre. */
    fun isOverDismissTarget(fingerX: Float, fingerY: Float, targetCenterX: Float, targetCenterY: Float, radius: Float): Boolean =
        hypot(fingerX - targetCenterX, fingerY - targetCenterY) <= radius
}
