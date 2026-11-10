package com.genned.app.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/**
 * The "✕" circle shown at the bottom of the screen while the bubble is being dragged.
 * Dropping the bubble on it turns the overlay off. Plain Canvas for the same reason as
 * [BubbleView]. It never takes touches; the bubble's own drag decides the outcome.
 */
class DismissTargetView(context: Context) : View(context) {

    /** True while the dragged bubble is over the target, so it visibly "arms". */
    var armed: Boolean = false
        set(value) {
            if (field != value) {
                field = value
                invalidate()
            }
        }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val crossPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cx = width / 2f
        val cy = height / 2f
        val radius = minOf(width, height) / 2f * (if (armed) 1f else 0.82f)
        fillPaint.color = if (armed) Color.parseColor("#D8342A") else Color.parseColor("#CC1B1F3B")
        canvas.drawCircle(cx, cy, radius, fillPaint)
        val arm = radius * 0.32f
        crossPaint.strokeWidth = radius * 0.12f
        canvas.drawLine(cx - arm, cy - arm, cx + arm, cy + arm, crossPaint)
        canvas.drawLine(cx - arm, cy + arm, cx + arm, cy - arm, crossPaint)
    }
}
