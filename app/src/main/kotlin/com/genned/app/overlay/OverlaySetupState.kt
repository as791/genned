package com.genned.app.overlay

/** The three things Android needs before the bubble can run, in the order they are asked for. */
enum class OverlaySetupStep {
    /** "Display over other apps", so the bubble can float over Instagram or WhatsApp. */
    DRAW_OVER_APPS,

    /** "Usage access", so the bubble knows when one of those apps is open (the app's name only). */
    USAGE_ACCESS,

    /** The screen-capture consent Android asks for each time the bubble is turned on. */
    SCREEN_CAPTURE,
}

/** What is already granted, and so which step comes next. */
data class OverlaySetupState(
    val canDrawOverApps: Boolean,
    val hasUsageAccess: Boolean,
    val running: Boolean,
) {
    /** The step the user has to do now; null once the bubble is running. */
    val nextStep: OverlaySetupStep? = when {
        running -> null
        !canDrawOverApps -> OverlaySetupStep.DRAW_OVER_APPS
        !hasUsageAccess -> OverlaySetupStep.USAGE_ACCESS
        else -> OverlaySetupStep.SCREEN_CAPTURE
    }

    fun isDone(step: OverlaySetupStep): Boolean = when (step) {
        OverlaySetupStep.DRAW_OVER_APPS -> canDrawOverApps
        OverlaySetupStep.USAGE_ACCESS -> hasUsageAccess
        OverlaySetupStep.SCREEN_CAPTURE -> running
    }
}
