package com.genned.app.overlay

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class OverlaySetupStateTest {

    @Test
    fun `steps are asked for in order, one at a time`() {
        assertThat(OverlaySetupState(canDrawOverApps = false, hasUsageAccess = false, running = false).nextStep)
            .isEqualTo(OverlaySetupStep.DRAW_OVER_APPS)
        assertThat(OverlaySetupState(canDrawOverApps = true, hasUsageAccess = false, running = false).nextStep)
            .isEqualTo(OverlaySetupStep.USAGE_ACCESS)
        assertThat(OverlaySetupState(canDrawOverApps = true, hasUsageAccess = true, running = false).nextStep)
            .isEqualTo(OverlaySetupStep.SCREEN_CAPTURE)
    }

    @Test
    fun `a permission granted out of order does not skip the missing one`() {
        val state = OverlaySetupState(canDrawOverApps = false, hasUsageAccess = true, running = false)
        assertThat(state.nextStep).isEqualTo(OverlaySetupStep.DRAW_OVER_APPS)
        assertThat(state.isDone(OverlaySetupStep.USAGE_ACCESS)).isTrue()
        assertThat(state.isDone(OverlaySetupStep.DRAW_OVER_APPS)).isFalse()
    }

    @Test
    fun `nothing is left to do while the bubble is running`() {
        val state = OverlaySetupState(canDrawOverApps = true, hasUsageAccess = true, running = true)
        assertThat(state.nextStep).isNull()
        assertThat(OverlaySetupStep.entries.all(state::isDone)).isTrue()
    }

    @Test
    fun `screen capture is asked again every time the bubble was turned off`() {
        val state = OverlaySetupState(canDrawOverApps = true, hasUsageAccess = true, running = false)
        assertThat(state.isDone(OverlaySetupStep.SCREEN_CAPTURE)).isFalse()
    }
}
