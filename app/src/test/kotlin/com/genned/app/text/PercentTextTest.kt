package com.genned.app.text

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class PercentTextTest {

    @Test
    fun `rounds to the nearest whole percent`() {
        assertThat(displayPercent(0.694f)).isEqualTo(69)
        assertThat(displayPercent(0.696f)).isEqualTo(70)
        assertThat(displayPercent(0.5f)).isEqualTo(50)
    }

    @Test
    fun `never shows a flat 0 or 100`() {
        assertThat(displayPercent(0f)).isEqualTo(1)
        assertThat(displayPercent(0.001f)).isEqualTo(1)
        assertThat(displayPercent(0.999f)).isEqualTo(99)
        assertThat(displayPercent(1f)).isEqualTo(99)
    }
}
