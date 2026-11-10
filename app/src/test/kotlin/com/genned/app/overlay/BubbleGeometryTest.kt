package com.genned.app.overlay

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class BubbleGeometryTest {

    @Test
    fun `a position inside the screen is kept`() {
        assertThat(BubbleGeometry.clampToScreen(100, 700, 150, 1080, 2400)).isEqualTo(100 to 700)
    }

    @Test
    fun `a position saved on a bigger or rotated screen is pulled back on screen`() {
        assertThat(BubbleGeometry.clampToScreen(2300, 900, 150, 1080, 2400)).isEqualTo(930 to 900)
        assertThat(BubbleGeometry.clampToScreen(-40, 5000, 150, 1080, 2400)).isEqualTo(0 to 2250)
    }

    @Test
    fun `a bubble larger than the screen stays at the origin`() {
        assertThat(BubbleGeometry.clampToScreen(10, 10, 500, 300, 300)).isEqualTo(0 to 0)
    }

    @Test
    fun `dropping on the dismiss target is a distance check`() {
        assertThat(BubbleGeometry.isOverDismissTarget(540f, 2200f, 540f, 2200f, 120f)).isTrue()
        assertThat(BubbleGeometry.isOverDismissTarget(600f, 2280f, 540f, 2200f, 120f)).isTrue() // 100 px away
        assertThat(BubbleGeometry.isOverDismissTarget(540f, 2000f, 540f, 2200f, 120f)).isFalse()
    }
}
