package com.genned.app.data.image

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import kotlin.random.Random

class ScreenshotCropTest {

    private val gridWidth = 108
    private val gridHeight = 240

    /**
     * A grid shaped like a social-app screenshot: flat [background] UI with sparse "text"
     * marks, and a noisy picture in rows [pictureTop, pictureBottom).
     */
    private fun screenshotGrid(
        background: Int,
        ink: Int,
        pictureTop: Int,
        pictureBottom: Int,
        pictureLeft: Int = 0,
        pictureRight: Int = gridWidth,
        seed: Int = 1,
    ): IntArray {
        val random = Random(seed)
        return IntArray(gridWidth * gridHeight) { i ->
            val x = i % gridWidth
            val y = i / gridWidth
            when {
                y in pictureTop until pictureBottom && x in pictureLeft until pictureRight ->
                    rgb(random.nextInt(30, 226), random.nextInt(30, 226), random.nextInt(30, 226))
                y % 12 == 4 && x in 4 until 70 && random.nextFloat() < 0.6f -> ink // text lines
                else -> background
            }
        }
    }

    private fun rgb(r: Int, g: Int, b: Int) = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    private val white = rgb(255, 255, 255)
    private val black = rgb(0, 0, 0)

    @Test
    fun `finds the picture band in a light screenshot`() {
        val grid = screenshotGrid(white, rgb(20, 20, 20), pictureTop = 40, pictureBottom = 175)
        val band = ScreenshotCrop.pictureBand(grid, gridWidth, gridHeight)!!
        assertThat(band.toList()).isEqualTo(listOf(0, 40, gridWidth, 175))
    }

    @Test
    fun `finds the picture band in a dark screenshot`() {
        val grid = screenshotGrid(black, rgb(235, 235, 235), pictureTop = 30, pictureBottom = 150)
        val band = ScreenshotCrop.pictureBand(grid, gridWidth, gridHeight)!!
        assertThat(band.toList()).isEqualTo(listOf(0, 30, gridWidth, 150))
    }

    @Test
    fun `trims side margins around a narrower picture`() {
        val grid = screenshotGrid(white, rgb(20, 20, 20), 50, 160, pictureLeft = 20, pictureRight = 90)
        val band = ScreenshotCrop.pictureBand(grid, gridWidth, gridHeight)!!
        assertThat(band.toList()).isEqualTo(listOf(20, 50, 90, 160))
    }

    @Test
    fun `scales the band back to full-size pixels`() {
        val grid = screenshotGrid(white, rgb(20, 20, 20), pictureTop = 40, pictureBottom = 175)
        val crop = ScreenshotCrop.classifierRegion(grid, gridWidth, gridHeight, 1080, 2400)
        assertThat(crop.left).isEqualTo(0)
        assertThat(crop.top).isEqualTo(400)
        assertThat(crop.width).isEqualTo(1080)
        assertThat(crop.top + crop.height).isEqualTo(1750)
    }

    @Test
    fun `a plain photo with no UI falls back to the centred square`() {
        val grid = screenshotGrid(white, white, pictureTop = 0, pictureBottom = gridHeight)
        assertThat(ScreenshotCrop.pictureBand(grid, gridWidth, gridHeight)).isNull()
        val crop = ScreenshotCrop.classifierRegion(grid, gridWidth, gridHeight, 1080, 2400)
        assertThat(crop.width).isEqualTo(1080)
        assertThat(crop.height).isEqualTo(1080)
        assertThat(crop.top).isEqualTo(96 + (2400 - 96 - 120 - 1080) / 2)
    }

    @Test
    fun `a sliver of picture is not trusted`() {
        val grid = screenshotGrid(white, rgb(20, 20, 20), pictureTop = 100, pictureBottom = 130)
        assertThat(ScreenshotCrop.pictureBand(grid, gridWidth, gridHeight)).isNull()
    }

    @Test
    fun `small gaps inside the picture are bridged`() {
        val grid = screenshotGrid(white, rgb(20, 20, 20), pictureTop = 40, pictureBottom = 175)
        // A white stripe 3 rows tall across the picture (e.g. a white object), under 2% of 240 = 5 rows.
        for (y in 100 until 103) for (x in 0 until gridWidth) grid[y * gridWidth + x] = white
        val band = ScreenshotCrop.pictureBand(grid, gridWidth, gridHeight)!!
        assertThat(band.toList()).isEqualTo(listOf(0, 40, gridWidth, 175))
    }

    @Test
    fun `longest run bridges short gaps only`() {
        val runs = booleanArrayOf(true, true, false, true, true, false, false, false, true)
        assertThat(ScreenshotCrop.longestRun(runs, maxGap = 1)).isEqualTo(0 to 5)
        assertThat(ScreenshotCrop.longestRun(runs, maxGap = 0)).isEqualTo(0 to 2)
        assertThat(ScreenshotCrop.longestRun(BooleanArray(3), maxGap = 1)).isEqualTo(0 to 0)
    }

    @Test
    fun `screenshot detection never crops a camera photo`() {
        assertThat(ScreenshotCrop.isLikelyScreenshot("Screenshot_2026.png", true, 1080, 2400, 1080, 2400)).isFalse()
        assertThat(ScreenshotCrop.isLikelyScreenshot("Screenshot_2026.png", false, 900, 900, null, null)).isTrue()
        assertThat(ScreenshotCrop.isLikelyScreenshot("IMG_0001.png", false, 2400, 1080, 1080, 2400)).isTrue()
        assertThat(ScreenshotCrop.isLikelyScreenshot("IMG_0001.png", false, 1080, 1350, 1080, 2400)).isFalse()
        assertThat(ScreenshotCrop.isLikelyScreenshot(null, false, 1080, 2400, null, null)).isFalse()
    }
}
