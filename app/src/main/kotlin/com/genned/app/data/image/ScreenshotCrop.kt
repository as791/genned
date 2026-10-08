package com.genned.app.data.image

import com.genned.app.overlay.CaptureCrop
import com.genned.domain.model.CropRect
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Finds the picture inside a shared phone screenshot, so the classifier sees the photo
 * rather than the app around it (status bar, app bar, captions, buttons). The overlay
 * already crops its live captures ([CaptureCrop]); this does the same for screenshot files.
 *
 * The picture is the tallest band of rows that are *not* mostly the app's flat background
 * colour, and within it the widest run of such columns. When that isn't clear (no band, or
 * hardly any UI), it falls back to the centred square between the system bars, as the
 * overlay does.
 *
 * Pure logic on a small ARGB grid (about [GRID_WIDTH] wide), no Android dependency. The
 * Python mirror the benchmarks use is tools/screenshot_crop.py; keep the two in sync.
 */
object ScreenshotCrop {
    const val GRID_WIDTH = 108
    private const val BACKGROUND_TOLERANCE = 24
    private const val UI_FRACTION = 0.6
    private const val GAP_FRACTION = 0.02
    private const val MIN_BAND_FRACTION = 0.25
    private const val MIN_UI_FRACTION = 0.15
    private const val FALLBACK_TOP = 0.04
    private const val FALLBACK_BOTTOM = 0.05

    /** The grid size to downsample a [width] x [height] image to before [classifierRegion]. */
    fun gridSize(width: Int, height: Int): Pair<Int, Int> {
        val gridWidth = min(GRID_WIDTH, width)
        return gridWidth to max(1, (height.toLong() * gridWidth / width.toDouble()).roundToInt())
    }

    /**
     * The part of a [width] x [height] screenshot to classify, given its downsampled
     * [grid] (ARGB, [gridWidth] x [gridHeight], see [gridSize]).
     */
    fun classifierRegion(grid: IntArray, gridWidth: Int, gridHeight: Int, width: Int, height: Int): CropRect {
        require(grid.size == gridWidth * gridHeight) { "Grid is ${grid.size}, expected ${gridWidth * gridHeight}" }
        val band = pictureBand(grid, gridWidth, gridHeight)
        if (band == null) {
            val square = CaptureCrop.contentSquare(
                width, height, (FALLBACK_TOP * height).roundToInt(), (FALLBACK_BOTTOM * height).roundToInt(),
            )
            return CropRect(square[0], square[1], square[2], square[2])
        }
        val (gl, gt, gr, gb) = band
        val left = (gl.toLong() * width / gridWidth).toInt()
        val top = (gt.toLong() * height / gridHeight).toInt()
        val right = min(width.toLong(), ceilDiv(gr.toLong() * width, gridWidth.toLong())).toInt()
        val bottom = min(height.toLong(), ceilDiv(gb.toLong() * height, gridHeight.toLong())).toInt()
        return CropRect(left, top, right - left, bottom - top)
    }

    /** `[left, top, right, bottom]` of the picture in grid cells, or null if not confident. */
    internal fun pictureBand(grid: IntArray, gridWidth: Int, gridHeight: Int): IntArray? {
        val background = background(grid)
        val near = BooleanArray(grid.size) { distance(grid[it], background) <= BACKGROUND_TOLERANCE }

        var uiRowCount = 0
        val contentRows = BooleanArray(gridHeight) { y ->
            var bg = 0
            for (x in 0 until gridWidth) if (near[y * gridWidth + x]) bg++
            val ui = bg >= UI_FRACTION * gridWidth
            if (ui) uiRowCount++
            !ui
        }
        val (top, bottom) = longestRun(contentRows, max(1, (GAP_FRACTION * gridHeight).roundToInt()))
        val bandHeight = bottom - top
        if (bandHeight < MIN_BAND_FRACTION * gridHeight || uiRowCount < MIN_UI_FRACTION * gridHeight ||
            bandHeight >= gridHeight
        ) return null

        val contentColumns = BooleanArray(gridWidth) { x ->
            var bg = 0
            for (y in top until bottom) if (near[y * gridWidth + x]) bg++
            bg < UI_FRACTION * bandHeight
        }
        var (left, right) = longestRun(contentColumns, max(1, (GAP_FRACTION * gridWidth).roundToInt()))
        if (right - left < MIN_BAND_FRACTION * gridWidth) {
            left = 0
            right = gridWidth
        }
        return intArrayOf(left, top, right, bottom)
    }

    /** `[start, end)` of the longest run of true values, bridging false gaps up to [maxGap]. */
    internal fun longestRun(isContent: BooleanArray, maxGap: Int): Pair<Int, Int> {
        var best = 0 to 0
        var start = -1
        var lastTrue = -1
        for (i in isContent.indices) {
            if (!isContent[i]) continue
            if (start < 0 || i - lastTrue - 1 > maxGap) start = i
            lastTrue = i
            if (lastTrue + 1 - start > best.second - best.first) best = start to lastTrue + 1
        }
        return best
    }

    /** Mean RGB of the most common 4-bit-per-channel colour bucket. */
    private fun background(grid: IntArray): IntArray {
        val counts = IntArray(4096)
        for (pixel in grid) counts[bucket(pixel)]++
        val mode = counts.indices.maxByOrNull { counts[it] } ?: 0
        var r = 0L
        var g = 0L
        var b = 0L
        var n = 0
        for (pixel in grid) {
            if (bucket(pixel) != mode) continue
            r += (pixel shr 16) and 0xFF
            g += (pixel shr 8) and 0xFF
            b += pixel and 0xFF
            n++
        }
        return intArrayOf((r / n).toInt(), (g / n).toInt(), (b / n).toInt())
    }

    private fun bucket(pixel: Int): Int =
        (((pixel shr 20) and 0xF) shl 8) or (((pixel shr 12) and 0xF) shl 4) or ((pixel shr 4) and 0xF)

    private fun distance(pixel: Int, background: IntArray): Int = max(
        abs(((pixel shr 16) and 0xFF) - background[0]),
        max(abs(((pixel shr 8) and 0xFF) - background[1]), abs((pixel and 0xFF) - background[2])),
    )

    private fun ceilDiv(a: Long, b: Long): Long = (a + b - 1) / b

    /**
     * Whether a shared image is a screenshot worth cropping. Never for a camera photo (it
     * has camera EXIF). Otherwise when its file name says so (Android names them
     * "Screenshot_…") or its size is exactly this phone's screen, either way round.
     */
    fun isLikelyScreenshot(
        displayName: String?,
        hasCameraExif: Boolean,
        width: Int,
        height: Int,
        screenWidth: Int?,
        screenHeight: Int?,
    ): Boolean {
        if (hasCameraExif) return false
        if (displayName?.contains("screenshot", ignoreCase = true) == true) return true
        if (screenWidth == null || screenHeight == null) return false
        return (width == screenWidth && height == screenHeight) || (width == screenHeight && height == screenWidth)
    }
}
