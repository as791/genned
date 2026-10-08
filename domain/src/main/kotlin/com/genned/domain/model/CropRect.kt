package com.genned.domain.model

/** A pixel rectangle in an image: [left], [top] and its [width] x [height]. */
data class CropRect(val left: Int, val top: Int, val width: Int, val height: Int) {
    init {
        require(left >= 0 && top >= 0 && width > 0 && height > 0) { "Invalid crop $this" }
    }
}
