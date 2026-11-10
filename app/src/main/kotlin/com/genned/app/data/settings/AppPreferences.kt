package com.genned.app.data.settings

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Small on-device settings: where the bubble was left and whether the home screen's bubble
 * card was dismissed. Stored in the app's private preferences; nothing here is about what
 * you check.
 */
class AppPreferences(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("genned_prefs", Context.MODE_PRIVATE)

    private val _overlayCardDismissed = MutableStateFlow(prefs.getBoolean(KEY_OVERLAY_CARD_DISMISSED, false))

    /** The home screen's "check Reels without leaving Instagram" card was closed by the user. */
    val overlayCardDismissed: StateFlow<Boolean> = _overlayCardDismissed.asStateFlow()

    fun dismissOverlayCard() {
        prefs.edit().putBoolean(KEY_OVERLAY_CARD_DISMISSED, true).apply()
        _overlayCardDismissed.value = true
    }

    /** The bubble's last position in screen pixels, or null if it was never moved. */
    var bubblePosition: Pair<Int, Int>?
        get() = if (prefs.contains(KEY_BUBBLE_X) && prefs.contains(KEY_BUBBLE_Y)) {
            prefs.getInt(KEY_BUBBLE_X, 0) to prefs.getInt(KEY_BUBBLE_Y, 0)
        } else {
            null
        }
        set(value) {
            prefs.edit().apply {
                if (value == null) {
                    remove(KEY_BUBBLE_X)
                    remove(KEY_BUBBLE_Y)
                } else {
                    putInt(KEY_BUBBLE_X, value.first)
                    putInt(KEY_BUBBLE_Y, value.second)
                }
            }.apply()
        }

    private companion object {
        const val KEY_OVERLAY_CARD_DISMISSED = "overlay_card_dismissed"
        const val KEY_BUBBLE_X = "bubble_x"
        const val KEY_BUBBLE_Y = "bubble_y"
    }
}
