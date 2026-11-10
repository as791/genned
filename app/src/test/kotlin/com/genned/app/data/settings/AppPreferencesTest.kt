package com.genned.app.data.settings

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class AppPreferencesTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()

    @Test
    fun `bubble position is remembered across instances`() {
        assertThat(AppPreferences(context).bubblePosition).isNull()
        AppPreferences(context).bubblePosition = 120 to 640
        assertThat(AppPreferences(context).bubblePosition).isEqualTo(120 to 640)
        AppPreferences(context).bubblePosition = null
        assertThat(AppPreferences(context).bubblePosition).isNull()
    }

    @Test
    fun `dismissing the home card sticks`() {
        val prefs = AppPreferences(context)
        assertThat(prefs.overlayCardDismissed.value).isFalse()
        prefs.dismissOverlayCard()
        assertThat(prefs.overlayCardDismissed.value).isTrue()
        assertThat(AppPreferences(context).overlayCardDismissed.value).isTrue()
    }
}
