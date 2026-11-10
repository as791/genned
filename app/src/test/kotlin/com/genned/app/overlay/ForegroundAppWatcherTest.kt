package com.genned.app.overlay

import android.app.usage.UsageEvents
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Test

/** Pure logic, no Android dependency — runs as a plain JUnit test. */
class ForegroundAppWatcherTest {

    private fun resumed(pkg: String) = UsageEvents.Event.MOVE_TO_FOREGROUND to pkg
    private fun paused(pkg: String) = UsageEvents.Event.MOVE_TO_BACKGROUND to pkg

    @Test
    fun `an app that keeps producing no new events stays foreground`() {
        // The regression seen on-device: Instagram opened, then the user sat on one
        // Reel for longer than the old 10s lookback window with no new usage-stats
        // event - the old code re-derived "current" from that empty window on every
        // poll and returned null, hiding the bubble mid-Reel even though Instagram
        // was never backgrounded. An empty event batch must leave the carried-forward
        // package untouched.
        val afterOpen = ForegroundAppWatcher.applyEvents(listOf(resumed("com.instagram.android")), start = null)
        assertThat(afterOpen).isEqualTo("com.instagram.android")

        val afterManyQuietPolls = ForegroundAppWatcher.applyEvents(emptyList(), start = afterOpen)
        assertThat(afterManyQuietPolls).isEqualTo("com.instagram.android")
    }

    @Test
    fun `backgrounding the current app clears it`() {
        val current = ForegroundAppWatcher.applyEvents(listOf(paused("com.instagram.android")), start = "com.instagram.android")
        assertThat(current).isNull()
    }

    @Test
    fun `a background event for a different package is ignored`() {
        // Stale/out-of-order MOVE_TO_BACKGROUND for a package that isn't the current
        // one (e.g. the previous app) must not clear the real current package.
        val current = ForegroundAppWatcher.applyEvents(listOf(paused("com.whatsapp")), start = "com.instagram.android")
        assertThat(current).isEqualTo("com.instagram.android")
    }

    @Test
    fun `switching apps replaces the current package`() {
        val current = ForegroundAppWatcher.applyEvents(
            listOf(paused("com.instagram.android"), resumed("com.sec.android.app.launcher")),
            start = "com.instagram.android",
        )
        assertThat(current).isEqualTo("com.sec.android.app.launcher")
    }

    @Test
    fun `no events and no prior state stays null`() {
        assertThat(ForegroundAppWatcher.applyEvents(emptyList(), start = null)).isNull()
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun `no usage queries while the screen is off, and nothing is missed after`() = runTest {
        val screenOn = MutableStateFlow(true)
        val queries = mutableListOf<Pair<Long, Long>>()
        var clock = 100_000L
        var pending = listOf(resumed("com.instagram.android"))
        val emitted = mutableListOf<String?>()
        backgroundScope.launch {
            ForegroundAppWatcher.watch(screenOn, pollIntervalMs = 1_500L, now = { clock }) { start, end ->
                queries += start to end
                pending.also { pending = emptyList() }
            }.collect { emitted += it }
        }
        runCurrent()
        assertThat(queries).hasSize(1)
        assertThat(emitted).containsExactly("com.instagram.android")

        // Screen goes off: a minute passes with no queries at all.
        screenOn.value = false
        clock += 1_500L
        advanceTimeBy(1_500L)
        runCurrent()
        val queriesWhenOff = queries.size
        clock += 60_000L
        advanceTimeBy(60_000L)
        runCurrent()
        assertThat(queries).hasSize(queriesWhenOff)

        // Screen on: the next query covers the whole dark period, so an app switch made
        // then (e.g. unlocked straight into WhatsApp) is seen.
        pending = listOf(paused("com.instagram.android"), resumed("com.whatsapp"))
        screenOn.value = true
        runCurrent()
        assertThat(queries).hasSize(queriesWhenOff + 1)
        assertThat(queries.last().first).isEqualTo(queries[queriesWhenOff - 1].second)
        assertThat(emitted.last()).isEqualTo("com.whatsapp")
    }
}
