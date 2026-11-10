package com.genned.app.ui.history

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId

class DayGroupsTest {

    private val zone = ZoneId.of("Asia/Kolkata")
    private val today = LocalDate.of(2026, 11, 10)

    private fun at(date: LocalDate, hour: Int, minute: Int = 0): Long =
        LocalDateTime.of(date, java.time.LocalTime.of(hour, minute)).atZone(zone).toInstant().toEpochMilli()

    @Test
    fun `days are split at local midnight`() {
        assertThat(dayGroupOf(at(today, 0, 0), today, zone)).isEqualTo(DayGroup.TODAY)
        assertThat(dayGroupOf(at(today, 23, 59), today, zone)).isEqualTo(DayGroup.TODAY)
        assertThat(dayGroupOf(at(today.minusDays(1), 23, 59), today, zone)).isEqualTo(DayGroup.YESTERDAY)
        assertThat(dayGroupOf(at(today.minusDays(2), 12), today, zone)).isEqualTo(DayGroup.EARLIER)
        assertThat(dayGroupOf(at(today.plusDays(1), 1), today, zone)).isEqualTo(DayGroup.TODAY)
    }

    @Test
    fun `groups keep newest-first order and skip empty days`() {
        val newest = at(today, 18)
        val older = at(today, 9)
        val lastWeek = at(today.minusDays(6), 9)
        val groups = groupByDay(listOf(newest, older, lastWeek), today, zone) { it }
        assertThat(groups).containsExactly(
            DayGroup.TODAY to listOf(newest, older),
            DayGroup.EARLIER to listOf(lastWeek),
        ).inOrder()
        assertThat(groupByDay(emptyList<Long>(), today, zone) { it }).isEmpty()
    }
}
