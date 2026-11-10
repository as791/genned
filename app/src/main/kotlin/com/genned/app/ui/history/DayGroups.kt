package com.genned.app.ui.history

import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

/** The date headers in History. */
enum class DayGroup { TODAY, YESTERDAY, EARLIER }

fun dayGroupOf(timestampMillis: Long, today: LocalDate, zone: ZoneId): DayGroup {
    val day = Instant.ofEpochMilli(timestampMillis).atZone(zone).toLocalDate()
    return when {
        !day.isBefore(today) -> DayGroup.TODAY // also anything dated in the future (clock changes)
        day == today.minusDays(1) -> DayGroup.YESTERDAY
        else -> DayGroup.EARLIER
    }
}

/** Splits a newest-first list into its day groups, keeping the order; empty groups are left out. */
fun <T> groupByDay(
    items: List<T>,
    today: LocalDate,
    zone: ZoneId,
    timestampOf: (T) -> Long,
): List<Pair<DayGroup, List<T>>> =
    items.groupBy { dayGroupOf(timestampOf(it), today, zone) }
        .toSortedMap()
        .map { (group, entries) -> group to entries }
