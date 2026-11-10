package com.genned.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.genned.app.R
import com.genned.app.data.storage.HistoryEntry
import com.genned.app.data.storage.MediaKind
import com.genned.app.text.displayPercent
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

// The phone's own date and time style and language.
private val timeFormatter = DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM, FormatStyle.SHORT)

/** One saved check. The whole row opens it. */
@Composable
fun HistoryRow(
    entry: HistoryEntry,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Box {
            FileThumbnail(path = entry.thumbnailPath, contentDescription = null)
            if (entry.mediaKind == MediaKind.VIDEO) {
                Icon(
                    imageVector = Icons.Default.PlayArrow,
                    contentDescription = null, // the row's text says "Video"
                    tint = Color.White,
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(4.dp)
                        .size(18.dp)
                        .background(Color.Black.copy(alpha = 0.6f), CircleShape),
                )
            }
        }
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = classificationLabel(entry.classification),
                style = MaterialTheme.typography.titleMedium,
                color = classificationColor(entry.classification),
            )
            val kind = when (entry.mediaKind) {
                MediaKind.VIDEO -> stringResource(R.string.history_row_video) + " · "
                MediaKind.SCREEN -> stringResource(R.string.history_row_screen) + " · "
                MediaKind.PHOTO, null -> ""
            }
            Text(
                text = kind + "${displayPercent(entry.aiLikelihood)}% · " +
                    timeFormatter.format(
                        Instant.ofEpochMilli(entry.timestampMillis).atZone(ZoneId.systemDefault()),
                    ),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
