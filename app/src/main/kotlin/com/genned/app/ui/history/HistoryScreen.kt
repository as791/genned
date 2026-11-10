package com.genned.app.ui.history

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.SwipeToDismissBox
import androidx.compose.material3.SwipeToDismissBoxValue
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberSwipeToDismissBoxState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.CustomAccessibilityAction
import androidx.compose.ui.semantics.customActions
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.genned.app.R
import com.genned.app.data.storage.HistoryEntry
import com.genned.app.ui.components.HistoryRow
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.ZoneId

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(
    onBack: () -> Unit,
    onOpenItem: (Long) -> Unit,
    viewModel: HistoryViewModel = viewModel(factory = HistoryViewModel.Factory),
) {
    val entries by viewModel.entries.collectAsState()
    var showClearAllConfirm by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    // Hides the row at once; the snackbar then undoes or commits (see HistoryViewModel).
    val deleteWithUndo: (Long) -> Unit = { id ->
        viewModel.delete(id)
        scope.launch {
            val result = snackbarHostState.showSnackbar(
                message = context.getString(R.string.history_deleted),
                actionLabel = context.getString(R.string.history_undo),
                duration = SnackbarDuration.Short,
            )
            if (result == SnackbarResult.ActionPerformed) viewModel.undoDelete(id) else viewModel.commitDelete(id)
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(R.string.history_title)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.cd_back))
                    }
                },
                actions = {
                    if (entries.isNotEmpty()) {
                        IconButton(onClick = { showClearAllConfirm = true }) {
                            Icon(Icons.Default.DeleteSweep, contentDescription = stringResource(R.string.history_clear_all))
                        }
                    }
                },
            )
        },
    ) { padding ->
        if (entries.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize().padding(padding), contentAlignment = Alignment.Center) {
                Text(text = stringResource(R.string.history_empty), color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            val groups = remember(entries) {
                groupByDay(entries, LocalDate.now(), ZoneId.systemDefault()) { it.timestampMillis }
            }
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(horizontal = 24.dp),
                contentPadding = PaddingValues(vertical = 8.dp),
            ) {
                groups.forEach { (group, groupEntries) ->
                    item(key = group.name) {
                        Text(
                            text = stringResource(groupLabel(group)),
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(top = 12.dp, bottom = 4.dp).semantics { heading() },
                        )
                    }
                    items(groupEntries, key = { it.id }) { entry ->
                        SwipeToDeleteRow(
                            entry = entry,
                            onOpen = { onOpenItem(entry.id) },
                            onDelete = { deleteWithUndo(entry.id) },
                        )
                    }
                }
            }
        }
    }

    if (showClearAllConfirm) {
        AlertDialog(
            onDismissRequest = { showClearAllConfirm = false },
            title = { Text(text = stringResource(R.string.history_clear_all_confirm_title)) },
            text = { Text(text = stringResource(R.string.history_clear_all_confirm_body)) },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.clearAll()
                    showClearAllConfirm = false
                }) {
                    Text(text = stringResource(R.string.confirm))
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearAllConfirm = false }) {
                    Text(text = stringResource(R.string.cancel))
                }
            },
        )
    }
}

private fun groupLabel(group: DayGroup): Int = when (group) {
    DayGroup.TODAY -> R.string.history_group_today
    DayGroup.YESTERDAY -> R.string.history_group_yesterday
    DayGroup.EARLIER -> R.string.history_group_earlier
}

/**
 * A history row that is deleted by swiping it away in either direction. Swiping is not
 * possible with a screen reader or a switch, so the same delete is also offered as an
 * accessibility action on the row.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SwipeToDeleteRow(entry: HistoryEntry, onOpen: () -> Unit, onDelete: () -> Unit) {
    val dismissState = rememberSwipeToDismissBoxState(
        confirmValueChange = { value ->
            if (value != SwipeToDismissBoxValue.Settled) onDelete()
            true
        },
    )
    val deleteLabel = stringResource(R.string.cd_delete_icon)

    SwipeToDismissBox(
        state = dismissState,
        backgroundContent = {
            val alignment = when (dismissState.dismissDirection) {
                SwipeToDismissBoxValue.StartToEnd -> Alignment.CenterStart
                else -> Alignment.CenterEnd
            }
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.errorContainer)
                    .padding(horizontal = 20.dp),
                contentAlignment = alignment,
            ) {
                Icon(
                    imageVector = Icons.Default.Delete,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onErrorContainer,
                )
            }
        },
    ) {
        HistoryRow(
            entry = entry,
            onClick = onOpen,
            modifier = Modifier
                .background(MaterialTheme.colorScheme.background)
                .semantics {
                    customActions = listOf(
                        CustomAccessibilityAction(deleteLabel) {
                            onDelete()
                            true
                        },
                    )
                },
        )
    }
}
