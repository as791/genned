package com.genned.app.ui.history

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.genned.app.data.storage.HistoryEntry
import com.genned.app.data.storage.HistoryRepository
import com.genned.app.ui.appContainer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Deleting a single item is undoable: [delete] only hides it, and the screen's snackbar
 * then calls [undoDelete] or [commitDelete]. Deletes still pending when the screen goes
 * away are committed in [onCleared], so leaving History doesn't silently undo them.
 * Clear-all is not undoable and keeps its confirmation dialog.
 */
class HistoryViewModel(
    private val historyRepository: HistoryRepository,
    // Commits must outlive viewModelScope (cancelled in onCleared).
    private val commitScope: CoroutineScope = processScope,
) : ViewModel() {

    private val pendingDeletes = MutableStateFlow<Set<Long>>(emptySet())

    val entries: StateFlow<List<HistoryEntry>> =
        combine(historyRepository.observeAll(), pendingDeletes) { all, pending -> all.filterNot { it.id in pending } }
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    /** Hides the item right away; it's only deleted by [commitDelete] (or when the screen closes). */
    fun delete(id: Long) {
        pendingDeletes.update { it + id }
    }

    fun undoDelete(id: Long) {
        pendingDeletes.update { it - id }
    }

    fun commitDelete(id: Long) {
        if (id !in pendingDeletes.value) return
        commitScope.launch {
            historyRepository.delete(id)
            pendingDeletes.update { it - id }
        }
    }

    fun clearAll() {
        viewModelScope.launch { historyRepository.clearAll() }
    }

    override fun onCleared() {
        pendingDeletes.value.forEach(::commitDelete)
    }

    companion object {
        private val processScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

        val Factory = viewModelFactory {
            initializer { HistoryViewModel(appContainer().historyRepository) }
        }
    }
}
