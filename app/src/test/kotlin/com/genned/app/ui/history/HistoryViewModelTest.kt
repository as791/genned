package com.genned.app.ui.history

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.genned.app.data.storage.AnalysisDatabase
import com.genned.app.data.storage.HistoryRepository
import com.genned.app.data.storage.ThumbnailStore
import com.genned.domain.model.AnalysisResult
import com.genned.domain.model.Classification
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
class HistoryViewModelTest {

    private lateinit var database: AnalysisDatabase
    private lateinit var repository: HistoryRepository

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        val context = ApplicationProvider.getApplicationContext<Context>()
        database = Room.inMemoryDatabaseBuilder(context, AnalysisDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        repository = HistoryRepository(database.analysisDao(), ThumbnailStore(context))
    }

    @After
    fun tearDown() {
        database.close()
        Dispatchers.resetMain()
    }

    private suspend fun saveOne(): Long = repository.save(
        AnalysisResult(aiLikelihood = 0.4f, classification = Classification.UNCERTAIN, signals = emptyList(),
            limitations = emptyList()),
        File.createTempFile("preview", ".jpg").apply { writeBytes(ByteArray(16) { 1 }) },
    )

    @Test
    fun `delete hides the item and undo brings it back without touching the database`() = runTest {
        val id = saveOne()
        val viewModel = HistoryViewModel(repository, commitScope = backgroundScope)
        viewModel.entries.first { it.size == 1 }

        viewModel.delete(id)
        viewModel.entries.first { it.isEmpty() }
        assertThat(repository.observeAll().first()).hasSize(1)

        viewModel.undoDelete(id)
        assertThat(viewModel.entries.first { it.size == 1 }.single().id).isEqualTo(id)
    }

    @Test
    fun `a committed delete removes the item from the database`() = runTest {
        val id = saveOne()
        val viewModel = HistoryViewModel(repository, commitScope = backgroundScope)

        viewModel.delete(id)
        viewModel.commitDelete(id)

        repository.observeAll().first { it.isEmpty() }
    }

    @Test
    fun `commitDelete does nothing unless that item was deleted first`() = runTest {
        val id = saveOne()
        val viewModel = HistoryViewModel(repository, commitScope = backgroundScope)

        viewModel.commitDelete(id)

        assertThat(repository.observeAll().first()).hasSize(1)
    }
}
