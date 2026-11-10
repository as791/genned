package com.genned.app.data.storage

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/** Updating the app must keep the history saved by earlier versions. */
@RunWith(RobolectricTestRunner::class)
class MigrationTest {

    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val name = "migration-test.db"

    @After
    fun tearDown() {
        context.deleteDatabase(name)
    }

    @Test
    fun `history saved by version 1 survives and gets no media kind`() = runTest {
        // A version 1 database exactly as that version created it (app/schemas/.../1.json).
        context.openOrCreateDatabase(name, Context.MODE_PRIVATE, null).use { db ->
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS `analyses` (`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, " +
                    "`timestampMillis` INTEGER NOT NULL, `thumbnailPath` TEXT, `aiLikelihood` REAL NOT NULL, " +
                    "`classification` TEXT NOT NULL, `signalsJson` TEXT NOT NULL, `limitationsJson` TEXT NOT NULL)",
            )
            db.execSQL("CREATE TABLE IF NOT EXISTS room_master_table (id INTEGER PRIMARY KEY,identity_hash TEXT)")
            db.execSQL(
                "INSERT OR REPLACE INTO room_master_table (id,identity_hash) " +
                    "VALUES(42, '7df74e108223a05cdd77e2c11faf1b58')",
            )
            db.execSQL(
                "INSERT INTO analyses (timestampMillis, thumbnailPath, aiLikelihood, classification, signalsJson, limitationsJson) " +
                    "VALUES (1234, '/thumbs/a.jpg', 0.83, 'HIGH', '[]', '[\"old limitation\"]')",
            )
            db.version = 1
        }

        val database = Room.databaseBuilder(context, AnalysisDatabase::class.java, name)
            .addMigrations(AnalysisDatabase.MIGRATION_1_2)
            .allowMainThreadQueries()
            .build()
        try {
            val dao = database.analysisDao()
            val old = dao.getById(1)!!
            assertThat(old.timestampMillis).isEqualTo(1234)
            assertThat(old.thumbnailPath).isEqualTo("/thumbs/a.jpg")
            assertThat(old.aiLikelihood).isWithin(1e-6f).of(0.83f)
            assertThat(old.classification).isEqualTo("HIGH")
            assertThat(old.limitationsJson).isEqualTo("[\"old limitation\"]")
            assertThat(old.mediaKind).isNull()

            val newId = dao.insert(old.copy(id = 0, mediaKind = MediaKind.VIDEO.name))
            assertThat(dao.getById(newId)!!.mediaKind).isEqualTo("VIDEO")
        } finally {
            database.close()
        }
    }

    @Test
    fun `unknown or missing media kind names read as null`() {
        assertThat(MediaKind.fromName(null)).isNull()
        assertThat(MediaKind.fromName("HOLOGRAM")).isNull()
        assertThat(MediaKind.fromName("SCREEN")).isEqualTo(MediaKind.SCREEN)
    }
}
