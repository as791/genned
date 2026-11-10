package com.genned.app.data.storage

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(entities = [AnalysisEntity::class], version = 2, exportSchema = true)
abstract class AnalysisDatabase : RoomDatabase() {
    abstract fun analysisDao(): AnalysisDao

    companion object {
        @Volatile private var instance: AnalysisDatabase? = null

        fun getInstance(context: Context): AnalysisDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    AnalysisDatabase::class.java,
                    "genned.db",
                ).addMigrations(MIGRATION_1_2).build().also { instance = it }
            }

        /** Version 2 adds what was checked (photo, video or screen capture). Older checks keep null. */
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE analyses ADD COLUMN mediaKind TEXT")
            }
        }
    }
}
