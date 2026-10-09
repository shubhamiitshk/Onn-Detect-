package com.onndetect.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(entities = [ScanHistoryEntity::class], version = 3, exportSchema = true)
abstract class ScanHistoryDatabase : RoomDatabase() {
    abstract fun scanHistoryDao(): ScanHistoryDao
    companion object {
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE scan_history ADD COLUMN signals_json TEXT")
            }
        }
        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE scan_history ADD COLUMN status TEXT NOT NULL DEFAULT 'INCONCLUSIVE'")
                db.execSQL("ALTER TABLE scan_history ADD COLUMN verdict TEXT NOT NULL DEFAULT 'INCONCLUSIVE'")
                db.execSQL("ALTER TABLE scan_history ADD COLUMN reason TEXT")
                db.execSQL("ALTER TABLE scan_history ADD COLUMN model_validated INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE scan_history ADD COLUMN evidence_json TEXT")
                db.execSQL("UPDATE scan_history SET reason = 'Legacy assessment: model validation and coverage were not recorded.'")
                db.execSQL("CREATE INDEX IF NOT EXISTS index_scan_history_timestamp ON scan_history(timestamp)")
            }
        }
        @Volatile private var instance: ScanHistoryDatabase? = null
        fun create(context: Context): ScanHistoryDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(context.applicationContext, ScanHistoryDatabase::class.java, "onn_db")
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3).build().also { instance = it }
        }
    }
}
