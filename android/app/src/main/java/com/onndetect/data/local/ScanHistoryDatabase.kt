package com.onndetect.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

// SQLCipher hardening (net.zetetic:android-database-sqlcipher) can be layered
// in later by supplying an openHelperFactory; the schema is unaffected.
@Database(entities = [ScanHistoryEntity::class], version = 1, exportSchema = false)
abstract class ScanHistoryDatabase : RoomDatabase() {
    abstract fun scanHistoryDao(): ScanHistoryDao

    companion object {
        @Volatile
        private var instance: ScanHistoryDatabase? = null

        fun create(context: Context): ScanHistoryDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context,
                    ScanHistoryDatabase::class.java,
                    "onn_db"
                ).build().also { instance = it }
            }
    }
}
