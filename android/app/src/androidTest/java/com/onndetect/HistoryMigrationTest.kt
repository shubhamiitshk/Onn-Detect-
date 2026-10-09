package com.onndetect

import android.database.sqlite.SQLiteDatabase
import androidx.room.Room
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.data.local.ScanHistoryDatabase
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

/** Seeds actual pre-upgrade SQL, then lets Room validate the final exported schema. */
class HistoryMigrationTest {
    @Test fun preservesVersion1History() = verifyMigration(1)
    @Test fun preservesVersion2History() = verifyMigration(2)
    private fun verifyMigration(version: Int) = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = "migration-test-$version.db"
        context.deleteDatabase(name)
        try {
            context.getDatabasePath(name).parentFile!!.mkdirs()
            SQLiteDatabase.openOrCreateDatabase(context.getDatabasePath(name),null).use { old ->
                old.execSQL("CREATE TABLE scan_history (id TEXT NOT NULL PRIMARY KEY, timestamp INTEGER NOT NULL, scan_mode TEXT NOT NULL, score REAL NOT NULL, is_ai_generated INTEGER NOT NULL, confidence REAL NOT NULL, duration_ms INTEGER NOT NULL" + if(version==2) ", signals_json TEXT)" else ")")
                old.execSQL("INSERT INTO scan_history (id,timestamp,scan_mode,score,is_ai_generated,confidence,duration_ms) VALUES ('old',1,'IMAGE_ONLY',0.1,0,0.8,10)")
                old.version = version
            }
            val db = Room.databaseBuilder(context,ScanHistoryDatabase::class.java,name)
                .addMigrations(ScanHistoryDatabase.MIGRATION_1_2,ScanHistoryDatabase.MIGRATION_2_3).build()
            try {
                val entry = db.scanHistoryDao().findById("old")!!
                assertEquals("old",entry.id); assertEquals(0.1f,entry.score,0.0001f)
                assertEquals("INCONCLUSIVE",entry.verdict); assertFalse(entry.modelValidated)
                assertNotNull(entry.reason)
            } finally { db.close() }
        } finally { context.deleteDatabase(name) }
    }
}
