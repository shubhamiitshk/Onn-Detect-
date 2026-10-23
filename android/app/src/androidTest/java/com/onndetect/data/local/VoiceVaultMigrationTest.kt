package com.onndetect.data.local

import androidx.sqlite.db.SupportSQLiteDatabase
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.onndetect.data.repository.VoiceVaultRepository
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * F3 instrumentation: the v1 -> v2 vault migration and the sealing of
 * pre-v2 plaintext rows, against real SQLite and the real AES-GCM path
 * (the in-memory crypto double keeps the test deterministic; the
 * AndroidKeyStore binding is exercised by the app flow tests).
 */
@RunWith(AndroidJUnit4::class)
class VoiceVaultMigrationTest {

    private fun newDb(): SupportSQLiteDatabase {
        val helper = FrameworkSQLiteOpenHelperFactory().create(
            androidx.sqlite.db.SupportSQLiteOpenHelper.Configuration.builder(
                androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
            ).name(null).callback(object : androidx.sqlite.db.SupportSQLiteOpenHelper.Callback(1) {
                override fun onCreate(db: SupportSQLiteDatabase) = Unit
                override fun onUpgrade(db: SupportSQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit
            }).build()
        )
        return helper.writableDatabase
    }

    @Test
    fun migration1to2CarriesPlaintextAndAddsSealedColumns() {
        val db = newDb()
        try {
            // v1 schema exactly as shipped in release 66fce33.
            db.execSQL(
                """
                CREATE TABLE voice_vault_profiles (
                  id TEXT NOT NULL PRIMARY KEY,
                  contact_name TEXT NOT NULL,
                  relationship TEXT NOT NULL,
                  enrolled_at_ms INTEGER NOT NULL,
                  sample_duration_ms INTEGER NOT NULL,
                  embedding_csv TEXT NOT NULL,
                  sample_sha256 TEXT
                )
                """.trimIndent()
            )
            db.execSQL(
                "INSERT INTO voice_vault_profiles VALUES ('legacy-1','Mom','Mother',123,9000,'0.1,0.2,0.3',NULL)"
            )

            VoiceVaultDatabase.MIGRATION_1_2.migrate(db)

            val columns = mutableSetOf<String>()
            db.query("PRAGMA table_info(voice_vault_profiles)").use { cursor ->
                while (cursor.moveToNext()) columns += cursor.getString(1)
            }
            for (expected in listOf(
                "id", "contact_name", "relationship", "enrolled_at_ms", "sample_duration_ms",
                "embedding_csv", "sample_sha256", "encrypted_embedding", "embedding_iv",
                "spoof_score", "embedding_version",
            )) assertTrue("missing column $expected", expected in columns)

            db.query("SELECT embedding_csv FROM voice_vault_profiles WHERE id='legacy-1'").use {
                assertTrue(it.moveToFirst())
                assertEquals("0.1,0.2,0.3", it.getString(0)) // carried over unchanged
            }
        } finally {
            db.close()
        }
    }

    @Test
    fun ensureLegacyRowsSealedSealsPlaintextAndBlanksTheColumn() = kotlinx.coroutines.runBlocking {
        val context = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val db = androidx.room.Room.inMemoryDatabaseBuilder(
            context, VoiceVaultDatabase::class.java
        ).allowMainThreadQueries().build()
        try {
            val dao = db.voiceProfileDao()
            dao.insert(
                VoiceProfileEntity(
                    id = "legacy-2",
                    contactName = "Dad",
                    relationship = "Father",
                    enrolledAtMs = 456L,
                    sampleDurationMs = 11_000L,
                    embeddingCsv = "0.400000,-0.500000",
                )
            )
            val repository = VoiceVaultRepository(dao, InMemoryVoiceVaultCrypto())
            val sealedCount = repository.ensureLegacyRowsSealed()

            assertEquals(1, sealedCount)
            val row = dao.findById("legacy-2")
            assertNotNull(row)
            assertTrue("row must now be sealed", row!!.isSealed)
            assertFalse("plaintext CSV must be gone", row.hasLegacyPlaintext)

            // The sealed blob must unseal back to the original vector.
            val embedding = repository.unsealedEmbedding(row)
            assertNotNull("sealed row must unseal", embedding)
            assertEquals(2, embedding!!.size)
            assertEquals(0.4f, embedding[0], 1e-6f)
            assertEquals(-0.5f, embedding[1], 1e-6f)

            // Idempotent: a second pass finds nothing left to seal.
            assertEquals(0, repository.ensureLegacyRowsSealed())
        } finally {
            db.close()
        }
    }
}
