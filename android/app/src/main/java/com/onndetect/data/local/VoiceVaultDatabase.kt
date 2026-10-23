package com.onndetect.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(entities = [VoiceProfileEntity::class], version = 2, exportSchema = true)
abstract class VoiceVaultDatabase : RoomDatabase() {

    abstract fun voiceProfileDao(): VoiceProfileDao

    companion object {
        @Volatile
        private var instance: VoiceVaultDatabase? = null

        /**
         * v1 -> v2 (F3): rewrites the table with the sealed-embedding
         * columns. Existing plaintext CSV values are carried over unchanged;
         * [com.onndetect.data.repository.VoiceVaultRepository.ensureLegacyRowsSealed]
         * then seals them and NULLs the column, so the plaintext exists only
         * inside Room's write path during the upgrade window.
         */
        val MIGRATION_1_2: Migration = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS voice_vault_profiles_v2 (
                      id TEXT NOT NULL PRIMARY KEY,
                      contact_name TEXT NOT NULL,
                      relationship TEXT NOT NULL,
                      enrolled_at_ms INTEGER NOT NULL,
                      sample_duration_ms INTEGER NOT NULL,
                      embedding_csv TEXT,
                      sample_sha256 TEXT,
                      encrypted_embedding BLOB,
                      embedding_iv BLOB,
                      spoof_score REAL,
                      embedding_version INTEGER
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    """
                    INSERT INTO voice_vault_profiles_v2 (id, contact_name, relationship,
                      enrolled_at_ms, sample_duration_ms, embedding_csv, sample_sha256)
                    SELECT id, contact_name, relationship, enrolled_at_ms,
                      sample_duration_ms, embedding_csv, sample_sha256
                    FROM voice_vault_profiles
                    """.trimIndent()
                )
                db.execSQL("DROP TABLE voice_vault_profiles")
                db.execSQL("ALTER TABLE voice_vault_profiles_v2 RENAME TO voice_vault_profiles")
            }
        }

        fun create(context: Context): VoiceVaultDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                context.applicationContext,
                VoiceVaultDatabase::class.java,
                "voice_vault.db",
            )
                // v1 rows are PRESERVED and sealed in-place afterwards; we
                // never fall back to destructive migration for a biometric
                // vault — silently wiping enrolled family voiceprints is the
                // failure mode this builder used to hide.
                .addMigrations(MIGRATION_1_2)
                .build()
                .also { instance = it }
        }
    }
}
