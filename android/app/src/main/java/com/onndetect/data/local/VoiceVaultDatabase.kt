package com.onndetect.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [VoiceProfileEntity::class], version = 2, exportSchema = false)
abstract class VoiceVaultDatabase : RoomDatabase() {

    abstract fun voiceProfileDao(): VoiceProfileDao

    companion object {
        @Volatile
        private var instance: VoiceVaultDatabase? = null

        fun create(context: Context): VoiceVaultDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                context.applicationContext,
                VoiceVaultDatabase::class.java,
                "voice_vault.db",
            )
                .fallbackToDestructiveMigration(dropAllTables = true)
                .build()
                .also { instance = it }
        }
    }
}
