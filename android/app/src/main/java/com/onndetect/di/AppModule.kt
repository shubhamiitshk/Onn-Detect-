package com.onndetect.di

import android.app.ActivityManager
import android.content.Context
import com.onndetect.data.local.ScanHistoryDao
import com.onndetect.data.local.ScanHistoryDatabase
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.domain.models.DeviceTier
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideContext(@ApplicationContext context: Context): Context = context

    @Provides
    @Singleton
    fun provideReceiptSigner(@ApplicationContext context: Context): com.onndetect.detection.provenance.ReceiptSigner =
        com.onndetect.detection.provenance.StrongBoxReceiptSigner(context)

    @Provides
    @Singleton
    fun provideDeviceTier(@ApplicationContext context: Context): DeviceTier {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryClassMb = am.memoryClass
        val totalRamGb = ActivityManager.MemoryInfo()
            .also(am::getMemoryInfo)
            .totalMem / (1024.0 * 1024 * 1024)
        return when {
            totalRamGb >= 6 && memoryClassMb >= 192 -> DeviceTier.HIGH_END
            totalRamGb >= 4 -> DeviceTier.MID_RANGE
            else -> DeviceTier.LOW_END
        }
    }

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): ScanHistoryDatabase =
        ScanHistoryDatabase.create(context)

    @Provides
    fun provideScanHistoryDao(db: ScanHistoryDatabase): ScanHistoryDao = db.scanHistoryDao()

    @Provides
    @Singleton
    fun provideScanHistoryRepository(dao: ScanHistoryDao) =
        ScanHistoryRepository(dao)

    @Provides
    @Singleton
    fun provideVoiceVaultDatabase(@ApplicationContext context: Context): com.onndetect.data.local.VoiceVaultDatabase =
        com.onndetect.data.local.VoiceVaultDatabase.create(context)

    @Provides
    fun provideVoiceProfileDao(db: com.onndetect.data.local.VoiceVaultDatabase): com.onndetect.data.local.VoiceProfileDao =
        db.voiceProfileDao()
}
