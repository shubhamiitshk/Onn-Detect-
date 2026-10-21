package com.onndetect.di

import android.content.Context
import com.onndetect.detection.audio.AudioClassifier
import com.onndetect.detection.audio.AudioDecoder
import com.onndetect.detection.audio.LFCCExtractor
import com.onndetect.detection.fusion.XGBoostMetaClassifier
import com.onndetect.detection.temporal.VideoDecoder
import com.onndetect.detection.video.VideoForensicAnalyzer
import com.onndetect.detection.visual.FFTSpectralAnalyzer
import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import com.onndetect.detection.visual.SRMForensicAnalyzer
import com.onndetect.detection.visual.VfmProbeAnalyzer
import com.onndetect.domain.pipeline.DetectionPipeline
import com.onndetect.domain.pipeline.SignalFusion
import com.onndetect.inference.LiteRTInferenceEngine
import com.onndetect.inference.MemoryMappedModelLoader
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object PipelineModule {

    @Provides
    @Singleton
    fun provideModelLoader(@ApplicationContext context: Context) =
        MemoryMappedModelLoader(context)

    @Provides
    @Singleton
    fun provideInferenceEngine(loader: MemoryMappedModelLoader) =
        LiteRTInferenceEngine(loader)

    @Provides
    @Singleton
    fun provideSRMAnalyzer(
        @ApplicationContext context: Context,
        engine: LiteRTInferenceEngine,
    ) = SRMForensicAnalyzer(context, engine)

    @Provides
    @Singleton
    fun provideFFTAnalyzer() = FFTSpectralAnalyzer()

    @Provides
    @Singleton
    fun provideKLAnalyzer(@ApplicationContext context: Context) =
        KLCooccurrenceAnalyzer(context)

    @Provides
    @Singleton
    fun provideMetaClassifier() = XGBoostMetaClassifier()

    @Provides
    @Singleton
    fun provideVfmProbeAnalyzer(
        @ApplicationContext context: Context,
        engine: LiteRTInferenceEngine,
    ) = VfmProbeAnalyzer(context, engine)

    @Provides
    @Singleton
    fun provideSignalFusion() = SignalFusion()

    @Provides
    @Singleton
    fun provideAudioDecoder(@ApplicationContext context: Context) =
        AudioDecoder(context)

    @Provides
    @Singleton
    fun provideLFCCExtractor() = LFCCExtractor()

    @Provides
    @Singleton
    fun provideAudioClassifier(
        @ApplicationContext context: Context,
        engine: LiteRTInferenceEngine,
        decoder: AudioDecoder,
        lfcc: LFCCExtractor,
    ) = AudioClassifier(context, engine, decoder, lfcc)

    @Provides
    @Singleton
    fun provideVideoDecoder(@ApplicationContext context: Context) =
        VideoDecoder(context)

    @Provides
    @Singleton
    fun provideVideoForensicAnalyzer(
        @ApplicationContext context: Context,
        videoDecoder: VideoDecoder,
        srmAnalyzer: SRMForensicAnalyzer,
        fftAnalyzer: FFTSpectralAnalyzer,
        klAnalyzer: KLCooccurrenceAnalyzer,
        metaClassifier: XGBoostMetaClassifier,
        audioClassifier: AudioClassifier,
        vfmProbe: VfmProbeAnalyzer,
    ) = VideoForensicAnalyzer(
        context, videoDecoder, srmAnalyzer, fftAnalyzer,
        klAnalyzer, metaClassifier, audioClassifier, vfmProbe,
    )
}
