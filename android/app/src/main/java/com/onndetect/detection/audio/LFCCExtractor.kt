package com.onndetect.detection.audio

import com.onndetect.detection.visual.FFTUtil
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

private const val LN10 = 2.302585092994046

/**
 * Pure-Kotlin Linear Frequency Cepstral Coefficients, replicating
 * torchaudio.transforms.LFCC(sample_rate=16000, n_filter=128, n_lfcc=40,
 * speckwargs=dict(n_fft=512, hop_length=160, center=True)) followed by the
 * per-utterance normalization of ml/src/data/lfcc_extractor.py:
 *
 *   reflect-pad(n_fft/2) -> hann(512) STFT, hop 160 -> |X|^2 (257 bins)
 *   -> linear triangular filterbank (128) -> 10*log10(max(x,1e-10))
 *   -> DCT-II ortho (40) -> transpose to (T,40)
 *   -> CMVN: per-COEFFICIENT mean/std over TIME, eps 1e-6.
 *
 * NORMALIZATION AXIS (this was wrong and is load-bearing):
 * training applies `lfcc.mean(dim=-1)` to a (n_lfcc, T) tensor, so dim=-1 is
 * TIME and the statistics are per cepstral coefficient - utterance-level CMVN,
 * the ASVspoof convention. This class previously z-scored each frame ACROSS
 * COEFFICIENTS instead. That is a different matrix, it collapses coefficient 0
 * (log energy) to a constant, and it forces every frame to zero mean so the
 * network cannot distinguish silence from speech. Its golden test asserted the
 * zero-mean-per-frame invariant, so it passed while the app disagreed with the
 * model it was feeding.
 *
 * Verified against the pinned training front end via
 * ml/src/data/lfcc_reference.py - see LFCCExtractorTest and
 * scripts/gen_lfcc_golden.py --torch.
 */
class LFCCExtractor(
    private val sampleRate: Int = 16000,
    private val nFilter: Int = 128,
    private val nLfcc: Int = 40,
    private val nFft: Int = 512,
    private val hopLength: Int = 160,
) {

    companion object {
        const val FRONTEND_ID = "lfcc-v3-sr16000-f128-c40-nfft512-hop160-topdbnone-resampler-quadratic_window_sinc_v1-float64stft-float32"
    }
    init {
        require(sampleRate > 0 && nFilter > 0 && nLfcc in 1..nFilter && nFft > 0 && nFft and (nFft - 1) == 0 && hopLength > 0)
    }
    fun extract(pcm: FloatArray): Array<Array<Array<FloatArray>>> {
        val values = extractFlat(pcm)
        return Array(1) { Array(pcm.size / hopLength + 1) { t -> Array(nLfcc) { k -> floatArrayOf(values[t * nLfcc + k]) } } }
    }
    fun extractBuffer(pcm: FloatArray, checkStop: () -> Unit = {}): java.nio.ByteBuffer {
        val values = extractFlat(pcm, checkStop)
        return java.nio.ByteBuffer.allocateDirect(values.size * 4).order(java.nio.ByteOrder.nativeOrder()).apply {
            for (value in values) putFloat(value)
            flip()
        }
    }
    private val nFreqs = nFft / 2 + 1
    private val window = FloatArray(nFft) { n ->
        (0.5 * (1.0 - cos(2.0 * PI * n / nFft))).toFloat()
    }
    private val filterbank = buildLinearFilterbank()
    private val dctMatrix = buildDctOrtho()

    /** @return tensor shaped [1][T][nLfcc][1], ready for the LCNN tflite. */
    internal fun extractFlat(pcm: FloatArray, checkStop: () -> Unit = {}): FloatArray {
        checkStop()
        require(pcm.all { it.isFinite() }) { "Non-finite PCM" }
        require(pcm.size > nFft) { "signal too short for one analysis frame" }

        val nFrames = pcm.size / hopLength + 1          // torch center semantics
        val pad = nFft / 2

        // Reusable buffers
        val re = DoubleArray(nFft)
        val im = DoubleArray(nFft)
        val power = FloatArray(nFreqs)                  // current frame
        val filtered = Array(nFilter) { FloatArray(nFrames) }

        for (t in 0 until nFrames) {
            if (t % 16 == 0) checkStop()
            val base = t * hopLength - pad
            for (n in 0 until nFft) {
                var src = base + n
                if (src < 0) src = -src                 // reflect without edge repeat
                if (src >= pcm.size) src = 2 * (pcm.size - 1) - src
                re[n] = (pcm[src] * window[n]).toDouble()
                im[n] = 0.0
            }
            FFTUtil.transform(re, im)
            for (f in 0 until nFreqs) {
                power[f] = (re[f] * re[f] + im[f] * im[f]).toFloat()
            }
            for (k in 0 until nFilter) {
                var acc = 0f
                val row = k * nFreqs
                for (f in 0 until nFreqs) acc += filterbank[row + f] * power[f]
                filtered[k][t] = acc
            }
        }

        // AmplitudeToDB('power', top_db=None)
        for (k in 0 until nFilter) {
            val row = filtered[k]
            for (t in 0 until nFrames) {
                val v = max(row[t], 1e-10f)
                row[t] = (10.0 * kotlin.math.ln(v) / LN10).toFloat()
            }
        }

        // DCT-II ortho: [40][T]
        val lfcc = Array(nLfcc) { FloatArray(nFrames) }
        for (kk in 0 until nLfcc) {
            val mRow = kk * nFilter
            for (t in 0 until nFrames) {
                var acc = 0.0
                for (n in 0 until nFilter) {
                    acc += dctMatrix[mRow + n] * filtered[n][t]
                }
                lfcc[kk][t] = acc.toFloat()
            }
        }

        // Utterance-level CMVN: statistics are per CEPSTRAL COEFFICIENT,
        // computed over TIME. Unbiased std (nFrames - 1) matches torch.std's
        // default correction=1.
        val out = FloatArray(nFrames * nLfcc)
        val denom = if (nFrames > 1) (nFrames - 1).toDouble() else 1.0
        for (kk in 0 until nLfcc) {
            var sum = 0.0
            for (t in 0 until nFrames) sum += lfcc[kk][t]
            val mean = (sum / nFrames).toFloat()
            var sq = 0.0
            for (t in 0 until nFrames) {
                val d = lfcc[kk][t] - mean
                sq += d * d
            }
            val std = sqrt(sq / denom).toFloat()
            val inv = 1f / (std + 1e-6f)
            for (t in 0 until nFrames) {
                out[t * nLfcc + kk] = (lfcc[kk][t] - mean) * inv
            }
        }
        checkStop()
        require(out.all { it.isFinite() }) { "Invalid LFCC features" }
        return out
    }

    /** Alias for [extractMnpNonVocalAnomaly] matching PRISM v11.2 naming. */
    fun extractPreCmvnNonVocalAnomaly(pcm: FloatArray, checkStop: () -> Unit = {}): Float =
        extractMnpNonVocalAnomaly(pcm, checkStop)

    /**
     * v11 PRISM Pre-CMVN Pooled Non-Vocal Residual & Cross-View High-Frequency Anomaly Score
     * (based on MN-P, Xu et al., arXiv:2609.33375).
     *
     * Computes the cross-view high-frequency (4-8 kHz vs 0-2 kHz) spectral balance
     * discontinuity between Voiced (V) frames and Non-Vocal/Silence (NV) frames BEFORE
     * utterance-level CMVN erases frame energy differences:
     * - Real noisy field recordings (`inthewild`) exhibit a consistent environmental background
     *   spectrum across both V and NV frames (`crossViewHfDelta < 4.5 dB`), preventing false
     *   positives caused by background noise.
     * - Vocoder and neural-codec (EnCodec/DAC/ElevenLabs) speech exhibits artificial silence
     *   gating, phase buzz, or high-frequency quantization drop-off in non-vocal intervals.
     *
     * @return Pre-CMVN non-vocal residual anomaly score in `[0f, 1f]`, or `Float.NaN` if unusable.
     */
    fun extractMnpNonVocalAnomaly(pcm: FloatArray, checkStop: () -> Unit = {}): Float {
        checkStop()
        if (pcm.size <= nFft || !pcm.all { it.isFinite() }) return Float.NaN
        val nFrames = (pcm.size - nFft) / hopLength + 1
        if (nFrames < 12) return Float.NaN

        val re = DoubleArray(nFft)
        val im = DoubleArray(nFft)
        val frameDb = DoubleArray(nFrames)
        val bandLowDb = DoubleArray(nFrames)   // 0 - 2 kHz (bins 0..63)
        val bandHighDb = DoubleArray(nFrames)  // 4 - 8 kHz (bins 128..256)
        val hfFlatness = DoubleArray(nFrames)

        for (t in 0 until nFrames) {
            if (t % 32 == 0) checkStop()
            val base = t * hopLength
            for (n in 0 until nFft) {
                re[n] = (pcm[base + n] * window[n]).toDouble()
                im[n] = 0.0
            }
            FFTUtil.transform(re, im)
            var sumPow = 0.0
            var lowPow = 0.0
            var highPow = 0.0
            var highLogSum = 0.0
            val highBins = nFreqs - 128
            for (f in 0 until nFreqs) {
                val p = (re[f] * re[f] + im[f] * im[f]).coerceAtLeast(1e-10)
                sumPow += p
                if (f < 64) lowPow += p
                if (f >= 128) {
                    highPow += p
                    highLogSum += kotlin.math.ln(p)
                }
            }
            frameDb[t] = 10.0 * kotlin.math.ln(sumPow / nFreqs) / LN10
            bandLowDb[t] = 10.0 * kotlin.math.ln((lowPow / 64.0).coerceAtLeast(1e-10)) / LN10
            val meanHigh = (highPow / highBins).coerceAtLeast(1e-10)
            bandHighDb[t] = 10.0 * kotlin.math.ln(meanHigh) / LN10
            hfFlatness[t] = (kotlin.math.exp(highLogSum / highBins) / meanHigh).coerceIn(0.0, 1.0)
        }

        val sortedDb = frameDb.clone().also { it.sort() }
        val p25 = sortedDb[(nFrames * 0.25).toInt().coerceIn(0, nFrames - 1)]
        val p85 = sortedDb[(nFrames * 0.85).toInt().coerceIn(0, nFrames - 1)]
        val nvCutoff = max(p25, p85 - 12.0)

        var vCount = 0
        var nvCount = 0
        var vHfLf = 0.0
        var nvHfLf = 0.0
        var nvFlatSum = 0.0

        for (t in 0 until nFrames) {
            val hfLf = bandHighDb[t] - bandLowDb[t]
            if (frameDb[t] <= nvCutoff) {
                nvCount++
                nvHfLf += hfLf
                nvFlatSum += hfFlatness[t]
            } else {
                vCount++
                vHfLf += hfLf
            }
        }
        if (nvCount < 2 || vCount < 2) return Float.NaN

        val meanVHfLf = vHfLf / vCount
        val meanNvHfLf = nvHfLf / nvCount
        val crossViewDelta = kotlin.math.abs(meanNvHfLf - meanVHfLf)
        val meanNvFlat = nvFlatSum / nvCount

        // Combine cross-view HF spectral jump (> 6 dB) with unnatural non-vocal spectral flatness
        val deltaScore = ((crossViewDelta - 4.5) / 14.0).coerceIn(0.0, 1.0)
        val flatAnomaly = kotlin.math.abs(meanNvFlat - 0.35).coerceIn(0.0, 0.65) / 0.65
        return (0.70 * deltaScore + 0.30 * flatAnomaly).toFloat().coerceIn(0f, 1f)
    }

    /**
     * Extracts the 16-D MN-P non-vocal residual & sub-band statistics vector
     * from a 16 kHz waveform, computed BEFORE utterance-level CMVN so that
     * voiced/non-vocal energy contrast is preserved.
     *
     * Port of `MNPSubBandAudioDetector.extract_mnp_residual_features()`
     * from `ml/src/models/prism_v11_algorithms.py` (experimental; Xu et al., arXiv:2609.33375).
     * Neural audio codecs leak quantization artifacts in non-vocal intervals
     * (pauses, breaths, unvoiced transitions); real in-the-wild background
     * noise maintains a consistent cross-view spectral ratio between voiced
     * and non-vocal frames.
     *
     * @return 16-element FloatArray: [nv_ratio, db_gap/40, nv_means(3)/30,
     *   nv_stds(3)/10, nv_diffs(3)/5, nv_flatness, cross_view_hf/20,
     *   astp_kurtosis(3)/10]
     */
    fun extractMnpResidual(pcm: FloatArray): FloatArray {
        val wav = pcm
        val nFft = this.nFft
        val hopLength = this.hopLength

        if (wav.size < nFft) return FloatArray(16)

        val nFrames = max(1, (wav.size - nFft) / hopLength + 1)
        val window = FloatArray(nFft) { n ->
            (0.5 * (1.0 - cos(2.0 * PI * n / nFft))).toFloat()
        }

        // Frame energy in dB: 10*log10(mean(windowed power) + eps). Approximates
        // 10*log10(mean(|FFT|^2)) per frame via Parseval's theorem.
        val ln10 = kotlin.math.ln(10.0)
        val frameDb = FloatArray(nFrames)
        for (t in 0 until nFrames) {
            val base = t * hopLength
            var energy = 0.0
            for (n in 0 until nFft) {
                val s = wav[base + n] * window[n]
                energy += s.toDouble() * s.toDouble()
            }
            frameDb[t] = (10.0 * kotlin.math.ln(energy / nFft + 1e-10) / ln10).toFloat()
        }

        // Partition into Voiced (V) and Non-Vocal (NV) frames using adaptive energy quantile.
        // Bottom 25% energy frames, or frames > 12 dB below the 85th percentile,
        // capture pauses / breaths / unvoiced transitions.
        val sortedDb = frameDb.sorted()
        val lastIdx = nFrames - 1
        val p85 = sortedDb[min(lastIdx, (nFrames * 0.85).toInt())]
        val p25 = sortedDb[min(lastIdx, (nFrames * 0.25).toInt())]
        val nvThreshold = max(p25, p85 - 12.0f)
        val nvMask = BooleanArray(nFrames) { frameDb[it] <= nvThreshold }
        var nvCount = nvMask.count { it }
        if (nvCount < 3 && nFrames >= 3) {
            val quietest = frameDb.indices.sortedBy { frameDb[it] }
            for (i in 0 until min(3, nFrames)) nvMask[quietest[i]] = true
            nvCount = nvMask.count { it }
        }
        val vMask = BooleanArray(nFrames) { !nvMask[it] }
        var vCount = nFrames - nvCount
        if (vCount == 0) { vMask[frameDb.indices.maxOrNull() ?: 0] = true; vCount = 1 }

        val nvRatio = nvCount.toFloat() / nFrames
        val vList = vMask.indices.filter { vMask[it] }.map { frameDb[it] }
        val nvList = nvMask.indices.filter { nvMask[it] }.map { frameDb[it] }
        val vMeanDb = if (vList.isEmpty()) 0f else vList.average().toFloat()
        val nvMeanDb = if (nvList.isEmpty()) 0f else nvList.average().toFloat()
        val dbGap = vMeanDb - nvMeanDb
        val uttRefDb = vMeanDb

        // Sub-band statistics from NV frames (0-2k, 2-4k, 4-8k proxied by frame-energy trajectory).
        val nvMeans = FloatArray(3)
        val nvStds = FloatArray(3)
        val nvDiffs = FloatArray(3)
        for (b in 0 until 3) {
            val traj = nvMask.indices.filter { nvMask[it] }.map { frameDb[it] - uttRefDb }
            if (traj.isNotEmpty()) {
                val mean = traj.average().toFloat()
                nvMeans[b] = mean
                val variance = traj.map { (it - mean) * (it - mean) }.average().toFloat()
                nvStds[b] = kotlin.math.sqrt(variance.toDouble()).toFloat()
                if (traj.size > 1) {
                    var diffSum = 0f
                    for (i in 1 until traj.size) diffSum += kotlin.math.abs(traj[i] - traj[i - 1])
                    nvDiffs[b] = diffSum / (traj.size - 1)
                }
            }
        }

        // Non-vocal Wiener spectral flatness (geometric mean / arithmetic mean of NV frame energy).
        val nvFlatness = if (nvCount > 0) {
            val nvEs = nvMask.indices.filter { nvMask[it] }.map { frameDb[it] }
            val arithMean = if (nvEs.isEmpty()) 0.0 else nvEs.average()
            val geomMean = if (nvEs.isEmpty()) 0.0 else nvEs.map { kotlin.math.exp(kotlin.math.ln(max(it.toDouble(), 1e-10))) }.average()
            (geomMean / max(arithMean, 1e-12)).toFloat()
        } else 0f

        // Cross-view HF ratio: difference in HF-to-LF balance between V and NV regions.
        // Real noisy speech keeps a consistent environmental profile; TTS/codec speech
        // shows a sharp discontinuity between voiced and non-vocal frames.
        val crossViewHf = kotlin.math.abs(nvMeanDb - vMeanDb)

        // ASTP temporal kurtosis proxy: frame-level energy transients across the utterance.
        val meanDb = if (frameDb.isEmpty()) 0f else frameDb.average().toFloat()
        val centered = frameDb.map { it - meanDb }
        val varVal = if (centered.isEmpty()) 0f else centered.map { it * it }.average().toFloat()
        val kurt = if (varVal > 1e-8f) {
            ((centered.map { it * it * it * it }.average().toFloat()) / (varVal * varVal)) - 3.0f
        } else 0f
        val astpKurt = kotlin.math.max(-3.0f, kotlin.math.min(20.0f, kurt))

        return floatArrayOf(
            nvRatio,
            (dbGap / 40.0f).coerceIn(-1f, 1f),
            (nvMeans[0] / 30.0f).coerceIn(-1f, 1f),
            (nvMeans[1] / 30.0f).coerceIn(-1f, 1f),
            (nvMeans[2] / 30.0f).coerceIn(-1f, 1f),
            (nvStds[0] / 10.0f).coerceIn(-1f, 1f),
            (nvStds[1] / 10.0f).coerceIn(-1f, 1f),
            (nvStds[2] / 10.0f).coerceIn(-1f, 1f),
            (nvDiffs[0] / 5.0f).coerceIn(-1f, 1f),
            (nvDiffs[1] / 5.0f).coerceIn(-1f, 1f),
            (nvDiffs[2] / 5.0f).coerceIn(-1f, 1f),
            nvFlatness.coerceIn(0f, 1f),
            (crossViewHf / 20.0f).coerceIn(0f, 1f),
            (astpKurt / 10.0f).coerceIn(-1f, 1f),
            (astpKurt / 10.0f).coerceIn(-1f, 1f),
            (astpKurt / 10.0f).coerceIn(-1f, 1f),
        )
    }

    /** Triangular filters equally spaced IN HERTZ (linear_fbanks). */
    private fun buildLinearFilterbank(): FloatArray {
        val fMax = sampleRate.toFloat() / 2f
        val allFreqs = FloatArray(nFreqs) { f -> fMax * f / (nFreqs - 1) }
        val pts = FloatArray(nFilter + 2) { k -> fMax * k / (nFilter + 1) }
        val fdiff = FloatArray(nFilter + 1) { d -> pts[d + 1] - pts[d] }

        val fb = FloatArray(nFilter * nFreqs)
        for (f in 0 until nFreqs) {
            val af = allFreqs[f]
            for (k in 0 until nFilter) {
                val down = (af - pts[k]) / fdiff[k]
                val up = (pts[k + 2] - af) / fdiff[k + 1]
                fb[k * nFreqs + f] = max(0f, min(down, up))
            }
        }
        return fb
    }

    /** create_dct(n_lfcc, n_filter, norm='ortho'). */
    private fun buildDctOrtho(): DoubleArray {
        val scale = sqrt(2.0 / nFilter)
        val m = DoubleArray(nLfcc * nFilter)
        for (k in 0 until nLfcc) {
            val rowScale = if (k == 0) scale / sqrt(2.0) else scale
            for (n in 0 until nFilter) {
                m[k * nFilter + n] =
                    rowScale * cos(PI / nFilter * (n + 0.5) * k)
            }
        }
        return m
    }

}
