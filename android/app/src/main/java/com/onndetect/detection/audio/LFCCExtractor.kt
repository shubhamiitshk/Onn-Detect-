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
