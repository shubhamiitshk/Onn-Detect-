package com.onndetect.detection.audio

import android.content.Context
import javax.inject.Inject

/** Identity verification is unavailable until a licensed encoder is validated. */
class SpeakerVerifier @Inject constructor() {
    @Suppress("UNUSED_PARAMETER")
    fun verifyIdentity(context: Context, audioAmplitudes: FloatArray): Float =
        throw UnsupportedOperationException("Voice identity verification is unavailable")

    @Suppress("UNUSED_PARAMETER")
    fun enrollTrustedVoicePrint(context: Context, audioAmplitudes: FloatArray): Unit =
        throw UnsupportedOperationException("Voice enrollment is unavailable")
}
