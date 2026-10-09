package com.onndetect.inference

import com.onndetect.domain.models.AnalysisStatus

/** A packaged model or its caller's tensor violates the deployed inference contract. */
class ModelContractException(message: String, cause: Throwable? = null) : IllegalStateException(message, cause)

internal inline fun requireModelContract(condition: Boolean, message: () -> String) {
    if (!condition) throw ModelContractException(message())
}

/** Invalid media and invalid model interfaces require different recovery actions. */
object AnalysisFailurePolicy {
    fun status(error: Exception): AnalysisStatus {
        if (generateSequence<Throwable>(error) { it.cause }.take(16).any { it is ModelContractException }) {
            return AnalysisStatus.MODEL_FAILURE
        }
        return if (error is IllegalArgumentException) AnalysisStatus.UNSUPPORTED_INPUT else AnalysisStatus.MODEL_FAILURE
    }
}
