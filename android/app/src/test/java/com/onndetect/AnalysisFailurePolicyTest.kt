package com.onndetect

import com.onndetect.domain.models.AnalysisStatus
import com.onndetect.domain.models.DetectionVerdict
import com.onndetect.domain.models.VerdictPolicy
import com.onndetect.inference.AnalysisFailurePolicy
import com.onndetect.inference.ModelContractException
import org.junit.Assert.assertEquals
import org.junit.Test

class AnalysisFailurePolicyTest {
    @Test fun modelInterfaceRejectionsAreFailuresEvenWhenBackendUsesIllegalArgument() {
        val backend = IllegalArgumentException("Tensor shape mismatch")
        val packaged = ModelContractException("Packaged model invocation rejected its tensor interface", backend)
        for (error in listOf(packaged, IllegalArgumentException("Wrapped model error", packaged))) {
            val status = AnalysisFailurePolicy.status(error)
            assertEquals(AnalysisStatus.MODEL_FAILURE, status)
            assertEquals(DetectionVerdict.INCONCLUSIVE, VerdictPolicy.decide(.9f, status, true, .5f))
        }
    }

    @Test fun invalidMediaRemainsUnsupportedAndOtherExecutionFailuresStayModelFailures() {
        assertEquals(AnalysisStatus.UNSUPPORTED_INPUT, AnalysisFailurePolicy.status(IllegalArgumentException("No audio track")))
        assertEquals(AnalysisStatus.MODEL_FAILURE, AnalysisFailurePolicy.status(IllegalStateException("Invalid model output")))
    }
}
