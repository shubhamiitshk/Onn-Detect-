package com.onndetect

import com.onndetect.domain.models.*
import org.junit.Assert.assertEquals
import org.junit.Test

class VerdictPolicyTest {
    @Test fun `all failures and unvalidated scores abstain even above threshold`() {
        for (status in AnalysisStatus.values().filter { it != AnalysisStatus.SUCCESS }) {
            assertEquals(DetectionVerdict.INCONCLUSIVE, VerdictPolicy.decide(.99f, status, true, .2f))
        }
        assertEquals(DetectionVerdict.INCONCLUSIVE, VerdictPolicy.decide(.99f, AnalysisStatus.SUCCESS, false, .2f))
    }
    @Test fun `missing and nonfinite scores and calibration abstain`() {
        for (score in listOf(-1f, Float.NaN, Float.POSITIVE_INFINITY, 2f)) {
            assertEquals(DetectionVerdict.INCONCLUSIVE, VerdictPolicy.decide(score, AnalysisStatus.SUCCESS, true, .2f))
        }
        for (threshold in listOf(null, Float.NaN, -.1f, 1.1f)) {
            assertEquals(DetectionVerdict.INCONCLUSIVE, VerdictPolicy.decide(.8f, AnalysisStatus.SUCCESS, true, threshold))
        }
    }
    @Test fun `validated operating point supplies coherent verdict`() {
        assertEquals(DetectionVerdict.SYNTHETIC, VerdictPolicy.decide(.24f, AnalysisStatus.SUCCESS, true, .225f))
        assertEquals(DetectionVerdict.AUTHENTIC, VerdictPolicy.decide(.2f, AnalysisStatus.SUCCESS, true, .225f))
    }
}
