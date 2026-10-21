package com.onndetect

import com.onndetect.domain.models.AnalysisStatus
import com.onndetect.domain.models.DetectionVerdict
import com.onndetect.domain.models.VerdictPolicy
import com.onndetect.inference.ModelIntegrityChecker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.InputStream

/**
 * The chain that produced NO verdict and NO percentage.
 *
 * The pipeline bails out with an INCONCLUSIVE result when the release
 * report is not `validated`, and `VerdictPolicy.decide` abstains from a
 * definite verdict unless it is given a validated release and a calibrated
 * threshold. A release whose independent ON-DEVICE evaluation is still
 * outstanding is desktop-verified: its packaged bytes are hash-bound,
 * measured on the sealed test split and calibrated, so it is USABLE - but it
 * must not be presented as device-verified.
 *
 * This test runs the REAL checker against the REAL shipped assets and then
 * drives the REAL policy with the result, so a regression in either is
 * caught against the bytes that actually ship.
 */
class PackagedVerdictChainTest {

    private val assets = File("src/main/assets")

    private fun open(path: String): InputStream = File(assets, path).inputStream()

    private fun report() = ModelIntegrityChecker.verifyModelsDetailed(
        setOf(ModelIntegrityChecker.IMAGE_MODEL),
    ) { path -> open(path) }

    @Test fun `the packaged release yields a usable calibrated report`() {
        val report = report()
        assertTrue("the packaged assets failed the gate: ${report.failures}", report.allOk)
        assertTrue("the packaged release must be usable", report.validated)
        assertNotNull("the fusion's operating point must be known", report.threshold)
        assertNotNull("the fusion's operating point must be known", report.fusionThreshold)
    }

    @Test fun `the packaged release is honest about the device evaluation`() {
        // The desktop pipeline measured the packaged bytes; no phone has
        // verified them yet. The report must say which.
        val report = report()
        assertTrue(
            "a release with no device report must not claim device validation",
            !report.deviceValidated,
        )
    }

    @Test fun `the second engine actually runs`() {
        // The frozen SigLIP-B/16 tower is in slot 6. If the gate refuses it,
        // the fusion runs without the probe and the whole release is a lie.
        val report = report()
        assertTrue("the slot-6 probe must clear the gate", report.cascadeValidated)
    }

    @Test fun `a score above the threshold produces a definite synthetic verdict`() {
        val report = report()
        val threshold = report.fusionThreshold!!
        val verdict = VerdictPolicy.decide(
            threshold + 0.1f, AnalysisStatus.SUCCESS, report.validated, report.threshold,
        )
        assertEquals(DetectionVerdict.SYNTHETIC, verdict)
    }

    @Test fun `a score below the threshold produces a definite authentic verdict`() {
        val report = report()
        val threshold = report.fusionThreshold!!
        val verdict = VerdictPolicy.decide(
            (threshold - 0.1f).coerceAtLeast(0f), AnalysisStatus.SUCCESS, report.validated, report.threshold,
        )
        assertEquals(DetectionVerdict.AUTHENTIC, verdict)
    }

    @Test fun `an unvalidated release still abstains`() {
        // The gate must keep refusing a release that is neither device- nor
        // desktop-verified. The fix is not "always produce a verdict".
        val verdict = VerdictPolicy.decide(0.9f, AnalysisStatus.SUCCESS, false, 0.5f)
        assertEquals(DetectionVerdict.INCONCLUSIVE, verdict)
    }
}
