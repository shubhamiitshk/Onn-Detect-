package com.onndetect

import com.onndetect.inference.ModelIntegrityChecker
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.InputStream

/**
 * Runs the REAL integrity checker against the REAL shipped assets.
 *
 * The unit tests elsewhere use synthetic assets, so a release can be
 * internally consistent in the test harness and still be rejected by the
 * app. This test reads the assets that are actually packaged.
 */
class PackagedAssetsIntegrityTest {

    private val assets = File("src/main/assets")

    private fun open(path: String): InputStream =
        File(assets, path).inputStream()

    @Test fun `the packaged assets pass the integrity gate`() {
        val report = ModelIntegrityChecker.verifyModelsDetailed(
            setOf(ModelIntegrityChecker.IMAGE_MODEL),
        ) { path -> open(path) }
        println("allOk=${report.allOk} validated=${report.validated} " +
            "deviceValidated=${report.deviceValidated} cascade=${report.cascadeValidated} " +
            "threshold=${report.threshold} fusion=${report.fusionThreshold} reason=${report.validationReason}")
        println("failures=${report.failures}")
        assertTrue("packaged assets failed the integrity gate: ${report.failures}", report.allOk)
        assertTrue("the packaged release is not validated", report.validated)
    }
}
