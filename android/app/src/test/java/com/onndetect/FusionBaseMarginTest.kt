package com.onndetect

import com.onndetect.detection.fusion.FusionBaseMargin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the Kotlin base margin to the Python implementation.
 *
 * The reference values below were computed by
 * `ml/scripts/train_xgboost_meta.py::probe_base_margin` in float64, from the
 * FLOAT32 probe score the pipeline actually carries - the score crosses the
 * Python/Java boundary as a float32, so a reference computed from a float64
 * literal would be comparing two different numbers.
 *
 * The transpiled margin is only correct if the two agree, and the transpiled
 * margin is what the app compares against the release threshold, so a silent
 * divergence here would move every verdict.
 */
class FusionBaseMarginTest {

    private fun assertClose(expected: Double, actual: Float, tolerance: Float = 1e-5f) {
        assertEquals(
            "base margin $actual differs from the Python reference $expected",
            expected.toFloat(),
            actual,
            tolerance,
        )
    }

    @Test fun `matches the python reference logits`() {
        assertClose(-5.741399363570731, FusionBaseMargin.logit(0.0032f))
        assertClose(-2.758649754399018, FusionBaseMargin.logit(0.0596f))
        assertClose(-1.3862943424934393, FusionBaseMargin.logit(0.2f))
        assertClose(0.0, FusionBaseMargin.logit(0.5f))
        assertClose(0.7236032572447939, FusionBaseMargin.logit(0.6734f))
        assertClose(2.6162486843374846, FusionBaseMargin.logit(0.9319f))
        assertClose(6.724216053698151, FusionBaseMargin.logit(0.9988f))
    }

    @Test fun `an unavailable probe contributes no margin`() {
        // NaN is what the pipeline sends when Engine C did not run. The
        // training rows with a masked slot 6 carried a margin of 0, so the
        // tree's learned missing branch stays consistent.
        assertEquals(0f, FusionBaseMargin.logit(Float.NaN), 0f)
    }

    @Test fun `a saturated probability contributes no margin`() {
        // log(0) and log(1) are not finite; the Python side clips to
        // 1e-6, and a zero margin is the safe, defined behaviour.
        assertEquals(0f, FusionBaseMargin.logit(0f), 0f)
        assertEquals(0f, FusionBaseMargin.logit(1f), 0f)
        assertEquals(0f, FusionBaseMargin.logit(-0.5f), 0f)
        assertEquals(0f, FusionBaseMargin.logit(1.5f), 0f)
    }

    @Test fun `the margin is monotone in the probe score`() {
        var previous = Float.NEGATIVE_INFINITY
        for (step in 1..99) {
            val probe = step / 100f
            val current = FusionBaseMargin.logit(probe)
            assertTrue("logit must increase with the probe score", current > previous)
            previous = current
        }
    }
}
