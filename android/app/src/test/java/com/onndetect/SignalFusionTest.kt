package com.onndetect

import com.onndetect.domain.models.ProvenanceInfo
import com.onndetect.domain.pipeline.SignalFusion
import org.junit.Assert.assertEquals
import org.junit.Test

class SignalFusionTest {
    private val fusion = SignalFusion()
    @Test fun `absent metadata leaves content unchanged`() {
        assertEquals(.73f, fusion.fuse(.73f, null), 0f)
    }
    @Test fun `unverified credential flags never alter content score`() {
        for (c2pa in listOf(false, true)) for (synth in listOf(false, true)) {
            val provenance = ProvenanceInfo(c2pa, synth, emptyMap())
            for (score in listOf(.01f, .2f, .5f, .8f, .99f)) assertEquals(score, fusion.fuse(score, provenance), 0f)
        }
    }
    @Test(expected = IllegalArgumentException::class) fun `nonfinite scores are rejected`() { fusion.fuse(Float.NaN, null) }
    @Test(expected = IllegalArgumentException::class) fun `out of range scores are rejected`() { fusion.fuse(-.01f, null) }
}
