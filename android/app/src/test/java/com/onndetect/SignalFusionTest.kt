package com.onndetect

import com.onndetect.domain.models.ProvenanceInfo
import com.onndetect.domain.pipeline.SignalFusion
import org.junit.Assert.assertEquals
import org.junit.Test

class SignalFusionTest {

    private val fusion = SignalFusion()

    @Test
    fun `no provenance passes score through`() {
        assertEquals(0.73f, fusion.fuse(0.73f, null), 1e-6f)
        assertEquals(0.01f, fusion.fuse(0.01f, null), 1e-6f)
    }

    @Test
    fun `synthID watermark forces definitive AI verdict`() {
        val provenance = ProvenanceInfo(hasC2PA = false, hasSynthID = true, exifData = emptyMap())
        val fused = fusion.fuse(0.05f, provenance)
        assertEquals(1.0f, fused, 1e-6f)
    }

    @Test
    fun `valid C2PA strongly dampens the synthetic estimate`() {
        val provenance = ProvenanceInfo(hasC2PA = true, hasSynthID = false, exifData = emptyMap())
        assertEquals(0.08f, fusion.fuse(0.80f, provenance), 1e-5f)
        assertEquals(0.02f, fusion.fuse(0.20f, provenance), 1e-5f)
    }

    @Test
    fun `provenance with neither credential is passthrough`() {
        val provenance = ProvenanceInfo(hasC2PA = false, hasSynthID = false, exifData = emptyMap())
        assertEquals(0.42f, fusion.fuse(0.42f, provenance), 1e-6f)
    }

    @Test
    fun `synthID wins over C2PA`() {
        val both = ProvenanceInfo(hasC2PA = true, hasSynthID = true, exifData = emptyMap())
        assertEquals(1.0f, fusion.fuse(0.10f, both), 1e-6f)
    }
}
