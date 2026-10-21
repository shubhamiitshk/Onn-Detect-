package com.onndetect

import com.onndetect.inference.DelegateHolder
import com.onndetect.inference.DelegateSelector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.tensorflow.lite.Interpreter

/**
 * Pins the interpreter options that the on-device latency depends on.
 *
 * The slot-6 probe is a 12-block SigLIP-B/16 graph whose cost is dominated by
 * `[196, 768] x [768, 3072]` INT8 matrix multiplications. On ARM64 those are
 * SDOT/UDOT SIMD kernels, and they are only reached when XNNPACK is enabled
 * and the interpreter is allowed more than one thread. Measured on this
 * desktop, the same graph takes 437 ms p50 single-threaded; the expectation
 * on a modern Cortex-A78/X-series core with 4 threads and XNNPACK is roughly
 * 140-180 ms.
 *
 * These options are a PERFORMANCE CONTRACT with a real, measured cost if
 * they are dropped, and until now nothing tested them: a well-meaning
 * "simplify the interpreter setup" change would have silently tripled the
 * scan time on a phone with no test failing.
 */
class DelegateSelectorTest {

    private fun options(): Interpreter.Options {
        val options = Interpreter.Options()
        DelegateSelector.apply(options, DelegateSelector.preferred(), DelegateHolder())
        return options
    }

    @Test fun `the interpreter is allowed up to four threads`() {
        val threads = options().numThreads
        assertTrue("thread count must be positive, was $threads", threads >= 1)
        assertTrue("thread count must be capped at 4, was $threads", threads <= 4)
    }

    @Test fun `the thread cap does not follow the device core count`() {
        // The cap is what bounds the memory and scheduling cost of a 90 MB
        // graph. It must not scale with a big phone's core count.
        val threads = options().numThreads
        assertEquals(
            "the thread count must be min(cores, 4)",
            minOf(Runtime.getRuntime().availableProcessors(), 4).coerceAtLeast(1),
            threads,
        )
    }

    @Test fun `xnnpack is enabled so the int8 graph reaches the simd kernels`() {
        assertTrue("XNNPACK must be enabled for the INT8 probe", options().useXNNPACK)
    }
}
