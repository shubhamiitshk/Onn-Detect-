package com.onndetect.detection.kyc

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The false-positive paths of the V-CIP shield.
 *
 * This detector REJECTS a KYC session, so a false positive denies service
 * to a genuine customer. That is the worst failure mode for an anti-fraud
 * check, and the two paths that could cause it are pinned here:
 *
 *  - `Build.BRAND == "generic"` and `Build.BOARD == "unknown"` are NOT
 *    emulator signals. Plenty of genuine low-cost OEM devices report
 *    exactly those values, and including them meant a real customer's
 *    session could be rejected as "emulator detected".
 *  - a camera that merely reports the LEGACY hardware level is a WEAK
 *    signal on its own, and must not by itself read as a virtual camera.
 */
class VirtualCameraFalsePositiveTest {

    @Test fun `a genuine device reporting a generic brand is NOT an emulator`() {
        val detector = VirtualCameraDetector()
        // A real fingerprint and hardware, with the brand/board values that
        // genuine low-cost devices report.
        val emulator = detector.checkEmulator(
            fingerprint = "google/hardware_xxxx/device:14/UP1A.231005.007/1234567:user/release-keys",
            hardware = "marlin",
            product = "marlin",
            manufacturer = "Google",
            board = "unknown",
            brand = "generic",
        )
        assertFalse("a genuine device must not be reported as an emulator", emulator)
    }

    @Test fun `a real fingerprint with a generic brand is still not an emulator`() {
        val detector = VirtualCameraDetector()
        val emulator = detector.checkEmulator(
            fingerprint = "google/sunfish/device:13/TQ3A.230805.001/0123456:user/release-keys",
            hardware = "sunfish",
            product = "sunfish",
            manufacturer = "Google",
            board = "marlin",
            brand = "generic",
        )
        assertFalse("the brand value alone must not decide it", emulator)
    }

    @Test fun `a genuine emulator is still detected`() {
        val detector = VirtualCameraDetector()
        val emulator = detector.checkEmulator(
            fingerprint = "generic/sdk_gphone64_x86_64/emu64x:14/UP1A.231005.007/1234567:userdebug/x86_64",
            hardware = "ranchu",
            product = "sdk_x86_64",
            manufacturer = "Google",
            board = "ranchu",
            brand = "google",
        )
        assertTrue("a real emulator must still be detected", emulator)
    }

    @Test fun `a legacy camera alone does not read as a virtual camera`() {
        val detector = VirtualCameraDetector()
        val report = detector.evaluateProfile(
            hasPhysicalFrontCamera = true,
            hardwareLevel = android.hardware.camera2.CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY,
            hasValidSensorSize = true,
            hasExposureRange = true,
            isHookPresent = false,
            isEmulator = false,
        )
        assertTrue(
            "a legacy HAL on a real device must not be called a virtual camera",
            !report.virtualCameraDetected,
        )
        assertTrue(
            "a legacy HAL alone must not cross the injection threshold",
            report.injectionScore < VirtualCameraDetector.INJECTION_THRESHOLD,
        )
    }
}
