package com.onndetect.detection.kyc

import android.hardware.camera2.CameraCharacteristics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VirtualCameraDetectorTest {

    private val detector = VirtualCameraDetector()

    @Test
    fun `genuine physical camera hardware profile passes with low injection score`() {
        val report = detector.evaluateProfile(
            hasPhysicalFrontCamera = true,
            hardwareLevel = CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL,
            hasValidSensorSize = true,
            hasExposureRange = true,
            isHookPresent = false,
            isEmulator = false,
            frozenFrameLoop = false,
        )

        assertTrue("Genuine physical camera should be marked genuine", report.isHardwareGenuine)
        assertFalse("Virtual camera should not be detected", report.virtualCameraDetected)
        assertFalse("No hook should be detected", report.hookDetected)
        assertFalse("No emulator should be detected", report.emulatorDetected)
        assertEquals("FULL", report.cameraHardwareLevel)
        assertTrue("Physical sensor must be verified", report.physicalSensorVerified)
        assertTrue("Injection score should be 0.0", report.injectionScore < 0.10f)
        assertTrue("Flags should be empty for clean physical hardware", report.flags.isEmpty())
    }

    @Test
    fun `Xposed or VCam injection hook triggers immediate virtual camera alert`() {
        val report = detector.evaluateProfile(
            hasPhysicalFrontCamera = true,
            hardwareLevel = CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL,
            hasValidSensorSize = true,
            hasExposureRange = true,
            isHookPresent = true,
            isEmulator = false,
        )

        assertFalse("Hooked camera must not be marked genuine", report.isHardwareGenuine)
        assertTrue("Virtual camera / injection must be flagged", report.virtualCameraDetected)
        assertTrue("Hook must be detected", report.hookDetected)
        assertTrue("Injection score should be high (>= 0.80)", report.injectionScore >= 0.80f)
        assertTrue("Flags should mention dynamic hooking framework", report.flags.any { it.contains("hooking framework") })
    }

    @Test
    fun `virtual camera driver lacking physical sensor size and reporting LEGACY level is flagged`() {
        val report = detector.evaluateProfile(
            hasPhysicalFrontCamera = true,
            hardwareLevel = CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY,
            hasValidSensorSize = false, // Virtual driver cannot report physical silicon dimensions
            hasExposureRange = false,
            isHookPresent = false,
            isEmulator = false,
        )

        assertFalse("Virtual camera driver must not be marked genuine", report.isHardwareGenuine)
        assertTrue("Virtual camera must be flagged", report.virtualCameraDetected)
        assertFalse("Physical sensor must not be verified", report.physicalSensorVerified)
        assertEquals("LEGACY", report.cameraHardwareLevel)
        assertTrue("Injection score should be high (>= 0.70)", report.injectionScore >= 0.70f)
        assertTrue("Flags should mention missing physical silicon sensor", report.flags.any { it.contains("physical silicon sensor") })
    }

    @Test
    fun `emulator environment triggers virtual camera injection alert`() {
        val report = detector.evaluateProfile(
            hasPhysicalFrontCamera = false,
            hardwareLevel = -1,
            hasValidSensorSize = false,
            hasExposureRange = false,
            isHookPresent = false,
            isEmulator = true,
        )

        assertFalse("Emulator must not be marked genuine", report.isHardwareGenuine)
        assertTrue("Emulator must be detected", report.emulatorDetected)
        assertTrue("Injection score should be high", report.injectionScore >= 0.80f)
    }

    @Test
    fun `frozen video frame loop triggers temporal injection alert`() {
        val report = detector.evaluateProfile(
            hasPhysicalFrontCamera = true,
            hardwareLevel = CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL,
            hasValidSensorSize = true,
            hasExposureRange = true,
            isHookPresent = false,
            isEmulator = false,
            frozenFrameLoop = true, // Replayed loop
        )

        assertFalse("Frozen loop should fail genuine certification", report.isHardwareGenuine)
        assertTrue("Flags should indicate static re-animation or frozen loop", report.flags.any { it.contains("frozen video frame loop") })
    }

    @Test
    fun `emulator fingerprinting correctly identifies known emulator environments and accepts real hardware`() {
        // Genymotion
        assertTrue(detector.checkEmulator(manufacturer = "Genymotion"))

        // Google emulator (ranchu / goldfish)
        assertTrue(detector.checkEmulator(hardware = "ranchu", fingerprint = "google/sdk_gphone64_x86_64/emu64x:13/TP1A"))

        // BlueStacks / VirtualBox
        assertTrue(detector.checkEmulator(fingerprint = "unknown/vbox86p/vbox86p:9/PPR1"))

        // Real Google Pixel phone
        assertFalse(
            detector.checkEmulator(
                fingerprint = "google/cheetah/cheetah:14/UD1A.230803.022.A3/10808018:user/release-keys",
                hardware = "tensor2",
                product = "cheetah",
                manufacturer = "Google",
                board = "cheetah",
                brand = "google"
            )
        )

        // Real Samsung Galaxy phone
        assertFalse(
            detector.checkEmulator(
                fingerprint = "samsung/dm3qxxx/dm3q:14/UP1A.231005.007/S918BXXU3BWJM:user/release-keys",
                hardware = "qcom",
                product = "dm3q",
                manufacturer = "samsung",
                board = "kalama",
                brand = "samsung"
            )
        )
    }
}
