package com.onndetect.detection.kyc

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.util.SizeF

/**
 * Audit report for Video-KYC (V-CIP) camera injection, virtual camera drivers, and live stream tampering.
 */
data class KycInjectionReport(
    val isHardwareGenuine: Boolean,
    val virtualCameraDetected: Boolean,
    val hookDetected: Boolean,
    val emulatorDetected: Boolean,
    val cameraHardwareLevel: String, // "FULL", "LIMITED", "LEVEL_3", "LEGACY", "UNKNOWN"
    val physicalSensorVerified: Boolean,
    val injectionScore: Float, // 0.0 (Genuine Physical Sensor) to 1.0 (Injected / Virtual)
    val flags: List<String>,
)

/**
 * On-device camera HAL inspector and virtual-injection detector for Video-KYC (V-CIP).
 * Defeats OBS/ManyCam/VCam hooks, Xposed/Pine dynamic interception, and emulator video injection
 * directly on the device with zero network calls and zero additional runtime permissions.
 */
class VirtualCameraDetector {

    companion object {
        const val INJECTION_THRESHOLD = 0.50f

        val KNOWN_HOOK_CLASSES = listOf(
            "de.robv.android.xposed.XposedBridge",
            "de.robv.android.xposed.XposedHelpers",
            "top.canyie.pine.Pine",
            "me.weishu.epic.art.Epic",
            "io.github.lsposed.LSPosed",
            "com.vcam.hook.VCam",
            "com.vcam.hook.HookMain",
            "com.manycam.android.VirtualCameraService",
        )
    }

    /**
     * Inspects device camera-HAL characteristics, dynamic hooking frameworks,
     * and emulator environments on Android.
     */
    fun inspect(context: Context, videoTemporalCosine: Float? = null): KycInjectionReport {
        var hasPhysicalFrontCamera = false
        var hardwareLevelInt = -1
        var hasValidSensorSize = false
        var hasExposureRange = false

        runCatching {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
            val cameraIds = cameraManager?.cameraIdList ?: emptyArray()

            for (id in cameraIds) {
                val chars = cameraManager?.getCameraCharacteristics(id) ?: continue
                val facing = chars.get(CameraCharacteristics.LENS_FACING)

                if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    hasPhysicalFrontCamera = true
                    hardwareLevelInt = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ?: -1

                    val sensorSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    if (sensorSize != null && sensorSize.width > 0f && sensorSize.height > 0f) {
                        hasValidSensorSize = true
                    }

                    val exposureRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
                    if (exposureRange != null && exposureRange.lower > 0) {
                        hasExposureRange = true
                    }
                    break
                }
            }
        }

        val isHookPresent = checkHookClasses()
        val isEmulatorEnv = checkEmulator()
        val isFrozenLoop = videoTemporalCosine != null && videoTemporalCosine > 0.9998f

        return evaluateProfile(
            hasPhysicalFrontCamera = hasPhysicalFrontCamera,
            hardwareLevel = hardwareLevelInt,
            hasValidSensorSize = hasValidSensorSize,
            hasExposureRange = hasExposureRange,
            isHookPresent = isHookPresent,
            isEmulator = isEmulatorEnv,
            frozenFrameLoop = isFrozenLoop,
        )
    }

    /**
     * Pure unit-testable evaluation of hardware camera properties, hooks, and emulator environment.
     */
    fun evaluateProfile(
        hasPhysicalFrontCamera: Boolean,
        hardwareLevel: Int,
        hasValidSensorSize: Boolean,
        hasExposureRange: Boolean,
        isHookPresent: Boolean,
        isEmulator: Boolean,
        frozenFrameLoop: Boolean = false,
    ): KycInjectionReport {
        val flags = mutableListOf<String>()

        val levelString = when (hardwareLevel) {
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
            else -> "UNKNOWN"
        }

        var injectionScore = 0.0f
        var virtualCameraDetected = false

        // 1. Hooking framework check
        if (isHookPresent) {
            injectionScore += 0.85f
            flags.add("Dynamic hooking framework active (Xposed/Pine/VCam injection module)")
        }

        // 2. Emulator environment check
        if (isEmulator) {
            injectionScore += 0.80f
            flags.add("Virtual emulator environment detected (Android system fingerprint mock)")
        }

        // 3. Physical front camera presence check
        if (!hasPhysicalFrontCamera) {
            injectionScore += 0.70f
            virtualCameraDetected = true
            flags.add("Missing physical front-facing camera sensor in CameraManager HAL")
        } else {
            // Virtual camera drivers on Android typically report LEGACY level and lack physical sensor bounds
            if (hardwareLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
                injectionScore += 0.40f
                flags.add("Camera reports LEGACY driver HAL (typical of software/virtual camera drivers)")
            }
            if (!hasValidSensorSize) {
                injectionScore += 0.50f
                virtualCameraDetected = true
                flags.add("Camera lacks physical silicon sensor dimensions (virtual camera indicator)")
            }
            if (!hasExposureRange) {
                injectionScore += 0.30f
                flags.add("Missing hardware shutter exposure range")
            }
        }

        // 4. Temporal frozen loop check
        if (frozenFrameLoop) {
            injectionScore += 0.60f
            flags.add("Static photo re-animation or frozen video frame loop detected (cosine > 0.9998)")
        }

        val finalScore = injectionScore.coerceIn(0f, 1f)
        val isGenuine = finalScore < INJECTION_THRESHOLD && !isHookPresent && !isEmulator && hasPhysicalFrontCamera

        return KycInjectionReport(
            isHardwareGenuine = isGenuine,
            virtualCameraDetected = virtualCameraDetected || isHookPresent,
            hookDetected = isHookPresent,
            emulatorDetected = isEmulator,
            cameraHardwareLevel = levelString,
            physicalSensorVerified = hasPhysicalFrontCamera && hasValidSensorSize,
            injectionScore = finalScore,
            flags = flags,
        )
    }

    private fun checkHookClasses(): Boolean {
        for (className in KNOWN_HOOK_CLASSES) {
            try {
                Class.forName(className)
                return true
            } catch (_: ClassNotFoundException) {
                // Class not found - expected on clean devices
            } catch (_: Throwable) {
                // ClassLoader error or security restriction
            }
        }
        return false
    }

    fun checkEmulator(
        fingerprint: String? = runCatching { Build.FINGERPRINT }.getOrNull(),
        hardware: String? = runCatching { Build.HARDWARE }.getOrNull(),
        product: String? = runCatching { Build.PRODUCT }.getOrNull(),
        manufacturer: String? = runCatching { Build.MANUFACTURER }.getOrNull(),
        board: String? = runCatching { Build.BOARD }.getOrNull(),
        brand: String? = runCatching { Build.BRAND }.getOrNull(),
    ): Boolean {
        val fp = fingerprint?.lowercase() ?: ""
        val hw = hardware?.lowercase() ?: ""
        val prod = product?.lowercase() ?: ""
        val mfg = manufacturer?.lowercase() ?: ""

        // Only the FINGERPRINT, HARDWARE and PRODUCT signals are decisive.
        //
        // `Build.BRAND == "generic"` and `Build.BOARD == "unknown"` were
        // previously part of this test, and they are NOT emulator signals:
        // plenty of genuine low-cost OEM devices report exactly those
        // values. Including them meant a real customer's KYC session could
        // be rejected as "emulator detected", which is the worst possible
        // failure for an anti-fraud check. They are still read (the
        // parameters exist) but they are no longer consulted.
        return fp.startsWith("generic") ||
            fp.startsWith("unknown") ||
            fp.contains("vbox") ||
            fp.contains("sdk_gphone") ||
            fp.contains("emulator") ||
            hw.contains("goldfish") ||
            hw.contains("ranchu") ||
            prod.contains("sdk") ||
            prod.contains("vbox") ||
            prod.contains("emulator") ||
            mfg.contains("genymotion")
    }
}
