package com.onndetect.domain.models

enum class DeviceTier(val minRamGb: Int) {
    LOW_END(4), MID_RANGE(6), HIGH_END(8)
}
