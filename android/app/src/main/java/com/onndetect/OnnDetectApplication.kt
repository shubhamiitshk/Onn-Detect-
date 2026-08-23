package com.onndetect

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class OnnDetectApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Initialize any global crash reporters or loggers here if needed
    }
}
