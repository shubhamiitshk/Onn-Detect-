package com.onndetect

import android.app.Application
import android.os.StrictMode
import com.onndetect.detection.provenance.ReceiptSigner
import dagger.hilt.android.HiltAndroidApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltAndroidApp
class OnnDetectApplication : Application() {
    @Inject lateinit var receiptSigner: ReceiptSigner

    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        // Pre-generate the signing key off the main thread so the first scan
        // does not pay the key-generation cost on the UI thread.
        appScope.launch {
            (receiptSigner as? com.onndetect.detection.provenance.StrongBoxReceiptSigner)?.warmUp()
        }

        // Debug builds only: surface leaked Closeables ("A resource failed to
        // call close."), activity leaks and main-thread I/O in logcat with a
        // stack trace identifying the culprit.
        if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            StrictMode.setThreadPolicy(
                StrictMode.ThreadPolicy.Builder()
                    .detectAll()
                    .penaltyLog()
                    .build()
            )
            StrictMode.setVmPolicy(
                StrictMode.VmPolicy.Builder()
                    .detectLeakedClosableObjects()
                    .detectActivityLeaks()
                    .detectLeakedSqlLiteObjects()
                    .penaltyLog()
                    .build()
            )
        }
    }
}
