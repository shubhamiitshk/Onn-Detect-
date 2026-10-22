package com.onndetect

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.navigation.compose.rememberNavController
import com.onndetect.ui.navigation.ONNNavGraph
import com.onndetect.ui.share.SharedMedia
import com.onndetect.ui.share.SharedMediaIntents
import com.onndetect.ui.theme.OnnDetectTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    private val liveSharedMedia = mutableStateOf<SharedMedia?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT), navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT))
        // A share hand-off is consumed ONCE. Android re-delivers the same
        // Intent after a configuration change, and a scan must not run twice
        // because the user rotated the device.
        liveSharedMedia.value = SharedMediaIntents.consume(intent)
        setContent {
            OnnDetectTheme {
                Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    ONNNavGraph(rememberNavController(), sharedMedia = liveSharedMedia.value)
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        liveSharedMedia.value = SharedMediaIntents.consume(intent)
    }
}
