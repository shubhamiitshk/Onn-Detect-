package com.onndetect.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// Enforce Dark Mode for the Cyber Forensic Aesthetic
private val ForensicColorScheme = darkColorScheme(
    primary = CyberCyan,
    onPrimary = BackgroundObsidian,
    primaryContainer = DeepIndigo,
    onPrimaryContainer = TextPrimary,
    
    secondary = NeonPurple,
    onSecondary = BackgroundObsidian,
    
    tertiary = AuthenticGreen,
    onTertiary = BackgroundObsidian,
    
    error = FakeRed,
    onError = TextPrimary,
    errorContainer = FakeRed.copy(alpha = 0.2f),
    
    background = BackgroundObsidian,
    onBackground = TextPrimary,
    
    surface = SurfaceSlate,
    onSurface = TextPrimary,
    
    surfaceVariant = SurfaceVariantDark,
    onSurfaceVariant = TextSecondary,
    
    outline = OutlineDark
)

@Composable
fun OnnDetectTheme(
    darkTheme: Boolean = true, // Force dark theme for the forensic aesthetic
    dynamicColor: Boolean = false, // Disable dynamic color to enforce our cyber palette
    content: @Composable () -> Unit
) {
    val colorScheme = ForensicColorScheme
    
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            window.navigationBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = false
            WindowCompat.getInsetsController(window, view).isAppearanceLightNavigationBars = false
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
