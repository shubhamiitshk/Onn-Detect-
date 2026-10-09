package com.onndetect.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

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
fun OnnDetectTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = ForensicColorScheme, typography = Typography, content = content)
}
