package com.onndetect.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val ForensicColorScheme = darkColorScheme(
    primary = CyberCyan,
    onPrimary = BackgroundVoid,
    primaryContainer = DeepIndigo,
    onPrimaryContainer = TextPrimary,

    secondary = NeonPurple,
    onSecondary = BackgroundVoid,
    secondaryContainer = SurfaceElevated,
    onSecondaryContainer = NeonPurpleLight,

    tertiary = AuthenticGreen,
    onTertiary = BackgroundVoid,
    tertiaryContainer = AuthenticGreenContainer,
    onTertiaryContainer = AuthenticGreenLight,

    error = SyntheticRed,
    onError = TextPrimary,
    errorContainer = SyntheticRedContainer,
    onErrorContainer = SyntheticRedLight,

    background = BackgroundVoid,
    onBackground = TextPrimary,

    surface = SurfaceCard,
    onSurface = TextPrimary,

    surfaceVariant = SurfaceElevated,
    onSurfaceVariant = TextSecondary,

    surfaceContainer = SurfaceCard,
    surfaceContainerHigh = SurfaceElevated,
    surfaceContainerHighest = SurfaceHighlight,

    outline = BorderSubtle,
    outlineVariant = BorderLight
)

@Composable
fun OnnDetectTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = ForensicColorScheme,
        typography = Typography,
        content = content
    )
}
