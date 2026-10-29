package com.onndetect.ui.theme

import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

// ==============================================================================
// ONN-DETECT glass palette
// Quiet cool surfaces, soft specular edges, and accessible semantic colors.
// ==============================================================================

// Primary Accents - Precision & Optics
val CyberCyan = Color(0xFF81DDEB)
val CyberCyanLight = Color(0xFFB6EDF4)
val CyberCyanDark = Color(0xFF3AA8BC)
val DeepIndigo = Color(0xFF6475D9)
val ElectricBlue = Color(0xFF7DA8EF)
val NeonPurple = Color(0xFFA9A1E8)
val NeonPurpleLight = Color(0xFFCCC7F4)

// Dual-Engine Branding Accents
val EngineAColor = CyberCyan
val EngineCColor = NeonPurple
val CascadeArbitratedGlow = Color(0x3300E5FF)

// Semantic Verdict Triad - Authentic, Synthetic, Inconclusive
val AuthenticGreen = Color(0xFF10B981)
val AuthenticGreenLight = Color(0xFF34D399)
val AuthenticGreenContainer = Color(0xFF064E3B)
val AuthenticGreenGlow = Color(0x3310B981)

val SyntheticRed = Color(0xFFF43F5E)
val SyntheticRedLight = Color(0xFFFB7185)
val SyntheticRedContainer = Color(0xFF4C0519)
val SyntheticRedGlow = Color(0x33F43F5E)

val WarningAmber = Color(0xFFF59E0B)
val WarningAmberLight = Color(0xFFFBBF24)
val WarningAmberContainer = Color(0xFF451A03)
val WarningAmberGlow = Color(0x33F59E0B)

// Obsidian Surface Hierarchy (True dark mode with deep slate undertones)
val BackgroundVoid = Color(0xFF0A0E16)
val SurfaceCard = Color(0xFF131A26)
val SurfaceElevated = Color(0xFF1B2433)
val SurfaceHighlight = Color(0xFF253145)
val SurfaceInteractive = Color(0xFF304057)

// Liquid Glass Surfaces & Specular Highlights
val GlassSurfaceBase = Color(0xB31A2434)
val GlassSurfaceElevated = Color(0xCC253143)
val GlassDockSurface = Color(0xE6151D2B)
val GlassSpecularTop = Color(0x33FFFFFF)
val GlassSpecularEdge = Color(0x1F81DDEB)
val GlassInnerShadow = Color(0x6603060C)

// Borders & Dividers
val BorderSubtle = Color(0xFF2A3649)
val BorderLight = Color(0xFF3A4961)
val BorderCyanGlow = Color(0x2781DDEB)
val BorderPurpleGlow = Color(0x27A9A1E8)

// Text & Content Hierarchy
val TextPrimary = Color(0xFFF8FAFC)
val TextSecondary = Color(0xFFB2BFD1)
val TextTertiary = Color(0xFF8998AE)
val TextDisabled = Color(0xFF627087)

// Gradients
val PrimaryGradient = Brush.horizontalGradient(
    colors = listOf(CyberCyan, DeepIndigo)
)

val ScanGlowGradient = Brush.verticalGradient(
    colors = listOf(CyberCyan.copy(alpha = 0.25f), Color.Transparent)
)

val CardBorderGradient = Brush.linearGradient(
    colors = listOf(Color.White.copy(alpha = 0.17f), CyberCyan.copy(alpha = 0.12f), Color.White.copy(alpha = 0.05f))
)

val LiquidGlassBorderBrush = Brush.verticalGradient(
    colors = listOf(
        Color.White.copy(alpha = 0.20f),
        CyberCyan.copy(alpha = 0.10f),
        Color.White.copy(alpha = 0.05f),
    )
)

val LiquidGlassSurfaceBrush = Brush.verticalGradient(
    colors = listOf(
        Color(0xD9253042),
        Color(0xB3131B28),
    )
)

val HeroCardGradient = Brush.verticalGradient(
    colors = listOf(SurfaceElevated, SurfaceCard)
)

val AuthenticBannerGradient = Brush.verticalGradient(
    colors = listOf(AuthenticGreenContainer.copy(alpha = 0.7f), SurfaceCard)
)

val SyntheticBannerGradient = Brush.verticalGradient(
    colors = listOf(SyntheticRedContainer.copy(alpha = 0.7f), SurfaceCard)
)

val InconclusiveBannerGradient = Brush.verticalGradient(
    colors = listOf(WarningAmberContainer.copy(alpha = 0.7f), SurfaceCard)
)
