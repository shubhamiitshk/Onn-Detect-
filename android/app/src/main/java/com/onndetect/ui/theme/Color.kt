package com.onndetect.ui.theme

import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

// ==============================================================================
// ONN-DETECT CYBER-FORENSIC & LIQUID GLASS PALETTE
// Built on aerospace obsidian surfaces, translucent frosted glass, precision laser cyan,
// and semantic tri-color indicators
// ==============================================================================

// Primary Accents - Precision & Optics
val CyberCyan = Color(0xFF00E5FF)
val CyberCyanLight = Color(0xFF80F3FF)
val CyberCyanDark = Color(0xFF00B4D8)
val DeepIndigo = Color(0xFF3D5AFE)
val ElectricBlue = Color(0xFF2979FF)
val NeonPurple = Color(0xFFA855F7)
val NeonPurpleLight = Color(0xFFC084FC)

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
val BackgroundVoid = Color(0xFF070A10)
val SurfaceCard = Color(0xFF0E1422)
val SurfaceElevated = Color(0xFF151D30)
val SurfaceHighlight = Color(0xFF1D273F)
val SurfaceInteractive = Color(0xFF253350)

// Liquid Glass Surfaces & Specular Highlights
val GlassSurfaceBase = Color(0x99121B2E)
val GlassSurfaceElevated = Color(0xB319253E)
val GlassDockSurface = Color(0xCC0D1424)
val GlassSpecularTop = Color(0x55FFFFFF)
val GlassSpecularEdge = Color(0x2200E5FF)
val GlassInnerShadow = Color(0x6603060C)

// Borders & Dividers
val BorderSubtle = Color(0xFF1F2B45)
val BorderLight = Color(0xFF2C3E63)
val BorderCyanGlow = Color(0x4000E5FF)
val BorderPurpleGlow = Color(0x40A855F7)

// Text & Content Hierarchy
val TextPrimary = Color(0xFFF8FAFC)
val TextSecondary = Color(0xFF94A3B8)
val TextTertiary = Color(0xFF64748B)
val TextDisabled = Color(0xFF475569)

// Gradients
val PrimaryGradient = Brush.horizontalGradient(
    colors = listOf(CyberCyan, DeepIndigo)
)

val ScanGlowGradient = Brush.verticalGradient(
    colors = listOf(CyberCyan.copy(alpha = 0.25f), Color.Transparent)
)

val CardBorderGradient = Brush.linearGradient(
    colors = listOf(CyberCyan.copy(alpha = 0.4f), DeepIndigo.copy(alpha = 0.1f), NeonPurple.copy(alpha = 0.3f))
)

val LiquidGlassBorderBrush = Brush.verticalGradient(
    colors = listOf(
        Color.White.copy(alpha = 0.32f),
        CyberCyan.copy(alpha = 0.22f),
        NeonPurple.copy(alpha = 0.18f),
    )
)

val LiquidGlassSurfaceBrush = Brush.verticalGradient(
    colors = listOf(
        Color(0xCC1A2640),
        Color(0x990E1525),
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
