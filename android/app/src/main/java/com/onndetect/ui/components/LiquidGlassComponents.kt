package com.onndetect.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.onndetect.ui.navigation.Routes
import com.onndetect.ui.theme.*
import kotlin.math.cos
import kotlin.math.sin

/**
 * Primary destinations in the 3-Window Bottom Liquid Glass Dock:
 * 1. [DETECT]: On-Device Forensic Detection (Image, Audio, Video + History)
 * 2. [CHAT_RAG]: Multimodal Forensic RAG Chatbot (Video/Audio/Image Q&A & Timeline Search)
 * 3. [VOICE_VAULT]: Family Safe-Voice Biometric Vault & Instant Scam Guard
 */
enum class GlassBottomTab(
    val route: String,
    val title: String,
    val subtitle: String,
    val iconSymbol: String,
    val accentColor: Color,
) {
    DETECT(
        route = Routes.HOME,
        title = "Detect",
        subtitle = "Media scans",
        iconSymbol = "🔍",
        accentColor = CyberCyan,
    ),
    CHAT_RAG(
        route = Routes.CHAT_RAG_BASE,
        title = "Chat",
        subtitle = "Private assistant",
        iconSymbol = "💬",
        accentColor = NeonPurple,
    ),
    VOICE_VAULT(
        route = Routes.VOICE_VAULT,
        title = "Voice Vault",
        subtitle = "Saved voices",
        iconSymbol = "🛡️",
        accentColor = AuthenticGreen,
    ),
}

@Composable
fun LiquidGlassScaffold(
    activeTab: GlassBottomTab,
    onSelectTab: (GlassBottomTab) -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.(PaddingValues) -> Unit,
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(BackgroundVoid)
    ) {
        Box(
            modifier = Modifier
                .size(280.dp)
                .offset(x = (-60).dp, y = (-40).dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(CyberCyan.copy(alpha = 0.075f), Color.Transparent)
                    ),
                    shape = CircleShape
                )
        )
        Box(
            modifier = Modifier
                .size(300.dp)
                .align(Alignment.CenterEnd)
                .offset(x = 80.dp, y = (-20).dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(DeepIndigo.copy(alpha = 0.065f), Color.Transparent)
                    ),
                    shape = CircleShape
                )
        )
        Box(
            modifier = Modifier
                .size(260.dp)
                .align(Alignment.BottomCenter)
                .offset(y = 40.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(DeepIndigo.copy(alpha = 0.08f), Color.Transparent)
                    ),
                    shape = CircleShape
                )
        )

        val imeBottom = WindowInsets.ime.getBottom(LocalDensity.current)
        val contentPadding = PaddingValues(bottom = if (imeBottom == 0) 96.dp else 0.dp)
        content(contentPadding)
        AnimatedVisibility(
            visible = imeBottom == 0,
            enter = fadeIn(tween(150)),
            exit = fadeOut(tween(120)),
            modifier = Modifier.align(Alignment.BottomCenter)
        ) {
            LiquidGlassBottomBar(
                activeTab = activeTab,
                onSelectTab = onSelectTab,
                modifier = Modifier
                    .navigationBarsPadding()
                    .padding(horizontal = 16.dp, vertical = 10.dp)
            )
        }
    }
}

@Composable
fun LiquidGlassCard(
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 22.dp,
    accentColor: Color = CyberCyan,
    onClick: (() -> Unit)? = null,
    enabled: Boolean = true,
    content: @Composable BoxScope.() -> Unit,
) {
    val shape = RoundedCornerShape(cornerRadius)
    val borderBrush = Brush.verticalGradient(
        colors = listOf(
            Color.White.copy(alpha = 0.20f),
            accentColor.copy(alpha = 0.11f),
            Color.White.copy(alpha = 0.045f),
        )
    )
    val glassFill = Brush.verticalGradient(
        colors = listOf(
            SurfaceElevated.copy(alpha = 0.86f),
            SurfaceCard.copy(alpha = 0.76f),
        )
    )

    val clickableMod = if (onClick != null) {
        Modifier.sizeIn(minHeight = 48.dp).clickable(enabled = enabled, role = Role.Button, onClick = onClick)
    } else {
        Modifier
    }

    Box(
        modifier = modifier
            .shadow(8.dp, shape, clip = false, ambientColor = Color.Black.copy(alpha = 0.12f), spotColor = Color.Black.copy(alpha = 0.18f))
            .clip(shape)
            .background(glassFill)
            .border(1.dp, borderBrush, shape)
            .then(clickableMod)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(
                    Brush.horizontalGradient(
                        listOf(
                            Color.Transparent,
                            Color.White.copy(alpha = 0.24f),
                            accentColor.copy(alpha = 0.16f),
                            Color.Transparent,
                        )
                    )
                )
        )
        content()
    }
}

@Composable
fun LiquidGlassBottomBar(
    activeTab: GlassBottomTab,
    onSelectTab: (GlassBottomTab) -> Unit,
    modifier: Modifier = Modifier,
) {
    val outerShape = RoundedCornerShape(28.dp)
    Box(
        modifier = modifier
            .widthIn(max = 560.dp)
            .fillMaxWidth()
            .shadow(16.dp, outerShape, clip = false, ambientColor = Color.Black.copy(alpha = 0.18f), spotColor = Color.Black.copy(alpha = 0.22f))
            .clip(outerShape)
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(0xF0242D3D),
                        Color(0xF2121925),
                    )
                )
            )
            .border(
                width = 1.dp,
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        Color.White.copy(alpha = 0.20f),
                        CyberCyan.copy(alpha = 0.11f),
                        Color.White.copy(alpha = 0.045f),
                    )
                ),
                shape = outerShape
            )
            .padding(horizontal = 8.dp, vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            GlassBottomTab.entries.forEach { tab ->
                val isSelected = tab == activeTab
                val itemShape = RoundedCornerShape(20.dp)
                val bgBrush = if (isSelected) {
                    Brush.verticalGradient(
                        colors = listOf(
                            Color.White.copy(alpha = 0.10f),
                            Color.White.copy(alpha = 0.035f),
                        )
                    )
                } else {
                    Brush.verticalGradient(
                        colors = listOf(
                            Color.White.copy(alpha = 0.04f),
                            Color.Transparent,
                        )
                    )
                }
                val itemBorder = if (isSelected) {
                    Brush.verticalGradient(
                        colors = listOf(
                            Color.White.copy(alpha = 0.20f),
                            tab.accentColor.copy(alpha = 0.16f),
                        )
                    )
                } else {
                    Brush.verticalGradient(
                        colors = listOf(
                            Color.White.copy(alpha = 0.08f),
                            Color.Transparent,
                        )
                    )
                }

                Box(
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 56.dp)
                        .clip(itemShape)
                        .background(bgBrush)
                        .border(1.dp, itemBorder, itemShape)
                        .clickable(role = Role.Tab) { onSelectTab(tab) }
                        .padding(vertical = 8.dp, horizontal = 6.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(5.dp)
                        ) {
                            Text(text = tab.iconSymbol, fontSize = 15.sp)
                            Text(
                                text = tab.title,
                                style = MaterialTheme.typography.labelLarge,
                                color = if (isSelected) TextPrimary else TextSecondary,
                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.SemiBold,
                                fontSize = 12.sp
                            )
                        }
                        Text(
                            text = tab.subtitle,
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isSelected) CyberCyanLight else TextTertiary,
                            fontSize = 10.sp,
                            fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal
                        )
                    }
                }
            }
        }
    }
}

/**
 * High-tech spinning liquid glass running circle thinking indicator.
 * Combines an infinite sweep gradient ring with an orbital glow particle
 * and a pulsing core orb, symbolizing real-time neural prefill & forensic analysis.
 */
@Composable
fun RunningCircleIndicator(
    modifier: Modifier = Modifier,
    size: Dp = 22.dp,
    strokeWidth: Dp = 2.5.dp,
    primaryColor: Color = CyberCyan,
    secondaryColor: Color = NeonPurple,
) {
    val infiniteTransition = rememberInfiniteTransition(label = "running_circle")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(1100, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "rotation"
    )
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.30f,
        targetValue = 0.95f,
        animationSpec = infiniteRepeatable(
            animation = tween(850, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulseAlpha"
    )
    val coreScale by infiniteTransition.animateFloat(
        initialValue = 0.20f,
        targetValue = 0.36f,
        animationSpec = infiniteRepeatable(
            animation = tween(850, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "coreScale"
    )

    Canvas(modifier = modifier.size(size).semantics { progressBarRangeInfo = ProgressBarRangeInfo.Indeterminate }) {
        val strokePx = strokeWidth.toPx()
        val diameter = this.size.minDimension
        val ringRadius = (diameter - strokePx) / 2f
        val centerOffset = Offset(this.size.width / 2f, this.size.height / 2f)

        val sweepBrush = Brush.sweepGradient(
            colors = listOf(
                primaryColor.copy(alpha = 0.05f),
                primaryColor.copy(alpha = 0.35f),
                secondaryColor,
                primaryColor,
            ),
            center = centerOffset
        )
        // Center pulsing neural core
        drawCircle(
            color = primaryColor.copy(alpha = pulseAlpha),
            radius = ringRadius * coreScale,
            center = centerOffset
        )
        // Rotating outer gradient ring + orbiting glowing particle
        rotate(rotation, pivot = centerOffset) {
            drawCircle(
                brush = sweepBrush,
                radius = ringRadius,
                center = centerOffset,
                style = Stroke(width = strokePx, cap = StrokeCap.Round)
            )
            // Orbiting particle tip at the head of the sweep gradient
            val radAngle = 0.0
            val particleX = centerOffset.x + ringRadius * cos(radAngle).toFloat()
            val particleY = centerOffset.y + ringRadius * sin(radAngle).toFloat()
            drawCircle(
                color = Color.White,
                radius = strokePx * 0.95f,
                center = Offset(particleX, particleY)
            )
        }
    }
}

/**
 * Compact frosted liquid glass pill container for badges, citations, or status chips.
 */
@Composable
fun LiquidGlassPill(
    modifier: Modifier = Modifier,
    accentColor: Color = CyberCyan,
    cornerRadius: Dp = 14.dp,
    onClick: (() -> Unit)? = null,
    content: @Composable RowScope.() -> Unit,
) {
    val shape = RoundedCornerShape(cornerRadius)
    val clickableMod = if (onClick != null) Modifier.sizeIn(minHeight = 48.dp).clickable(role = Role.Button, onClick = onClick) else Modifier
    Box(
        modifier = modifier
            .clip(shape)
            .background(
                Brush.verticalGradient(
                    listOf(
                        SurfaceElevated.copy(alpha = 0.85f),
                        SurfaceCard.copy(alpha = 0.72f)
                    )
                )
            )
            .border(
                1.dp,
                Brush.horizontalGradient(
                    listOf(
                        Color.White.copy(alpha = 0.18f),
                        accentColor.copy(alpha = 0.13f),
                        Color.White.copy(alpha = 0.05f)
                    )
                ),
                shape
            )
            .then(clickableMod)
            .padding(horizontal = 9.dp, vertical = 5.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            content()
        }
    }
}

