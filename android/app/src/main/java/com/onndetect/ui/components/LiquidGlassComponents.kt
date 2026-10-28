package com.onndetect.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.onndetect.ui.navigation.Routes
import com.onndetect.ui.theme.*

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
        subtitle = "Img • Aud • Vid",
        iconSymbol = "🔍",
        accentColor = CyberCyan,
    ),
    CHAT_RAG(
        route = Routes.CHAT_RAG_BASE,
        title = "RAG Chat",
        subtitle = "Qwen 0.8B • VLM",
        iconSymbol = "💬",
        accentColor = NeonPurple,
    ),
    VOICE_VAULT(
        route = Routes.VOICE_VAULT,
        title = "Voice Vault",
        subtitle = "Family Guard",
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
                        colors = listOf(CyberCyan.copy(alpha = 0.16f), Color.Transparent)
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
                        colors = listOf(NeonPurple.copy(alpha = 0.14f), Color.Transparent)
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
                        colors = listOf(DeepIndigo.copy(alpha = 0.18f), Color.Transparent)
                    ),
                    shape = CircleShape
                )
        )

        val contentPadding = PaddingValues(bottom = 104.dp)
        content(contentPadding)

        LiquidGlassBottomBar(
            activeTab = activeTab,
            onSelectTab = onSelectTab,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 10.dp)
        )
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
            Color.White.copy(alpha = 0.30f),
            accentColor.copy(alpha = 0.38f),
            DeepIndigo.copy(alpha = 0.16f),
        )
    )
    val glassFill = Brush.verticalGradient(
        colors = listOf(
            SurfaceElevated.copy(alpha = 0.78f),
            SurfaceCard.copy(alpha = 0.64f),
        )
    )

    val clickableMod = if (onClick != null) {
        Modifier.clickable(enabled = enabled, onClick = onClick)
    } else {
        Modifier
    }

    Box(
        modifier = modifier
            .clip(shape)
            .background(glassFill)
            .border(1.dp, borderBrush, shape)
            .then(clickableMod)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(2.dp)
                .background(
                    Brush.horizontalGradient(
                        listOf(
                            Color.Transparent,
                            Color.White.copy(alpha = 0.35f),
                            accentColor.copy(alpha = 0.45f),
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
            .fillMaxWidth()
            .clip(outerShape)
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(0xD917223B),
                        Color(0xE60B101E),
                    )
                )
            )
            .border(
                width = 1.2.dp,
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        Color.White.copy(alpha = 0.28f),
                        CyberCyan.copy(alpha = 0.40f),
                        NeonPurple.copy(alpha = 0.35f),
                        AuthenticGreen.copy(alpha = 0.35f),
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
                            tab.accentColor.copy(alpha = 0.28f),
                            tab.accentColor.copy(alpha = 0.10f),
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
                            Color.White.copy(alpha = 0.45f),
                            tab.accentColor.copy(alpha = 0.65f),
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
                        .clip(itemShape)
                        .background(bgBrush)
                        .border(1.dp, itemBorder, itemShape)
                        .clickable { onSelectTab(tab) }
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
                            color = if (isSelected) tab.accentColor else TextTertiary,
                            fontSize = 9.sp,
                            fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal
                        )
                    }
                }
            }
        }
    }
}
