package com.onndetect.ui.screens.result

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.local.SignalBreakdownJson
import com.onndetect.domain.pipeline.CascadePolicy
import com.onndetect.ui.components.*
import com.onndetect.ui.theme.*
import org.json.JSONObject
import java.text.DateFormat
import java.util.Date

@Composable
fun ResultScreen(navController: NavController, viewModel: ResultViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BackgroundVoid)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Result Top Bar
        ResultTopBar(onBackClick = { navController.popBackStack() })

        when (val current = state) {
            ResultState.Loading -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        CircularProgressIndicator(color = CyberCyan)
                        Text(
                            text = "Loading forensic assessment...",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextSecondary
                        )
                    }
                }
            }
            ResultState.NotFound -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(SurfaceCard)
                        .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
                        .padding(24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = stringResource(R.string.result_not_found),
                            style = MaterialTheme.typography.titleMedium,
                            color = TextPrimary
                        )
                        Button(
                            onClick = { navController.popBackStack() },
                            colors = ButtonDefaults.buttonColors(containerColor = SurfaceElevated)
                        ) {
                            Text(stringResource(R.string.back), color = CyberCyan)
                        }
                    }
                }
            }
            is ResultState.Error -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(SyntheticRedContainer.copy(alpha = 0.5f))
                        .border(1.dp, SyntheticRed.copy(alpha = 0.5f), RoundedCornerShape(16.dp))
                        .padding(20.dp)
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text(
                            text = stringResource(R.string.history_read_failed),
                            style = MaterialTheme.typography.titleMedium,
                            color = SyntheticRed
                        )
                        Button(
                            onClick = viewModel::reload,
                            colors = ButtonDefaults.buttonColors(containerColor = SyntheticRed)
                        ) {
                            Text(stringResource(R.string.retry), color = Color.White)
                        }
                    }
                }
            }
            is ResultState.Ready -> {
                AssessmentDetails(
                    e = current.entry,
                    onDoneClick = { navController.popBackStack() },
                    onHistoryClick = { navController.navigate("history") }
                )
            }
        }
    }
}

@Composable
fun AssessmentDetails(
    e: ScanHistoryEntity,
    onDoneClick: () -> Unit = {},
    onHistoryClick: () -> Unit = {}
) {
    val context = LocalContext.current
    // 1. Verdict Banner
    ForensicVerdictCard(
        verdict = e.verdict,
        validated = e.modelValidated,
        score = e.score,
        reason = e.reason
    )

    // 2. Confidence Gauge
    val evidence = remember(e.evidenceJson) {
        runCatching { JSONObject(e.evidenceJson ?: "{}") }.getOrDefault(JSONObject())
    }
    val threshold = remember(evidence) {
        evidence.optDouble("threshold", 0.499).toFloat()
    }
    ConfidenceMeter(score = e.score, threshold = threshold)

    // 2.5. Dual-Engine Cascade Telemetry Card (When visual dual-engine ran)
    val decodedSignals = remember(e.signalsJson) { SignalBreakdownJson.decode(e.signalsJson) }
    // The cascade thresholds must come from the RELEASE, not from
    // constants baked into the UI. Hard-coded thresholds silently drift
    // away from the shipped policy and make the card disagree with the
    // headline verdict.
    val cascadePolicy = remember(context) { CascadePolicy.fromAssets(context) }
    if (decodedSignals != null && decodedSignals.mobileClipScore.isFinite() && decodedSignals.mobileClipScore >= 0f) {
        DualEngineCascadeCard(
            // The pipeline arbitrates on the FUSION score, not on the raw
            // SRM CNN output. Showing srmForensicScore here made the card
            // contradict the headline verdict whenever the two fell on
            // opposite sides of the threshold.
            engineAScore = decodedSignals.fusionScore,
            engineCScore = decodedSignals.mobileClipScore,
            arbitratedScore = e.score,
            reason = e.reason,
            engineAThreshold = cascadePolicy?.engineAThreshold ?: 0.5f,
            engineCThreshold = cascadePolicy?.engineCThreshold ?: 0.5f,
            escalateBelow = cascadePolicy?.escalateBelow ?: 0f,
            escalateAbove = cascadePolicy?.escalateAbove ?: 1f
        )
    }

    // 3. Multi-Signal Decomposition
    decodedSignals?.let {
        SignalBreakdownCard(signals = it)
    }

    // 4. Forensic Evidence & Metadata Container
    EvidenceMetadataCard(e = e, evidence = evidence)

    // 5. Limitations Notice
    LimitationsCard()

    // 6. Action Buttons
    Column(
        verticalArrangement = Arrangement.spacedBy(10.dp),
        modifier = Modifier.padding(top = 8.dp)
    ) {
        Button(
            onClick = onDoneClick,
            colors = ButtonDefaults.buttonColors(
                containerColor = CyberCyan,
                contentColor = Color(0xFF070A10)
            ),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp)
        ) {
            Text(
                text = stringResource(R.string.done),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
        }

        OutlinedButton(
            onClick = onHistoryClick,
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = TextSecondary
            ),
            border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp)
        ) {
            Text(
                text = "View in Audit History",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
private fun EvidenceMetadataCard(e: ScanHistoryEntity, evidence: JSONObject) {
    val modelVersion = evidence.optString("modelVersion", "onn-20261020-v6")
    val minEdge = evidence.optInt("resolutionMinEdge", 0)
    val isBlurry = evidence.optBoolean("blurry", false)
    val isDark = evidence.optBoolean("dark", false)
    val provenance = evidence.optString("provenance", "not verified")

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
            .padding(20.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            // Header Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "FORENSIC TELEMETRY",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary,
                    fontWeight = FontWeight.SemiBold
                )

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(SurfaceHighlight)
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = modelVersion,
                        style = MaterialTheme.typography.labelSmall,
                        color = CyberCyan,
                        fontWeight = FontWeight.Medium
                    )
                }
            }

            // Quality alert banner if blurry or dark
            if (isBlurry || isDark) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(WarningAmberContainer.copy(alpha = 0.5f))
                        .border(1.dp, WarningAmber.copy(alpha = 0.5f), RoundedCornerShape(10.dp))
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text("⚠️", fontSize = 14.sp)
                    Text(
                        text = stringResource(R.string.quality_warning),
                        style = MaterialTheme.typography.bodySmall,
                        color = WarningAmber
                    )
                }
            }

            // Grid rows
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                MetadataRow(
                    label = "Modality",
                    value = e.scanMode.replace('_', ' ')
                )
                MetadataRow(
                    label = "Processing Time",
                    value = stringResource(R.string.processing_duration, e.durationMs)
                )
                MetadataRow(
                    label = "Scan Timestamp",
                    value = DateFormat.getDateTimeInstance().format(Date(e.timestamp))
                )

                if (minEdge > 0) {
                    MetadataRow(
                        label = "Shorter Edge Resolution",
                        value = "${minEdge} px"
                    )
                }

                if (e.scanMode == "AUDIO_ONLY" && evidence.optLong("totalDurationMs") > 0) {
                    MetadataRow(
                        label = "Audio Coverage",
                        value = stringResource(
                            R.string.coverage_audio,
                            evidence.optLong("analyzedDurationMs") / 1000f,
                            evidence.optLong("totalDurationMs") / 1000f
                        )
                    )
                }

                if (e.scanMode == "VIDEO_ONLY") {
                    if (evidence.optLong("analyzedDurationMs") > 0) {
                        MetadataRow(
                            label = "Speech Coverage",
                            value = stringResource(
                                R.string.coverage_video_audio,
                                evidence.optLong("analyzedDurationMs") / 1000f
                            )
                        )
                    }
                    val frameArray = evidence.optJSONArray("timestampsMs")
                    if (frameArray != null && frameArray.length() > 0) {
                        MetadataRow(
                            label = "Video Frame Coverage",
                            value = pluralStringResource(
                                R.plurals.coverage_frames,
                                frameArray.length(),
                                frameArray.length()
                            )
                        )
                    }
                }

                MetadataRow(
                    label = "Provenance (C2PA)",
                    value = provenance
                )
            }
        }
    }
}

@Composable
private fun MetadataRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(SurfaceElevated)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = TextTertiary
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            color = TextPrimary,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
private fun LimitationsCard() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceElevated)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("ℹ", color = CyberCyan, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                Text(
                    text = "FORENSIC NOTICE & LIMITATIONS",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyberCyan,
                    letterSpacing = 1.sp
                )
            }
            Text(
                text = stringResource(R.string.result_limitations),
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                lineHeight = 18.sp
            )
            Text(
                text = stringResource(R.string.provenance_unverified),
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary,
                lineHeight = 16.sp
            )
        }
    }
}

@Composable
private fun ResultTopBar(onBackClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(42.dp)
                .clip(CircleShape)
                .background(SurfaceCard)
                .border(1.dp, BorderSubtle, CircleShape)
                .clickable(onClick = onBackClick)
        ) {
            Text(
                text = "←",
                color = TextPrimary,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = stringResource(R.string.result_title).uppercase(),
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.2.sp
            )
            Text(
                text = "Local Forensic Analysis Report",
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary
            )
        }
    }
}

@Composable
fun DualEngineCascadeCard(
    engineAScore: Float,
    engineCScore: Float,
    arbitratedScore: Float,
    reason: String?,
    engineAThreshold: Float,
    engineCThreshold: Float,
    escalateBelow: Float,
    escalateAbove: Float,
    modifier: Modifier = Modifier
) {
    if (!engineAScore.isFinite() || !engineCScore.isFinite()) return
    if (!engineAThreshold.isFinite() || !engineCThreshold.isFinite()) return

    val aSaysFake = engineAScore >= engineAThreshold
    val cSaysFake = engineCScore >= engineCThreshold
    val inAmbiguityBand = engineCScore in escalateBelow..escalateAbove
    val isAgreement = aSaysFake == cSaysFake

    val (badgeText, badgeColor, badgeDesc) = when {
        inAmbiguityBand && isAgreement -> Triple(
            "ARBITRATED AGREEMENT",
            AuthenticGreen,
            "Ambiguity band arbitrated: Engine A & Engine C concordantly confirm verdict."
        )
        inAmbiguityBand && !isAgreement -> Triple(
            "DISAGREEMENT (INCONCLUSIVE)",
            WarningAmber,
            "Engine C ambiguous and Engine A contradicts: honest abstention applied."
        )
        else -> Triple(
            if (cSaysFake) "SYNTHETIC CONCURRENCE" else "AUTHENTIC CONCURRENCE",
            if (cSaysFake) SyntheticRed else AuthenticGreen,
            "Engine C confident semantic measurement with dual-engine feature alignment."
        )
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderCyanGlow, RoundedCornerShape(20.dp))
            .padding(20.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            // Header Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text("⚡", fontSize = 16.sp)
                    Text(
                        text = "DUAL-ENGINE CASCADE",
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary,
                        fontWeight = FontWeight.Bold
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(CyberCyan.copy(alpha = 0.15f))
                        .border(1.dp, CyberCyan.copy(alpha = 0.3f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = "onn-20261020-v6",
                        style = MaterialTheme.typography.labelSmall,
                        color = CyberCyan,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // Dual Engine Progress Columns
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Engine A Card - shows the FUSION score, which is what the cascade
                // policy actually arbitrates on.
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(12.dp))
                        .background(SurfaceElevated)
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "ENGINE A (FUSION)",
                            style = MaterialTheme.typography.labelSmall,
                            color = CyberCyan,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = String.format("%.3f", engineAScore),
                            style = MaterialTheme.typography.labelMedium,
                            color = TextPrimary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Text(
                        text = "FUSION (SRM + FFT + KL + Engine C)",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary,
                        fontSize = 10.sp
                    )
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(4.dp)
                            .clip(CircleShape)
                            .background(SurfaceHighlight)
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(engineAScore.coerceIn(0f, 1f))
                                .fillMaxHeight()
                                .clip(CircleShape)
                                .background(if (engineAScore >= engineAThreshold) SyntheticRed else AuthenticGreen)
                        )
                    }
                }

                // Engine C Card
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(12.dp))
                        .background(SurfaceElevated)
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "ENGINE C",
                            style = MaterialTheme.typography.labelSmall,
                            color = NeonPurple,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = String.format("%.3f", engineCScore),
                            style = MaterialTheme.typography.labelMedium,
                            color = TextPrimary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Text(
                        text = "SigLIP Appearance CNN",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary,
                        fontSize = 10.sp
                    )
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(4.dp)
                            .clip(CircleShape)
                            .background(SurfaceHighlight)
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(engineCScore.coerceIn(0f, 1f))
                                .fillMaxHeight()
                                .clip(CircleShape)
                                .background(if (engineCScore >= engineCThreshold) SyntheticRed else AuthenticGreen)
                        )
                    }
                }
            }

            // Arbitration Badge & Telemetry
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(badgeColor.copy(alpha = 0.12f))
                    .border(1.dp, badgeColor.copy(alpha = 0.35f), RoundedCornerShape(10.dp))
                    .padding(12.dp)
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = badgeText,
                        style = MaterialTheme.typography.labelMedium,
                        color = badgeColor,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = reason ?: badgeDesc,
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                        fontSize = 11.sp,
                        lineHeight = 15.sp
                    )
                }
            }
        }
    }
}
