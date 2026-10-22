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
                    onHistoryClick = { navController.navigate("history") },
                    onExportReceipt = { viewModel.generateReceipt(current.entry) }
                )
            }
        }
    }
}

@Composable
fun AssessmentDetails(
    e: ScanHistoryEntity,
    onDoneClick: () -> Unit = {},
    onHistoryClick: () -> Unit = {},
    onExportReceipt: () -> com.onndetect.detection.provenance.ForensicReceipt = {
        com.onndetect.detection.provenance.ForensicReceipt("", "", "", "TEE")
    }
) {
    val context = LocalContext.current
    // 1. Verdict Banner
    val evidence = remember(e.evidenceJson) {
        runCatching { JSONObject(e.evidenceJson ?: "{}") }.getOrDefault(JSONObject())
    }
    // Whether an ON-DEVICE evaluation verified the release. A release that
    // is only desktop-verified is usable, but the card must say so rather
    // than presenting it as fully device-verified.
    val deviceValidated = remember(e.evidenceJson) {
        runCatching { evidence.optBoolean("deviceValidated", false) }.getOrDefault(false)
    }
    ForensicVerdictCard(
        verdict = e.verdict,
        validated = e.modelValidated,
        score = e.score,
        reason = e.reason,
        deviceValidated = deviceValidated,
        releaseId = remember(e.evidenceJson) {
            runCatching { evidence.optString("modelVersion", "unknown") }.getOrDefault("unknown")
        },
        scanMode = e.scanMode
    )

    // 2. Confidence Gauge
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
    if (decodedSignals != null && decodedSignals.vfmProbeScore.isFinite() && decodedSignals.vfmProbeScore >= 0f) {
        DualEngineCascadeCard(
            // The pipeline arbitrates on the FUSION score, not on the raw
            // SRM CNN output. Showing srmForensicScore here made the card
            // contradict the headline verdict whenever the two fell on
            // opposite sides of the threshold.
            engineAScore = decodedSignals.fusionScore,
            engineCScore = decodedSignals.vfmProbeScore,
            arbitratedScore = e.score,
            reason = e.reason,
            engineAThreshold = cascadePolicy?.engineAThreshold ?: 0.5f,
            engineCThreshold = cascadePolicy?.engineCThreshold ?: 0.5f,
            escalateBelow = cascadePolicy?.escalateBelow ?: 0f,
            escalateAbove = cascadePolicy?.escalateAbove ?: 1f,
            releaseId = cascadePolicy?.releaseId ?: "unknown"
        )
    }

    // 3. Multi-Signal Decomposition
    decodedSignals?.let {
        SignalBreakdownCard(signals = it)
    }

    // 3.5. Spatial Residual Energy Heatmap (When visual heatmap was generated)
    val heatmap = remember(evidence) {
        evidence?.optString("heatmap")?.takeIf { it.isNotEmpty() }?.split(",")?.mapNotNull { it.toFloatOrNull() }?.toFloatArray()
    }
    if (heatmap != null && heatmap.isNotEmpty()) {
        HeatmapViewer(heatmap = heatmap)
    }

    // 4. Forensic Evidence & Metadata Container
    EvidenceMetadataCard(e = e, evidence = evidence)

    // 4.25. Document & Payment Screenshot Forgery Card
    val docForgery = remember(evidence) { evidence.optJSONObject("documentForgery") }
    if (docForgery != null && docForgery.optBoolean("isDocumentOrScreenshot", false)) {
        DocumentForgeryCard(docForgery = docForgery)
    }

    // 4.35. Video-KYC (V-CIP) Virtual Camera & Injection Shield Card
    val kycInjection = remember(evidence) { evidence.optJSONObject("kycInjection") }
    if (kycInjection != null) {
        VirtualCameraShieldCard(kycInjection = kycInjection)
    }

    // 4.5. Hardware Forensic Receipt Card (StrongBox / TEE)
    ForensicReceiptCard(e = e, evidence = evidence, onExportReceipt = onExportReceipt)

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
    // The release id comes from the assessment's own evidence, never from a
    // hard-coded string: the shipped release id changes with every release,
    // and a stale constant labels every assessment with the wrong release.
    val modelVersion = evidence.optString("modelVersion", "unknown")
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
    /**
     * The release id, from the shipped cascade policy. It was hard-coded,
     * so the card named a release that had been superseded.
     */
    releaseId: String = "unknown",
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
                        text = releaseId,
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
                        text = "SigLIP-B/16 vision tower",
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

@Composable
private fun ForensicReceiptCard(
    e: ScanHistoryEntity,
    evidence: JSONObject,
    onExportReceipt: () -> com.onndetect.detection.provenance.ForensicReceipt
) {
    val context = LocalContext.current
    var copied by remember { mutableStateOf(false) }
    // The receipt is generated ONCE for the card, so the rows below report
    // the receipt that would actually be exported - including whether it
    // carries a device attestation chain.
    val receipt = remember(e.id, e.evidenceJson) { runCatching { onExportReceipt() }.getOrNull() }
    val securityTier = receipt?.securityTier ?: "unavailable"
    val attested = receipt != null && receipt.certificateChainBase64.isNotEmpty()

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
            .padding(20.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text("🛡️", fontSize = 16.sp)
                    Text(
                        text = "HARDWARE FORENSIC RECEIPT",
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(CyberCyan.copy(alpha = 0.15f))
                        .border(1.dp, CyberCyan.copy(alpha = 0.4f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = "ECDSA P-256",
                        style = MaterialTheme.typography.labelSmall,
                        color = CyberCyan,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Text(
                text = "Cryptographically signed inside the Android Hardware Keystore enclave. Binds the verdict, the media SHA-256 digest, and the frozen model release hashes for enterprise audit compliance.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                lineHeight = 18.sp
            )

            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                MetadataRow(
                    label = "Attestation Standard",
                    value = "RFC 8785 (onn-receipt-v1)"
                )
                MetadataRow(
                    label = "Hardware Root",
                    value = if (attested) {
                        "Google Hardware Keystore Root CA (chain present)"
                    } else {
                        "NOT ATTESTED - no device chain"
                    }
                )
                MetadataRow(
                    label = "Security Tier",
                    value = securityTier
                )
                MetadataRow(
                    label = "Media Binding",
                    value = "SHA-256 Bound"
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                OutlinedButton(
                    onClick = {
                        val export = receipt ?: return@OutlinedButton
                        val clipboard = context.getSystemService(android.content.Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                        clipboard.setPrimaryClip(android.content.ClipData.newPlainText("ONN-DETECT Forensic Receipt", export.toExportJson()))
                        copied = true
                    },
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(10.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = CyberCyan)
                ) {
                    Text(if (copied) "Copied!" else "Copy JSON", fontSize = 13.sp)
                }

                Button(
                    onClick = {
                        val export = receipt ?: return@Button
                        val sendIntent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                            type = "application/json"
                            putExtra(android.content.Intent.EXTRA_TEXT, export.toExportJson())
                            putExtra(android.content.Intent.EXTRA_SUBJECT, "ONN-DETECT Forensic Receipt (${e.id.take(8)})")
                        }
                        context.startActivity(android.content.Intent.createChooser(sendIntent, "Export Forensic Receipt"))
                    },
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = SurfaceElevated, contentColor = TextPrimary)
                ) {
                    Text("Export Audit File", fontSize = 13.sp)
                }
            }
        }
    }
}

@Composable
private fun DocumentForgeryCard(docForgery: JSONObject) {
    val isForged = docForgery.optBoolean("isForged", false)
    val score = docForgery.optDouble("forgeryScore", 0.0).toFloat()
    val ela = docForgery.optDouble("elaPeakRatio", 1.0).toFloat()
    val noise = docForgery.optDouble("noiseDiscontinuity", 0.0).toFloat()
    val edge = docForgery.optDouble("edgeInconsistency", 0.0).toFloat()
    val layoutAnomaly = docForgery.optBoolean("layoutAnomaly", false)
    val indicatorsArray = docForgery.optJSONArray("indicators")
    val indicators = remember(indicatorsArray) {
        val list = mutableListOf<String>()
        if (indicatorsArray != null) {
            for (i in 0 until indicatorsArray.length()) {
                list.add(indicatorsArray.getString(i))
            }
        }
        list
    }

    val statusColor = if (isForged) SyntheticRed else CyberCyan
    val statusBg = if (isForged) SyntheticRedContainer.copy(alpha = 0.3f) else CyberCyan.copy(alpha = 0.1f)
    val statusBorder = if (isForged) SyntheticRed.copy(alpha = 0.6f) else CyberCyan.copy(alpha = 0.4f)

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
            .padding(20.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(if (isForged) "⚠️" else "📄", fontSize = 16.sp)
                    Text(
                        text = "DOCUMENT & PAYMENT AUDIT",
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(statusBg)
                        .border(1.dp, statusBorder, RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = if (isForged) {
                        stringResource(R.string.document_forgery_tampering_indicated)
                    } else {
                        stringResource(R.string.document_forgery_none_detected)
                    },
                        style = MaterialTheme.typography.labelSmall,
                        color = statusColor,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Text(
                text = if (isForged)
                    "Discontinuities in JPEG compression error levels and high-pass noise floor indicate copy-pasted or edited text (e.g. modified payment amount, UTR, or identity details)."
                else
                    "Spatial noise floor, error levels, and typography antialiasing are consistent across text blocks with no digital splicing detected.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                lineHeight = 18.sp
            )

            // Forgery Likelihood Progress Bar
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Forgery Likelihood",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary
                    )
                    Text(
                        text = "${(score * 100).toInt()}%",
                        style = MaterialTheme.typography.labelSmall,
                        color = statusColor,
                        fontWeight = FontWeight.Bold
                    )
                }
                LinearProgressIndicator(
                    progress = { score.coerceIn(0f, 1f) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp)),
                    color = statusColor,
                    trackColor = SurfaceElevated
                )
            }

            // Forensic Telemetry Rows
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                MetadataRow(
                    label = "Error Level Analysis (ELA)",
                    value = "Peak Ratio %.2fx".format(java.util.Locale.US, ela)
                )
                MetadataRow(
                    label = "Spatial Noise Uniformity",
                    value = if (noise > 0.5f) "Outlier Variance Detected" else "Uniform Noise Floor"
                )
                MetadataRow(
                    label = "Boundary Discontinuity",
                    value = if (edge > 0.55f) "Splicing Edges Found" else "Continuous Gradients"
                )
                MetadataRow(
                    label = "Viewport Geometry",
                    value = if (layoutAnomaly) "Non-standard Aspect Ratio" else "Standard Mobile Display"
                )
            }

            // Flagged indicators list
            if (indicators.isNotEmpty()) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(if (isForged) SyntheticRedContainer.copy(alpha = 0.25f) else SurfaceElevated)
                        .padding(12.dp)
                ) {
                    Text(
                        text = "FORENSIC FINDINGS:",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (isForged) SyntheticRed else CyberCyan,
                        fontWeight = FontWeight.Bold
                    )
                    indicators.forEach { item ->
                        Text(
                            text = "• $item",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextPrimary,
                            fontSize = 12.sp,
                            lineHeight = 16.sp
                        )
                    }

                    // The caveat is not optional. This detector is an
                    // uncalibrated heuristic, and its clean branch means
                    // "no anomaly detected", not "authentic".
                    Text(
                        text = stringResource(R.string.document_forgery_caveat),
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary,
                        lineHeight = 16.sp
                    )
                }
            }
        }
    }
}

@Composable
private fun VirtualCameraShieldCard(kycInjection: JSONObject) {
    val isGenuine = kycInjection.optBoolean("isHardwareGenuine", true)
    val hookDetected = kycInjection.optBoolean("hookDetected", false)
    val emulatorDetected = kycInjection.optBoolean("emulatorDetected", false)
    val halLevel = kycInjection.optString("cameraHardwareLevel", "UNKNOWN")
    val physicalSensor = kycInjection.optBoolean("physicalSensorVerified", false)
    val score = kycInjection.optDouble("injectionScore", 0.0).toFloat()
    val flagsArray = kycInjection.optJSONArray("flags")
    val flags = remember(flagsArray) {
        val list = mutableListOf<String>()
        if (flagsArray != null) {
            for (i in 0 until flagsArray.length()) {
                list.add(flagsArray.getString(i))
            }
        }
        list
    }

    val statusColor = if (!isGenuine) SyntheticRed else CyberCyan
    val statusBg = if (!isGenuine) SyntheticRedContainer.copy(alpha = 0.3f) else CyberCyan.copy(alpha = 0.1f)
    val statusBorder = if (!isGenuine) SyntheticRed.copy(alpha = 0.6f) else CyberCyan.copy(alpha = 0.4f)

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
            .padding(20.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(if (!isGenuine) "⚠️" else "🛡️", fontSize = 16.sp)
                    Text(
                        text = stringResource(R.string.kyc_virtual_camera_title),
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(statusBg)
                        .border(1.dp, statusBorder, RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = if (!isGenuine) {
                            stringResource(R.string.kyc_tampering_detected)
                        } else {
                            stringResource(R.string.kyc_genuine_camera)
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = statusColor,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Text(
                text = if (!isGenuine)
                    "Camera HAL telemetry, physical sensor metrics, or memory inspection detected a virtual camera driver, dynamic injection framework (Xposed/Pine/VCam), or emulator stream feed."
                else
                    "Camera HAL registers confirm physical front-facing camera silicon with valid hardware shutter bounds, no runtime injection hooks, and genuine hardware execution.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                lineHeight = 18.sp
            )

            // Injection Likelihood Progress Bar
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Injection Likelihood",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary
                    )
                    Text(
                        text = "${(score * 100).toInt()}%",
                        style = MaterialTheme.typography.labelSmall,
                        color = statusColor,
                        fontWeight = FontWeight.Bold
                    )
                }
                LinearProgressIndicator(
                    progress = { score.coerceIn(0f, 1f) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp)),
                    color = statusColor,
                    trackColor = SurfaceElevated
                )
            }

            // Forensic Telemetry Rows
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                MetadataRow(
                    label = "Camera Hardware HAL",
                    value = "Level $halLevel"
                )
                MetadataRow(
                    label = "Silicon Sensor Geometry",
                    value = if (physicalSensor) "Verified Physical Sensor" else "Missing / Virtual Sensor"
                )
                MetadataRow(
                    label = "Dynamic Injection Hooks",
                    value = if (hookDetected) "ACTIVE (Xposed / VCam detected)" else "Clean Memory Space"
                )
                MetadataRow(
                    label = "Execution Environment",
                    value = if (emulatorDetected) "Virtual Emulator (Mock Fingerprint)" else "Physical Device"
                )
            }

            // Flagged indicators list
            if (flags.isNotEmpty()) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(if (!isGenuine) SyntheticRedContainer.copy(alpha = 0.25f) else SurfaceElevated)
                        .padding(12.dp)
                ) {
                    Text(
                        text = "INJECTION ALERTS:",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (!isGenuine) SyntheticRed else CyberCyan,
                        fontWeight = FontWeight.Bold
                    )
                    flags.forEach { item ->
                        Text(
                            text = "• $item",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextPrimary,
                            fontSize = 12.sp,
                            lineHeight = 16.sp
                        )
                    }

                    Text(
                        text = stringResource(R.string.kyc_hardware_caveat),
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary,
                        lineHeight = 16.sp
                    )
                }
            }
        }
    }
}
