package com.onndetect.ui.screens.history

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.ui.components.assessmentLabel
import com.onndetect.ui.theme.*
import java.text.DateFormat
import java.util.Date

@Composable
fun HistoryScreen(navController: NavController, viewModel: HistoryViewModel = hiltViewModel()) {
    val entries by viewModel.entries.collectAsStateWithLifecycle()
    val error by viewModel.error.collectAsStateWithLifecycle()
    var deleteId by rememberSaveable { mutableStateOf<String?>(null) }
    var clearRequested by rememberSaveable { mutableStateOf(false) }
    var selectedFilter by rememberSaveable { mutableStateOf("ALL") }

    val filteredEntries = remember(entries, selectedFilter) {
        when (selectedFilter) {
            "IMAGE" -> entries.filter { it.scanMode == "IMAGE_ONLY" }
            "AUDIO" -> entries.filter { it.scanMode == "AUDIO_ONLY" }
            "VIDEO" -> entries.filter { it.scanMode == "VIDEO_ONLY" }
            else -> entries
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BackgroundVoid)
            .safeDrawingPadding()
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Top Navigation Bar
        HistoryTopBar(
            hasEntries = entries.isNotEmpty(),
            onBackClick = { navController.popBackStack() },
            onClearClick = { clearRequested = true }
        )

        // Error message if any
        error?.let {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(SyntheticRedContainer.copy(alpha = 0.5f))
                    .border(1.dp, SyntheticRed.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                    .padding(12.dp)
            ) {
                Text(
                    text = stringResource(R.string.history_update_failed),
                    color = SyntheticRed,
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }

        // Summary Metric Bar
        HistorySummaryBanner(entries = entries)

        // Modality Filter Chips Row
        ModalityFilterChips(
            entries = entries,
            selectedFilter = selectedFilter,
            onSelectFilter = { selectedFilter = it }
        )

        // List or Empty State
        if (filteredEntries.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center
            ) {
                HistoryEmptyState(isFiltering = entries.isNotEmpty() && filteredEntries.isEmpty())
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(filteredEntries, key = { it.id }) { entry ->
                    HistoryItemCard(
                        entry = entry,
                        onClick = { navController.navigate("result/${entry.id}") },
                        onDelete = { deleteId = entry.id }
                    )
                }
            }
        }
    }

    // Confirmation Alert Dialog for Deletion
    if (clearRequested || deleteId != null) {
        AlertDialog(
            onDismissRequest = {
                clearRequested = false
                deleteId = null
            },
            containerColor = SurfaceCard,
            title = {
                Text(
                    text = stringResource(R.string.delete_title),
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
            },
            text = {
                Text(
                    text = stringResource(R.string.delete_notice),
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextSecondary
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (clearRequested) viewModel.clearAll() else deleteId?.let(viewModel::delete)
                        clearRequested = false
                        deleteId = null
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = SyntheticRed)
                ) {
                    Text(
                        text = stringResource(R.string.delete),
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                }
            },
            dismissButton = {
                TextButton(onClick = {
                    clearRequested = false
                    deleteId = null
                }) {
                    Text(
                        text = stringResource(R.string.cancel),
                        color = TextSecondary
                    )
                }
            }
        )
    }
}

@Composable
private fun HistoryTopBar(
    hasEntries: Boolean,
    onBackClick: () -> Unit,
    onClearClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(
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
                Text("←", color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            }

            Column {
                Text(
                    text = stringResource(R.string.history_title).uppercase(),
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.2.sp
                )
                Text(
                    text = "On-Device Forensic Records",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }
        }

        if (hasEntries) {
            TextButton(
                onClick = onClearClick,
                colors = ButtonDefaults.textButtonColors(contentColor = SyntheticRed.copy(alpha = 0.9f))
            ) {
                Text(
                    text = stringResource(R.string.clear_history),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }
    }
}

@Composable
private fun HistorySummaryBanner(entries: List<ScanHistoryEntity>) {
    val syntheticCount = entries.count { it.verdict == "SYNTHETIC" }
    val authenticCount = entries.count { it.verdict == "AUTHENTIC" }
    val inconclusiveCount = entries.size - syntheticCount - authenticCount

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            // Stats Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                StatBadge(count = entries.size, label = "TOTAL SCANS", color = CyberCyan)
                StatBadge(count = syntheticCount, label = "SYNTHETIC", color = SyntheticRed)
                StatBadge(count = authenticCount, label = "AUTHENTIC", color = AuthenticGreen)
                StatBadge(count = inconclusiveCount, label = "INCONCLUSIVE", color = WarningAmber)
            }

            // Storage statement
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .background(BackgroundVoid.copy(alpha = 0.5f))
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(AuthenticGreen)
                )
                Text(
                    text = stringResource(R.string.history_policy),
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary,
                    lineHeight = 14.sp
                )
            }
        }
    }
}

@Composable
private fun StatBadge(count: Int, label: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = count.toString(),
            style = MaterialTheme.typography.titleLarge,
            color = color,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = TextTertiary,
            fontSize = 9.sp
        )
    }
}

@Composable
private fun ModalityFilterChips(
    entries: List<ScanHistoryEntity>,
    selectedFilter: String,
    onSelectFilter: (String) -> Unit
) {
    val filters = listOf(
        Triple("ALL", "All", entries.size),
        Triple("IMAGE", "Images", entries.count { it.scanMode == "IMAGE_ONLY" }),
        Triple("AUDIO", "Speech", entries.count { it.scanMode == "AUDIO_ONLY" }),
        Triple("VIDEO", "Videos", entries.count { it.scanMode == "VIDEO_ONLY" })
    )

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        filters.forEach { (key, label, count) ->
            val isSelected = selectedFilter == key
            val containerColor = if (isSelected) CyberCyan.copy(alpha = 0.15f) else SurfaceCard
            val borderColor = if (isSelected) CyberCyan else BorderSubtle
            val textColor = if (isSelected) CyberCyan else TextSecondary

            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(10.dp))
                    .background(containerColor)
                    .border(1.dp, borderColor, RoundedCornerShape(10.dp))
                    .clickable { onSelectFilter(key) }
                    .padding(vertical = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = label,
                        style = MaterialTheme.typography.labelMedium,
                        color = textColor,
                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium
                    )
                    Text(
                        text = count.toString(),
                        style = MaterialTheme.typography.labelSmall,
                        color = if (isSelected) CyberCyan.copy(alpha = 0.8f) else TextTertiary,
                        fontSize = 10.sp
                    )
                }
            }
        }
    }
}

@Composable
private fun HistoryItemCard(
    entry: ScanHistoryEntity,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    val (iconText, iconColor) = when (entry.scanMode) {
        "IMAGE_ONLY" -> Pair("🖼️", CyberCyan)
        "AUDIO_ONLY" -> Pair("🎙️", NeonPurple)
        "VIDEO_ONLY" -> Pair("🎬", ElectricBlue)
        else -> Pair("🔍", TextSecondary)
    }

    val (verdictColor, verdictBg) = when {
        entry.modelValidated && entry.verdict == "SYNTHETIC" -> Pair(SyntheticRed, SyntheticRedContainer)
        entry.modelValidated && entry.verdict == "AUTHENTIC" -> Pair(AuthenticGreen, AuthenticGreenContainer)
        else -> Pair(WarningAmber, WarningAmberContainer)
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .clickable(onClick = onClick)
            .padding(14.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Modality Icon Box
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(46.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(SurfaceElevated)
                    .border(1.dp, iconColor.copy(alpha = 0.4f), RoundedCornerShape(12.dp))
            ) {
                Text(text = iconText, fontSize = 20.sp)
            }

            // Info Column
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                // Verdict Pill & Score
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(verdictBg.copy(alpha = 0.6f))
                            .padding(horizontal = 8.dp, vertical = 2.dp)
                    ) {
                        Text(
                            text = assessmentLabel(entry.verdict, entry.modelValidated).uppercase(),
                            style = MaterialTheme.typography.labelSmall,
                            color = verdictColor,
                            fontWeight = FontWeight.Bold,
                            fontSize = 10.sp
                        )
                    }

                    if (entry.score.isFinite() && entry.score in 0f..1f) {
                        Text(
                            text = "${(entry.score * 100f).toInt()}%",
                            style = MaterialTheme.typography.labelMedium,
                            color = TextPrimary,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }

                // Date and Latency
                Text(
                    text = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(entry.timestamp)),
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }

            // Delete Action
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(36.dp)
                    .clip(CircleShape)
                    .background(SurfaceElevated)
                    .clickable(onClick = onDelete)
            ) {
                Text(
                    text = "✕",
                    color = TextTertiary,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
private fun HistoryEmptyState(isFiltering: Boolean) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp),
        modifier = Modifier.padding(32.dp)
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(72.dp)
                .clip(CircleShape)
                .background(SurfaceCard)
                .border(1.5.dp, CyberCyan.copy(alpha = 0.4f), CircleShape)
        ) {
            Text("🔍", fontSize = 28.sp)
        }

        Text(
            text = if (isFiltering) "No Matching Assessments" else stringResource(R.string.no_scans),
            style = MaterialTheme.typography.titleMedium,
            color = TextPrimary,
            fontWeight = FontWeight.SemiBold
        )

        Text(
            text = if (isFiltering) "Try switching filter chips above to view records in other modalities." else "Perform an image, speech, or video scan from the home dashboard to generate local forensic assessments.",
            style = MaterialTheme.typography.bodySmall,
            color = TextTertiary,
            lineHeight = 18.sp
        )
    }
}
