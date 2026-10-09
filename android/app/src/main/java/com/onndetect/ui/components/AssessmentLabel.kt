package com.onndetect.ui.components

import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import com.onndetect.R

/** Unknown and legacy values always abstain. */
fun assessmentStringId(verdict: String, validated: Boolean): Int = if (!validated) R.string.inconclusive else when (verdict) {
    "SYNTHETIC" -> R.string.synthetic
    "AUTHENTIC" -> R.string.no_strong_evidence
    else -> R.string.inconclusive
}
@Composable fun assessmentLabel(verdict: String, validated: Boolean) = stringResource(assessmentStringId(verdict, validated))
