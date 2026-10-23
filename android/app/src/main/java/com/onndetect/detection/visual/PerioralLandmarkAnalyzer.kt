package com.onndetect.detection.visual

import kotlin.math.*

/**
 * Computes pose-normalized biomechanical lip acceleration jitter (BioLip, arXiv:2604.16808)
 * across a sequence of per-frame 468/478 MediaPipe FaceMesh landmarks.
 * Returns Float.NaN when fewer than 3 consecutive frames contain a valid face.
 */
object PerioralLandmarkAnalyzer {

    data class Point(val x: Float, val y: Float)

    // 40 canonical MediaPipe inner + outer lip contour indices
    val LIP_INDICES = intArrayOf(
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415
    )
    private const val LEFT_EYE_OUTER = 33
    private const val RIGHT_EYE_OUTER = 263
    private const val NOSE_TIP = 1
    private const val UPPER_INNER_LIP = 13
    private const val LOWER_INNER_LIP = 14
    private const val LEFT_MOUTH_CORNER = 78
    private const val RIGHT_MOUTH_CORNER = 308

    fun computeJitterScore(frameLandmarks: List<List<Point>?>, minEyeDistPx: Float = 12f): Float {
        // Normalize each valid frame for in-plane head roll and inter-ocular scale
        val normalized = frameLandmarks.map { pts ->
            if (pts == null || pts.size <= 415) return@map null
            val le = pts[LEFT_EYE_OUTER]
            val re = pts[RIGHT_EYE_OUTER]
            val nose = pts[NOSE_TIP]
            val dx = re.x - le.x
            val dy = re.y - le.y
            val eyeDist = hypot(dx, dy)
            if (!eyeDist.isFinite() || eyeDist < minEyeDistPx) return@map null
            val cosA = dx / eyeDist
            val sinA = -dy / eyeDist
            fun norm(idx: Int): Point {
                val p = pts[idx]
                val rx = (p.x - nose.x) * cosA - (p.y - nose.y) * sinA
                val ry = (p.x - nose.x) * sinA + (p.y - nose.y) * cosA
                return Point(rx / eyeDist, ry / eyeDist)
            }
            val lipPts = Array(LIP_INDICES.size) { i -> norm(LIP_INDICES[i]) }
            val u = norm(UPPER_INNER_LIP); val l = norm(LOWER_INNER_LIP)
            val lm = norm(LEFT_MOUTH_CORNER); val rm = norm(RIGHT_MOUTH_CORNER)
            val mar = hypot(u.x - l.x, u.y - l.y) / (hypot(lm.x - rm.x, lm.y - rm.y) + 1e-6f)
            Pair(lipPts, mar)
        }

        var accelSum = 0.0
        var marJerkSum = 0.0
        var triplets = 0
        for (t in 1 until normalized.size - 1) {
            val prev = normalized[t - 1] ?: continue
            val curr = normalized[t] ?: continue
            val next = normalized[t + 1] ?: continue
            var frameAccel = 0.0
            for (i in LIP_INDICES.indices) {
                val ax = next.first[i].x - 2f * curr.first[i].x + prev.first[i].x
                val ay = next.first[i].y - 2f * curr.first[i].y + prev.first[i].y
                frameAccel += hypot(ax.toDouble(), ay.toDouble())
            }
            accelSum += frameAccel / LIP_INDICES.size
            marJerkSum += abs(next.second - 2f * curr.second + prev.second).toDouble()
            triplets++
        }
        if (triplets == 0) return Float.NaN
        val meanAccel = (accelSum / triplets).toFloat()
        val meanMarJerk = (marJerkSum / triplets).toFloat()
        val raw = 0.6f * meanAccel + 0.4f * meanMarJerk
        return (raw / 0.08f).coerceIn(0f, 1f)
    }

    /**
     * Platform adapter overload accepting [android.graphics.PointF].
     */
    fun computeJitterScoreFromPointF(frameLandmarks: List<List<android.graphics.PointF>?>, minEyeDistPx: Float = 12f): Float {
        return computeJitterScore(
            frameLandmarks.map { list -> list?.map { Point(it.x, it.y) } },
            minEyeDistPx
        )
    }
}
