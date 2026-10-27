"""Frozen-threshold evaluation with class counts and binomial uncertainty."""
import math
import numpy as np
from .metrics import roc_curve, balanced_accuracy_score
from .metrics import compute_auc, compute_eer, compute_accuracy

def wilson_interval(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    divisor = 1 + z * z / total
    center = (p + z * z / (2 * total)) / divisor
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return [max(0.0, center - radius), min(1.0, center + radius)]

def evaluation_summary(labels, scores, threshold, records=None):
    labels, scores = np.asarray(labels).reshape(-1), np.asarray(scores).reshape(-1)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Evaluation requires a finite frozen operational threshold')
    if records is not None and (len(records) != len(labels) or any(record.get('label') != label or not record.get('group_id') for record, label in zip(records, labels))):
        raise ValueError('Evaluation record labels/groups must align with every scored sample')
    predictions = scores >= threshold
    n_real, n_fake = int((labels == 0).sum()), int((labels == 1).sum())
    tp = int(((labels == 1) & predictions).sum())
    fp = int(((labels == 0) & predictions).sum())
    fpr, tpr, _ = roc_curve(labels, scores)
    result = {'auc': float(compute_auc(labels, scores)), 'eer': float(compute_eer(labels, scores)),
              'accuracy': float(compute_accuracy(labels, scores, threshold)),
              'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
              'threshold': float(threshold), 'n_real': n_real, 'n_fake': n_fake,
              'tpr': tp / n_fake, 'fpr': fp / n_real,
              'tpr_wilson_95pct': wilson_interval(tp, n_fake), 'fpr_wilson_95pct': wilson_interval(fp, n_real),
              'tpr_at_fpr_1pct': float(np.max(tpr[fpr <= .01])),
              'tpr_at_fpr_0_1pct': float(np.max(tpr[fpr <= .001])),
              'uncertainty_note': 'Wilson intervals assume independent examples; recording/source clustering can widen uncertainty.'}
    if records is not None:
        result['n_independent_declared_groups'] = len({r['group_id'] for r in records})
        result['groups'] = {}
        for field in ('source', 'generator', 'speaker', 'language', 'codec', 'platform'):
            values = sorted({str(r[field]) for r in records if r.get(field) is not None})
            if not values:
                continue
            result['groups'][field] = {}
            for value in values:
                mask = np.asarray([str(r.get(field)) == value for r in records])
                y, p = labels[mask], predictions[mask]
                real, fake = int((y == 0).sum()), int((y == 1).sum())
                result['groups'][field][value] = {'n_real': real, 'n_fake': fake,
                    'tpr': float(p[y == 1].mean()) if fake else None,
                    'fpr': float(p[y == 0].mean()) if real else None}
    return result


def assert_source_fpr_within_bound(report, max_fpr=0.05, min_real_per_source=15, context=""):
    """Reject a report whose real-image FPR is high for any single source.

    A detector that flags 44% of LAION images as synthetic has learned that
    LAION images are synthetic, not that they look generated: LAION-5B is the
    training corpus for Stable Diffusion and Flux. The same failure appears in
    audio, where the shipped detector flags 24.3% of real ``inthewild`` field
    recordings because ASVspoof2019 bona-fide speech is clean studio audio.

    This is a dataset shortcut, not detection, and it must fail the build
    rather than ship.

    The per-source breakdown comes from the manifest's ``source`` field via
    ``evaluation_summary``; it is not recoverable from the directory layout.
    """
    prefix = f"[{context}] " if context else ""
    sources = (report.get("groups") or {}).get("source") or {}
    offenders = []
    for source, stats in sources.items():
        if stats.get("n_real", 0) < min_real_per_source:
            continue
        fpr = stats.get("fpr")
        if fpr is not None and fpr > max_fpr:
            offenders.append(f"{source}: fpr={fpr:.3f} (n_real={stats['n_real']})")
    if offenders:
        raise ValueError(
            f"{prefix}Real-image false-positive rate exceeds the per-source bound "
            f"({max_fpr:.0%}). The detector is keying on acquisition source, "
            "not on synthesis. Offenders: " + "; ".join(sorted(offenders))
        )
