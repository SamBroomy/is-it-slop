//! The sklearn-checked evaluation oracle: AUC, threshold fitting, and threshold metrics.
//!
//! Ports the evaluation half of `notebooks/hierarchical_train.py` (removed; last present at commit
//! e1384e6, see git history) (`roc_auc_score`, `precision_recall_curve`, `threshold_at_precision`,
//! `threshold_at_best_f1`, `score`), each unit-tested against values computed by the sklearn
//! implementation it replaces. Labels are `{0.0, 1.0}` floats and scores are probabilities (or
//! anything monotone in them, for AUC).

/// Area under the ROC curve via the Mann-Whitney U statistic, with average ranks for ties.
///
/// Matches `sklearn.metrics.roc_auc_score`.
///
/// # Panics
///
/// Panics if `labels` and `scores` have different lengths, or if either class is absent — an AUC
/// over one class is meaningless and a corrupted validation split should fail loudly.
#[must_use]
pub fn roc_auc(labels: &[f32], scores: &[f32]) -> f64 {
    assert_eq!(
        labels.len(),
        scores.len(),
        "AUC needs exactly one score per label"
    );
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));

    // Average ranks over tied score runs (1-based ranks, as in the U statistic). The comparison
    // is deliberately exact: only bitwise-equal scores are a tie, matching sklearn.
    let mut ranks = vec![0.0f64; scores.len()];
    let mut i = 0;
    #[expect(
        clippy::float_cmp,
        reason = "ties are defined by bitwise-equal scores, as in sklearn"
    )]
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &idx in &order[i..=j] {
            ranks[idx] = avg;
        }
        i = j + 1;
    }

    let n_pos = labels.iter().filter(|&&l| l > 0.5).count();
    let n_neg = labels.len() - n_pos;
    assert!(n_pos > 0 && n_neg > 0, "AUC needs both classes present");

    let pos_rank_sum: f64 = labels
        .iter()
        .zip(&ranks)
        .filter(|&(&l, _)| l > 0.5)
        .map(|(_, &r)| r)
        .sum();
    (pos_rank_sum - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64
}

/// One precision/recall point at the decision rule `prob >= threshold`.
struct PrPoint {
    precision: f64,
    recall: f64,
    threshold: f32,
}

/// The precision-recall curve, one point per distinct score, thresholds ascending.
///
/// Matches `sklearn.metrics.precision_recall_curve` minus its trailing `(1, 0)` sentinel, which
/// corresponds to no threshold.
fn pr_curve(labels: &[f32], probs: &[f32]) -> Vec<PrPoint> {
    assert_eq!(
        labels.len(),
        probs.len(),
        "the PR curve needs exactly one probability per label"
    );
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let total_pos = labels.iter().filter(|&&l| l > 0.5).count();
    assert!(
        total_pos > 0 && total_pos < labels.len(),
        "threshold fitting needs both classes present"
    );

    let mut points = Vec::new();
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    #[expect(
        clippy::float_cmp,
        reason = "ties are defined by bitwise-equal scores, as in sklearn"
    )]
    for (i, &idx) in order.iter().enumerate() {
        if labels[idx] > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        // Only distinct score values are decision boundaries; exact comparison as in sklearn.
        if i + 1 == order.len() || probs[order[i + 1]] != probs[idx] {
            points.push(PrPoint {
                precision: tp / (tp + fp),
                recall: tp / total_pos as f64,
                threshold: probs[idx],
            });
        }
    }
    points.reverse();
    points
}

/// Lowest threshold whose precision on the AI class still clears `target`, maximising recall.
///
/// Magika's rule (Section V-D): fix precision, take the best remaining recall. For slop detection
/// the class worth protecting is human text, so precision on "AI" is the constraint that matters —
/// it bounds how often a human is wrongly accused. Falls back to the best-F1 threshold when no
/// threshold reaches the target.
#[must_use]
pub fn threshold_at_precision(labels: &[f32], probs: &[f32], target: f64) -> f32 {
    // Recall is non-increasing in the threshold, so the first qualifying point (ascending
    // thresholds) carries the maximum recall among them — sklearn's argmax picks the same one.
    pr_curve(labels, probs)
        .iter()
        .find(|p| p.precision >= target)
        .map_or_else(
            || {
                tracing::warn!(
                    "no threshold reaches precision {target:.2}; falling back to best F1"
                );
                threshold_at_best_f1(labels, probs)
            },
            |point| point.threshold,
        )
}

/// The threshold with the highest F1, first one on ties.
#[must_use]
pub fn threshold_at_best_f1(labels: &[f32], probs: &[f32]) -> f32 {
    let points = pr_curve(labels, probs);
    let mut best = (f64::MIN, points[0].threshold);
    for p in points {
        let f1 = if p.precision + p.recall > 0.0 {
            2.0 * p.precision * p.recall / (p.precision + p.recall)
        } else {
            0.0
        };
        if f1 > best.0 {
            best = (f1, p.threshold);
        }
    }
    best.1
}

/// The temperature that minimizes the binary NLL of `sigmoid(logit / T)` against the labels.
///
/// One-parameter post-hoc calibration (temperature scaling): fit on validation, and
/// `sigmoid(logit / T)` approximates the probability of being right, which makes
/// distance-from-threshold a meaningful confidence. Golden-section search over `ln T`; the NLL
/// is smooth and unimodal in `T` for a fixed score ranking.
///
/// # Panics
///
/// Panics if `labels` and `logits` have different lengths.
#[must_use]
pub fn fit_temperature(labels: &[f32], logits: &[f32]) -> f64 {
    assert_eq!(
        labels.len(),
        logits.len(),
        "temperature scaling needs exactly one logit per label"
    );
    let nll = |t: f64| -> f64 {
        labels
            .iter()
            .zip(logits)
            .map(|(&y, &z)| {
                let z = f64::from(z) / t;
                z.max(0.0) - z * f64::from(y) + (1.0 + (-z.abs()).exp()).ln()
            })
            .sum::<f64>()
            / labels.len() as f64
    };
    let (mut lo, mut hi) = (0.05f64.ln(), 20f64.ln());
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (hi - phi * (hi - lo), lo + phi * (hi - lo));
    let (mut fa, mut fb) = (nll(a.exp()), nll(b.exp()));
    for _ in 0..60 {
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - phi * (hi - lo);
            fa = nll(a.exp());
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + phi * (hi - lo);
            fb = nll(b.exp());
        }
    }
    f64::midpoint(lo, hi).exp()
}

/// AUC of a confidence signal at ranking correct predictions above errors (0.5 = no signal).
///
/// `None` when every prediction is correct or every one wrong — there is no ranking to measure.
#[must_use]
pub fn confidence_ranking_auc(correct: &[f32], confidence: &[f32]) -> Option<f64> {
    let right = correct.iter().filter(|&&c| c > 0.5).count();
    (right > 0 && right < correct.len()).then(|| roc_auc(correct, confidence))
}

/// Classification metrics at one decision threshold, `sklearn.metrics` semantics
/// (`zero_division=0`; MCC is 0 when any marginal is empty).
#[derive(Debug, serde::Serialize)]
pub struct ThresholdMetrics {
    /// The decision rule: AI iff `prob >= threshold`.
    pub threshold: f32,
    /// Fraction classified correctly.
    pub accuracy: f64,
    /// `tp / (tp + fp)` on the AI class.
    pub precision: f64,
    /// `tp / (tp + fn)` on the AI class.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
    /// Matthews correlation coefficient.
    pub mcc: f64,
    /// Threshold-independent ROC AUC of the same scores, for context.
    pub auc: f64,
}

/// Score `prob >= threshold` predictions against the labels.
#[must_use]
pub fn score(labels: &[f32], probs: &[f32], threshold: f32) -> ThresholdMetrics {
    let (mut tp, mut fp, mut tn, mut fn_) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&label, &prob) in labels.iter().zip(probs) {
        match (label > 0.5, prob >= threshold) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (false, false) => tn += 1.0,
            (true, false) => fn_ += 1.0,
        }
    }
    let ratio = |num: f64, den: f64| if den > 0.0 { num / den } else { 0.0 };
    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_);
    let mcc_den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    ThresholdMetrics {
        threshold,
        accuracy: (tp + tn) / labels.len() as f64,
        precision,
        recall,
        f1: ratio(2.0 * precision * recall, precision + recall),
        mcc: ratio(tp * tn - fp * fn_, mcc_den),
        auc: roc_auc(labels, probs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_matches_sklearn() {
        // sklearn.metrics.roc_auc_score on the same inputs, including tied scores within and
        // across classes.
        let cases: [(&[f32], &[f32], f64); 3] = [
            (&[0., 0., 1., 1.], &[0.1, 0.4, 0.35, 0.8], 0.75),
            (&[0., 1., 0., 1.], &[0.5, 0.5, 0.2, 0.9], 0.875),
            (
                &[1., 0., 1., 1., 0., 0.],
                &[0.9, 0.9, 0.8, 0.3, 0.3, 0.1],
                0.666_666_666_7,
            ),
        ];
        for (labels, scores, want) in cases {
            let got = roc_auc(labels, scores);
            assert!((got - want).abs() < 1e-9, "got {got}, sklearn says {want}");
        }
    }

    // Oracle cases computed with sklearn's precision_recall_curve and the Python
    // threshold_at_precision / threshold_at_best_f1 / score they feed.
    const A: (&[f32], &[f32]) = (&[0., 0., 1., 1.], &[0.1, 0.4, 0.35, 0.8]);
    const B: (&[f32], &[f32]) = (
        &[1., 0., 1., 1., 0., 0., 1., 0.],
        &[0.9, 0.9, 0.8, 0.3, 0.3, 0.1, 0.65, 0.5],
    );
    const C: (&[f32], &[f32]) = (&[0., 0., 0., 1., 1.], &[0.1, 0.2, 0.3, 0.7, 0.9]);

    #[test]
    fn threshold_at_precision_matches_sklearn() {
        let cases = [
            (A, 0.99, 0.8f32),
            (A, 0.7, 0.8),
            (A, 0.6, 0.35),
            (B, 0.7, 0.65),
            (B, 0.6, 0.5),
            (C, 0.99, 0.7),
            (C, 0.6, 0.3),
        ];
        for ((labels, probs), target, want) in cases {
            let got = threshold_at_precision(labels, probs, target);
            assert!(
                (got - want).abs() < 1e-9,
                "target {target}: got {got}, sklearn says {want}"
            );
        }
    }

    #[test]
    fn unreachable_precision_falls_back_to_best_f1() {
        // No point on B's curve reaches precision 0.99; sklearn's port falls back to 0.65.
        let got = threshold_at_precision(B.0, B.1, 0.99);
        assert!((got - 0.65).abs() < 1e-9, "got {got}, sklearn says 0.65");
    }

    #[test]
    fn threshold_at_best_f1_matches_sklearn() {
        let cases = [(A, 0.35f32), (B, 0.65), (C, 0.7)];
        for ((labels, probs), want) in cases {
            let got = threshold_at_best_f1(labels, probs);
            assert!((got - want).abs() < 1e-9, "got {got}, sklearn says {want}");
        }
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "p is in [0, 1], so the product is in [0, 1000]"
    )]
    fn fit_temperature_recovers_a_known_scale() {
        // Logits that are exactly 2x the calibrated log-odds: for each raw logit z, the fraction
        // of positives is sigmoid(z / 2), so the NLL-optimal temperature is 2 (up to count
        // rounding).
        let (mut labels, mut logits) = (Vec::new(), Vec::new());
        for z in [-4.0f32, -2.0, 0.0, 2.0, 4.0] {
            let p = 1.0 / (1.0 + (-z / 2.0).exp());
            let pos = (1000.0 * p).round() as usize;
            for i in 0..1000 {
                labels.push(f32::from(u8::from(i < pos)));
                logits.push(z);
            }
        }
        let t = fit_temperature(&labels, &logits);
        assert!((t - 2.0).abs() < 0.05, "got T={t}, expected ~2");
    }

    #[test]
    fn confidence_ranking_needs_both_outcomes() {
        assert!(confidence_ranking_auc(&[1.0, 1.0], &[0.9, 0.1]).is_none());
        // Confidence perfectly ranks the correct prediction above the error.
        let auc = confidence_ranking_auc(&[1.0, 0.0, 1.0], &[0.9, 0.1, 0.8]);
        assert_eq!(auc, Some(1.0));
    }

    #[test]
    fn score_matches_sklearn() {
        // sklearn accuracy/precision/recall/f1/mcc on case B at three thresholds.
        let cases = [
            (
                0.5f32,
                0.625,
                0.6,
                0.75,
                0.666_666_666_666_666_6,
                0.258_198_889_747_161_1,
            ),
            (0.65, 0.75, 0.75, 0.75, 0.75, 0.5),
            (0.9, 0.5, 0.5, 0.25, 0.333_333_333_333_333_3, 0.0),
        ];
        for (threshold, accuracy, precision, recall, f1, mcc) in cases {
            let got = score(B.0, B.1, threshold);
            for (name, got, want) in [
                ("accuracy", got.accuracy, accuracy),
                ("precision", got.precision, precision),
                ("recall", got.recall, recall),
                ("f1", got.f1, f1),
                ("mcc", got.mcc, mcc),
            ] {
                assert!(
                    (got - want).abs() < 1e-12,
                    "{name} at {threshold}: got {got}, sklearn says {want}"
                );
            }
        }
    }
}
