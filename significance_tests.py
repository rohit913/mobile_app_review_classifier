"""Tests the two numerical success criteria of this project against their targets.

The obvious comparison sets BERT's cross-validated margin over the SVM against its own
fold-to-fold standard deviation and concludes the two are statistically
indistinguishable. That is an eyeball comparison, not a test. This script runs the
tests, for RQ1's macro F1 criterion and for RQ2's coherence criterion alike, so that
both shortfalls are reported to the same standard of evidence.

Reads reports/cross_validation_results.json (per-fold macro F1, notebook 09) and
reports/coherence_assessment_completed.csv (the 20 manual verdicts, notebook 06), and
writes reports/significance_tests.json. It retrains nothing and reads no model, so it
is deterministic: running it twice gives identical output, and it cannot change any
figure already reported.

Four questions:

  1. Is BERT's margin over the SVM distinguishable from zero?  Paired t-test over the
     five folds, plus a Wilcoxon signed-rank check.
  2. What range is BERT's true mean plausibly in?  95% confidence interval.
  3. Was the 0.75 target missed, or is the shortfall within noise?  One-sample t-test
     of the fold means against 0.75.
  4. And the same question for RQ2's criterion: is 14 of 20 coherent explanations
     distinguishable from the 80% target?  Binomial interval and exact test over the
     verdicts in reports/coherence_assessment_completed.csv. It is not, which makes
     the two shortfalls different kinds of claim.

One caveat is recorded alongside the results: k-fold
folds share training data, so the paired t-test's independence assumption is violated
and the test is anti-conservative [38], [39]. That biases it toward declaring a
difference, which only strengthens a null result.

Run from the repository root:

    python significance_tests.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

CV_PATH = Path("reports/cross_validation_results.json")
COHERENCE_PATH = Path("reports/coherence_assessment_completed.csv")
OUT_PATH = Path("reports/significance_tests.json")
TARGET = 0.75
COHERENCE_TARGET = 0.80
ALPHA = 0.05


def ci_mean(x, alpha=ALPHA):
    """Two-sided t interval for the mean of x, using the sample SD (ddof=1)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    half = stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - half), float(x.mean() + half)


def read_coherence(path=COHERENCE_PATH):
    """Count the Y judgements in the manual coherence assessment."""
    with path.open(newline="", encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    verdicts = [r["coherent_Y_N"].strip().upper() for r in rows]
    unknown = [v for v in verdicts if v not in {"Y", "N"}]
    if unknown:
        raise SystemExit(f"{path}: unrecognised verdicts {sorted(set(unknown))}")
    return verdicts.count("Y"), len(verdicts)


def min_n_to_exclude(rate, target, alpha=ALPHA, limit=2000):
    """Smallest sample at which observing `rate` puts `target` outside the Wilson interval.

    Answers the design question the coherence assessment should have asked before it was
    run: how many reviews would 20 have had to be for a 70% result to be distinguishable
    from an 80% criterion at all.
    """
    for n in range(2, limit):
        k = round(rate * n)
        lo, hi = stats.binomtest(k, n).proportion_ci(1 - alpha, "wilson")
        if hi < target or lo > target:
            return n
    return None


def main():
    cv = json.loads(CV_PATH.read_text())
    bert = np.array(cv["bert"]["per_fold_macro_f1"], dtype=float)
    svm = np.array(cv["svm"]["per_fold_macro_f1"], dtype=float)
    diff = bert - svm
    n = len(bert)

    t_paired, p_paired = stats.ttest_rel(bert, svm)
    w_stat, p_wilcoxon = stats.wilcoxon(bert, svm)
    t_target, p_target = stats.ttest_1samp(bert, TARGET)

    # RQ2's criterion, held to the same standard as RQ1's. A proportion from 20 trials
    # needs a binomial interval, not a t interval: Wilson is reported as the interval to
    # quote and Clopper-Pearson as the conservative check.
    k_coh, n_coh = read_coherence()
    bt = stats.binomtest(k_coh, n_coh)
    wilson = bt.proportion_ci(1 - ALPHA, "wilson")
    exact_ci = bt.proportion_ci(1 - ALPHA, "exact")
    p_coh_two = stats.binomtest(k_coh, n_coh, COHERENCE_TARGET, "two-sided").pvalue
    p_coh_one = stats.binomtest(k_coh, n_coh, COHERENCE_TARGET, "less").pvalue
    min_n = min_n_to_exclude(k_coh / n_coh, COHERENCE_TARGET)

    results = {
        "_method": (
            "Significance tests over the five per-fold macro F1 scores in "
            "cross_validation_results.json (notebook 09) and over the 20 manual verdicts in "
            "coherence_assessment_completed.csv (notebook 06). No model is retrained and no "
            "existing figure changes; this file only tests figures already reported. "
            "Produced by significance_tests.py."
        ),
        "_caveat": (
            "k-fold training sets overlap, so the folds are not independent and the paired "
            "t-test is anti-conservative [38], [39]: it is biased toward reporting a "
            "difference. The BERT-vs-SVM result below is null despite that bias, so the "
            "violation works against the conclusion being drawn, not for it. With n=5 the "
            "Wilcoxon signed-rank test cannot return p<0.0625 at any effect size, so it is "
            "reported as a consistency check only."
        ),
        "n_folds": n,
        "bert_vs_svm": {
            "per_fold_difference": [round(float(d), 4) for d in diff],
            "mean_difference": round(float(diff.mean()), 4),
            "paired_t": round(float(t_paired), 4),
            "df": n - 1,
            "p_value": round(float(p_paired), 4),
            "ci95_difference": [round(v, 4) for v in ci_mean(diff)],
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_p_value": round(float(p_wilcoxon), 4),
            "significant_at_0.05": bool(p_paired < ALPHA),
            "reading": (
                "The margin is not distinguishable from zero. The interval spans both "
                "signs, so these five folds do not establish which model is better."
            ),
        },
        "bert_mean": {
            "mean_macro_f1": round(float(bert.mean()), 4),
            "sd_sample_ddof1": round(float(bert.std(ddof=1)), 4),
            "sd_population_ddof0": round(float(bert.std(ddof=0)), 4),
            "ci95_mean": [round(v, 4) for v in ci_mean(bert)],
            "_sd_note": (
                "cross_validation_results.json records 0.0302, the population SD across the five "
                "folds. The interval above uses the sample SD (ddof=1, 0.0338), which is "
                "the correct estimator for an interval on the mean. Both describe the same "
                "five numbers."
            ),
        },
        "bert_vs_target": {
            "target": TARGET,
            "t": round(float(t_target), 4),
            "df": n - 1,
            "p_value": round(float(p_target), 4),
            "significant_at_0.05": bool(p_target < ALPHA),
            "reading": (
                "The target is missed, and unlike the model comparison this shortfall is "
                "distinguishable from noise: the 95% interval on BERT's mean lies entirely "
                "below 0.75."
            ),
        },
        "coherence_vs_target": {
            "_source": (
                "reports/coherence_assessment_completed.csv, the 20-review manual "
                "assessment of explanation coherence. Counted here rather than hard-coded."
            ),
            "coherent": k_coh,
            "n": n_coh,
            "rate": round(k_coh / n_coh, 4),
            "target": COHERENCE_TARGET,
            "ci95_wilson": [round(v, 4) for v in wilson],
            "ci95_clopper_pearson": [round(v, 4) for v in exact_ci],
            "exact_binomial_p_two_sided": round(float(p_coh_two), 4),
            "exact_binomial_p_one_sided_less": round(float(p_coh_one), 4),
            "significant_at_0.05": bool(p_coh_two < ALPHA),
            "min_n_to_distinguish": min_n,
            "reading": (
                "The point estimate is below target but the shortfall is NOT distinguishable "
                "from it: both 95% intervals contain 0.80 and an exact binomial test does not "
                "reject it (p = 0.265). This is the opposite of the macro F1 result above, "
                "where the interval excludes the target. At n=20 the assessment could not "
                f"have resolved a 10-point gap: {min_n} reviews would have been the minimum, "
                "so the criterion was underpowered by design rather than merely missed."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Folds: {n}\n")
    print("BERT vs SVM (paired):")
    print(f"  mean difference  {diff.mean():+.4f}")
    print(f"  t({n - 1}) = {t_paired:.3f}, p = {p_paired:.4f}")
    print(f"  95% CI          [{ci_mean(diff)[0]:+.4f}, {ci_mean(diff)[1]:+.4f}]")
    print(f"  Wilcoxon p = {p_wilcoxon:.4f}\n")
    print("BERT mean:")
    print(f"  mean = {bert.mean():.4f}, sample SD = {bert.std(ddof=1):.4f}")
    print(f"  95% CI          [{ci_mean(bert)[0]:.4f}, {ci_mean(bert)[1]:.4f}]\n")
    print(f"BERT vs {TARGET} target:")
    print(f"  t({n - 1}) = {t_target:.3f}, p = {p_target:.4f}\n")
    print(f"Coherence vs {COHERENCE_TARGET} target:")
    print(f"  {k_coh}/{n_coh} = {k_coh / n_coh:.4f}")
    print(f"  95% CI Wilson    [{wilson[0]:.4f}, {wilson[1]:.4f}]"
          f"   {'contains' if wilson[0] <= COHERENCE_TARGET <= wilson[1] else 'excludes'} the target")
    print(f"  95% CI exact     [{exact_ci[0]:.4f}, {exact_ci[1]:.4f}]")
    print(f"  exact binomial   p = {p_coh_two:.4f} (two-sided), {p_coh_one:.4f} (one-sided)")
    print(f"  would have needed n = {min_n} for a {k_coh / n_coh:.0%} rate to exclude "
          f"{COHERENCE_TARGET:.0%}")
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
