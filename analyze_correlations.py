"""
Analyze inter-attribute and attribute-vs-length correlations in the scoring dataset.

This script helps decide whether to use --debiasing_dim in stage-2_train.py
by revealing which attribute dimensions correlate with each other and with
response length.

What it computes:
  1. Per-attribute non-null counts — shows how many dialogues have scores for
     each attribute. Because scoring data is domain-specific (each dialogue
     only has scores for its own domain), cross-domain pairs will always have
     zero overlap.

  2. Attribute vs response length (Spearman) — detects whether longer
     assistant responses systematically receive higher or lower scores on
     any given attribute.  A high positive correlation (e.g. 0.68 for
     mu_empathy) means the model may reward verbosity rather than quality.

  3. Inter-attribute correlations (Spearman) — for every pair of attributes
     that share enough non-null rows, computes rank correlation.  Because of
     the domain-specific scoring, these are effectively within-domain pairs
     only (e.g. co_* vs co_*, em_* vs em_*).

  4. Dimension dominance summary — counts how many high-correlation pairs
     each attribute participates in. An attribute with many high-corr pairs
     may be a "dominant" dimension leaking into others.

  5. Length bias warning — flags attributes whose correlation with response
     length exceeds the threshold, suggesting they may benefit from debiasing.

Correlation method:
  Spearman rank correlation is used throughout (not Pearson) because it
  measures monotonic relationships without assuming linearity, and is more
  robust to outliers in score distributions.

Usage:
    python3 analyze_correlations.py
    python3 analyze_correlations.py --dataset_path data/Multi-Domain-Data-Scoring.jsonl --threshold 0.5
"""

import json
import sys
from argparse import ArgumentParser
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from attributes import ATTRIBUTES, DOMAIN_PREFIXES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scoring_data(path):
    """
    Load the scoring JSONL file and extract parallel arrays of attribute
    scores, response lengths, and domain labels.

    Each row in the JSONL has a "scores" dict with all 23 attribute keys,
    but only the attributes belonging to the row's domain have non-null
    values. For example, a coherence-domain dialogue will have co_* scores
    filled and all em_*, cs_*, mu_* scores as null.

    Returns:
        scores_by_attr: dict mapping each attribute name to a list of values
                        (float or None) aligned with the row order.
        lengths:        numpy array of response lengths (character count of
                        all assistant messages concatenated).
        domains:        list of domain strings per row.
        n_total:        total number of dialogues loaded.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Initialize collectors: one list per attribute, same length as rows.
    scores_by_attr = {attr: [] for attr in ATTRIBUTES}
    lengths = []
    domains = []

    for row in rows:
        scores = row.get("scores", {})
        messages = row.get("messages", [])
        domain = row.get("domain", "unknown")

        # Response length = total character count of all assistant turns.
        # This is a proxy for verbosity. We use characters (not tokens)
        # because tokenization depends on the model, but character count
        # is model-agnostic and sufficient for correlation analysis.
        resp_len = sum(len(m["content"]) for m in messages if m.get("role") == "assistant")
        lengths.append(resp_len)
        domains.append(domain)

        # Collect each attribute's score (may be None for out-of-domain rows).
        for attr in ATTRIBUTES:
            val = scores.get(attr)
            scores_by_attr[attr].append(val)

    return scores_by_attr, np.array(lengths, dtype=float), domains, len(rows)


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_pairwise_correlations(scores_by_attr, threshold):
    """
    Compute Spearman correlation for every pair of attributes.

    Only rows where BOTH attributes have non-null scores are considered.
    In practice, this means only within-domain pairs produce results
    (e.g. co_topic_coherence vs co_logical_consistency), because
    cross-domain rows never share non-null scores.

    A minimum of 10 overlapping samples is required to compute a
    meaningful correlation. Pairs where either attribute is constant
    (zero variance) are skipped to avoid division-by-zero.

    Returns a list of dicts with keys: a, b, corr, pval, n.
    """
    results = []
    for a, b in combinations(ATTRIBUTES, 2):
        vals_a = scores_by_attr[a]
        vals_b = scores_by_attr[b]

        # Filter to rows where both attributes have a score.
        pairs = [(va, vb) for va, vb in zip(vals_a, vals_b) if va is not None and vb is not None]
        if len(pairs) < 10:
            # Not enough data for a reliable correlation.
            continue

        arr_a = np.array([p[0] for p in pairs])
        arr_b = np.array([p[1] for p in pairs])

        # Constant arrays would produce NaN correlation.
        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            continue

        corr, pval = spearmanr(arr_a, arr_b)
        if np.isnan(corr):
            continue

        results.append({"a": a, "b": b, "corr": corr, "pval": pval, "n": len(pairs)})
    return results


def compute_length_correlations(scores_by_attr, lengths):
    """
    Compute Spearman correlation between each attribute and response length.

    Unlike inter-attribute correlations, this CAN produce results for all
    attributes regardless of domain, because "length" is computed from the
    raw text and is always available.

    A positive correlation means longer responses tend to score higher on
    that attribute. A negative correlation means shorter responses score
    higher (common for coherence attributes, where concise dialogues are
    often more coherent).

    Returns a list of dicts with keys: attr, corr, pval, n.
    """
    results = []
    for attr in ATTRIBUTES:
        vals = scores_by_attr[attr]

        # Pair each non-null score with the corresponding response length.
        pairs = [(v, l) for v, l in zip(vals, lengths) if v is not None]
        if len(pairs) < 10:
            continue

        arr_v = np.array([p[0] for p in pairs])
        arr_l = np.array([p[1] for p in pairs])

        # Skip if scores or lengths are constant (no variance).
        if np.std(arr_v) == 0 or np.std(arr_l) == 0:
            continue

        corr, pval = spearmanr(arr_v, arr_l)
        if np.isnan(corr):
            continue

        results.append({"attr": attr, "corr": corr, "pval": pval, "n": len(pairs)})
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def domain_of(attr):
    """Map an attribute name to its domain using the prefix convention."""
    for domain, prefix in DOMAIN_PREFIXES.items():
        if attr.startswith(prefix):
            return domain
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Analyze attribute correlations and length bias in scoring data.")
    parser.add_argument("--dataset_path", type=str, default="data/Multi-Domain-Data-Scoring.jsonl",
                        help="Path to the Multi-Domain-Data-Scoring JSONL file.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Flag attribute pairs with |corr| above this value. "
                             "Default 0.5 is a common heuristic for 'strong' correlation.")
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading {args.dataset_path}...")
    scores_by_attr, lengths, domains, n_total = load_scoring_data(args.dataset_path)
    print(f"  Total dialogues: {n_total}")

    # ── Non-null counts ──
    # Shows the data distribution: how many dialogues per domain have scores.
    # Each domain contributes ~12,800 dialogues in the standard dataset.
    print(f"\n{'Attribute':<35} {'Non-null':>8}  {'Domain':<15}")
    print("-" * 62)
    for attr in ATTRIBUTES:
        n = sum(1 for v in scores_by_attr[attr] if v is not None)
        print(f"  {attr:<33} {n:>8}  {domain_of(attr):<15}")

    # ── Section 1: Attribute vs response length ──
    # This reveals length bias: do longer responses get systematically
    # higher or lower scores? If an attribute shows |corr| > threshold,
    # it may be measuring verbosity rather than true quality.
    print(f"\n{'=' * 70}")
    print("ATTRIBUTE vs RESPONSE LENGTH (Spearman)")
    print(f"{'=' * 70}")
    length_results = compute_length_correlations(scores_by_attr, lengths)
    # Sort by absolute correlation (strongest first).
    length_results.sort(key=lambda x: abs(x["corr"]), reverse=True)
    print(f"  {'Attribute':<35} {'Corr':>8} {'p-value':>10} {'N':>7}")
    print(f"  {'-' * 63}")
    for r in length_results:
        # Mark attributes that exceed the threshold with ***.
        flag = " ***" if abs(r["corr"]) > args.threshold else ""
        print(f"  {r['attr']:<35} {r['corr']:>8.4f} {r['pval']:>10.2e} {r['n']:>7}{flag}")

    # ── Section 2: Inter-attribute correlations ──
    # Shows which attributes move together. High correlations within a
    # domain are expected (e.g. co_mutual_grounding and co_overall_coherence
    # at 0.93). Cross-domain correlations would be more concerning but
    # cannot appear here because scores are domain-specific (null overlap).
    print(f"\n{'=' * 70}")
    print("INTER-ATTRIBUTE CORRELATIONS (Spearman)")
    print(f"  (only pairs with |corr| > {args.threshold} shown in detail)")
    print(f"{'=' * 70}")
    pairwise_results = compute_pairwise_correlations(scores_by_attr, args.threshold)
    pairwise_results.sort(key=lambda x: abs(x["corr"]), reverse=True)

    high_corr = [r for r in pairwise_results if abs(r["corr"]) > args.threshold]
    if high_corr:
        print(f"\n  {'Attr A':<30} {'Attr B':<30} {'Corr':>8} {'N':>7}")
        print(f"  {'-' * 78}")
        for r in high_corr:
            print(f"  {r['a']:<30} {r['b']:<30} {r['corr']:>8.4f} {r['n']:>7}")
    else:
        print(f"\n  No pairs found with |corr| > {args.threshold}")

    # ── Section 3: Dimension dominance ──
    # Counts how many high-correlation pairs each attribute participates in.
    # An attribute appearing in many pairs may be a "dominant" dimension
    # that leaks into others — a candidate for --debiasing_dim.
    print(f"\n{'=' * 70}")
    print("DIMENSION DOMINANCE SUMMARY")
    print(f"  (count of pairs with |corr| > {args.threshold} per attribute)")
    print(f"{'=' * 70}")
    counts = {}
    for r in high_corr:
        counts[r["a"]] = counts.get(r["a"], 0) + 1
        counts[r["b"]] = counts.get(r["b"], 0) + 1
    if counts:
        for attr, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {attr:<35} {count:>3} high-corr pairs")
    else:
        print("  No dominant dimensions found.")

    # ── Section 4: Length bias warning ──
    # If any attribute strongly correlates with length, flag it.
    # This is the most actionable output: these attributes might benefit
    # from debiasing via --debiasing_dim in stage-2_train.py.
    length_biased = [r for r in length_results if abs(r["corr"]) > args.threshold]
    if length_biased:
        print(f"\n{'=' * 70}")
        print("LENGTH BIAS WARNING")
        print(f"{'=' * 70}")
        for r in length_biased:
            print(f"  {r['attr']:<35} corr with length = {r['corr']:>8.4f}")
        print(f"\n  These attributes correlate strongly with response length.")
        print(f"  Consider using --debiasing_dim with one of these if needed.")


if __name__ == "__main__":
    main()
