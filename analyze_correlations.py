"""
Analyze inter-attribute and attribute-vs-length correlations in the scoring dataset.

This script helps decide whether to use --debiasing_dims in stage-2_train.py
by revealing which attribute dimensions correlate with each other and with
response length.

What it computes:
  1. Per-attribute statistics — non-null counts, unique values, range, mean, std.
  2. Attribute vs response length (Spearman) — detects verbosity bias.
  3. Inter-attribute correlations (Spearman) — within-domain pair analysis.
  4. Within-domain correlation matrices — full NxN heatmap per domain.
  5. PCA dimensionality analysis — effective independent dimensions per domain.
  6. Dimension dominance summary — which attributes dominate correlation pairs.
  7. Length bias warning — flags attributes correlated with response length.
  8. Debiasing recommendations — actionable suggestions based on all analyses.

Correlation method:
  Spearman rank correlation is used throughout (not Pearson) because it
  measures monotonic relationships without assuming linearity, and is more
  robust to outliers in score distributions.

Usage:
    python3 analyze_correlations.py
    python3 analyze_correlations.py --dataset_path data/dataset/Multi-Domain-Data-Scoring.jsonl --threshold 0.3
"""

import json
import sys
from datetime import datetime
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


def get_domain_indices():
    """Return dict mapping domain name -> list of attribute indices."""
    domains = {}
    for i, attr in enumerate(ATTRIBUTES):
        d = domain_of(attr)
        domains.setdefault(d, []).append(i)
    return domains


def compute_domain_correlation_matrix(scores_by_attr, domain_attrs):
    """
    Compute the full Spearman correlation matrix for a set of attributes.

    Only rows where ALL attributes in the set have non-null scores are used.
    Returns (corr_matrix, n_samples) or (None, 0) if insufficient data.
    """
    n = len(domain_attrs)
    # Collect rows where all attributes are non-null
    all_vals = []
    n_rows = len(scores_by_attr[domain_attrs[0]])
    for row_idx in range(n_rows):
        row = [scores_by_attr[a][row_idx] for a in domain_attrs]
        if all(v is not None for v in row):
            all_vals.append(row)

    if len(all_vals) < 10:
        return None, 0

    arr = np.array(all_vals)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = spearmanr(arr[:, i], arr[:, j])[0]
    return corr_matrix, len(all_vals)


def compute_pca_analysis(corr_matrix):
    """
    Compute PCA eigenvalue decomposition from a correlation matrix.

    Returns list of dicts with: component, eigenvalue, explained_pct, cumulative_pct.
    """
    eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]
    total = eigenvalues.sum()
    explained = eigenvalues / total * 100
    cumulative = np.cumsum(explained)

    results = []
    for k, (ev, exp, cum) in enumerate(zip(eigenvalues, explained, cumulative)):
        results.append({
            "component": k + 1,
            "eigenvalue": ev,
            "explained_pct": exp,
            "cumulative_pct": cum,
        })
    return results


def compute_attribute_stats(scores_by_attr):
    """
    Compute per-attribute statistics: n_nonnull, n_unique, min, max, mean, std.
    """
    stats = []
    for i, attr in enumerate(ATTRIBUTES):
        values = [v for v in scores_by_attr[attr] if v is not None]
        if not values:
            stats.append({"idx": i, "attr": attr, "n": 0})
            continue
        arr = np.array(values)
        unique = np.unique(arr)
        stats.append({
            "idx": i,
            "attr": attr,
            "n": len(values),
            "n_unique": len(unique),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "domain": domain_of(attr),
        })
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n### Analyze correlations started at {datetime.now().isoformat()} ###")
    parser = ArgumentParser(description="Analyze attribute correlations and length bias in scoring data.")
    parser.add_argument("--dataset_path", type=str, default="data/dataset/Multi-Domain-Data-Scoring.jsonl",
                        help="Path to the Multi-Domain-Data-Scoring JSONL file.")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Flag attribute pairs with |corr| above this value.")
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading {args.dataset_path}...")
    scores_by_attr, lengths, domains, n_total = load_scoring_data(args.dataset_path)
    print(f"  Total dialogues: {n_total}")

    # ── Section 1: Attribute statistics ──
    attr_stats = compute_attribute_stats(scores_by_attr)
    print(f"\n{'=' * 100}")
    print("ATTRIBUTE STATISTICS")
    print(f"{'=' * 100}")
    print(f"  {'[idx]':<6} {'Attribute':<35} {'N':>6} {'Unique':>7} {'Min':>6} {'Max':>6} {'Mean':>7} {'Std':>7} {'Domain':<15}")
    print(f"  {'-' * 97}")
    for s in attr_stats:
        if s["n"] == 0:
            print(f"  [{s['idx']:2d}]  {s['attr']:<35} {0:>6}")
            continue
        flag = " LOW-VAR" if s["std"] < 0.10 else ""
        print(f"  [{s['idx']:2d}]  {s['attr']:<35} {s['n']:>6} {s['n_unique']:>7} {s['min']:>6.2f} {s['max']:>6.2f} "
              f"{s['mean']:>7.3f} {s['std']:>7.4f} {s['domain']:<15}{flag}")

    # ── Section 2: Attribute vs response length ──
    print(f"\n{'=' * 100}")
    print("ATTRIBUTE vs RESPONSE LENGTH (Spearman)")
    print(f"{'=' * 100}")
    length_results = compute_length_correlations(scores_by_attr, lengths)
    length_results.sort(key=lambda x: abs(x["corr"]), reverse=True)
    print(f"  {'[idx]':<6} {'Attribute':<35} {'Corr':>8} {'p-value':>10} {'N':>7}")
    print(f"  {'-' * 69}")
    for r in length_results:
        idx = ATTRIBUTES.index(r["attr"])
        flag = " ***" if abs(r["corr"]) > args.threshold else ""
        print(f"  [{idx:2d}]  {r['attr']:<35} {r['corr']:>8.4f} {r['pval']:>10.2e} {r['n']:>7}{flag}")

    # ── Section 3: Inter-attribute correlations ──
    print(f"\n{'=' * 100}")
    print(f"INTER-ATTRIBUTE CORRELATIONS (Spearman, |corr| > {args.threshold})")
    print(f"{'=' * 100}")
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

    # ── Section 4: Within-domain correlation matrices ──
    domain_indices = get_domain_indices()
    print(f"\n{'=' * 100}")
    print("WITHIN-DOMAIN CORRELATION MATRICES (Spearman)")
    print(f"{'=' * 100}")

    for domain_name in ["coherence", "commonsense", "empathy", "multicultural"]:
        indices = domain_indices.get(domain_name, [])
        if not indices:
            continue
        domain_attrs = [ATTRIBUTES[i] for i in indices]
        corr_matrix, n_samples = compute_domain_correlation_matrix(scores_by_attr, domain_attrs)
        if corr_matrix is None:
            continue

        short_names = [a.split("_", 1)[1][:14] for a in domain_attrs]

        print(f"\n  ### {domain_name.upper()} ({n_samples} samples) ###")
        # Header
        print(f"  {'':>16}", end="")
        for s in short_names:
            print(f"  {s:>14}", end="")
        print()

        # Matrix rows
        for i, name in enumerate(short_names):
            print(f"  {name:>16}", end="")
            for j in range(len(indices)):
                val = corr_matrix[i, j]
                marker = "*" if i != j and abs(val) > 0.8 else " "
                print(f"  {val:>13.3f}{marker}", end="")
            print()

        # Average off-diagonal correlation
        n = len(indices)
        off_diag = [corr_matrix[i, j] for i in range(n) for j in range(n) if i != j]
        avg_corr = np.mean(off_diag)
        max_corr = np.max(off_diag)
        min_corr = np.min(off_diag)
        print(f"  Avg off-diagonal corr: {avg_corr:.3f}  (range: [{min_corr:.3f}, {max_corr:.3f}])")

    # ── Section 5: PCA dimensionality analysis ──
    print(f"\n{'=' * 100}")
    print("PCA DIMENSIONALITY ANALYSIS (per domain)")
    print("  How many independent dimensions does each domain effectively have?")
    print(f"{'=' * 100}")

    for domain_name in ["coherence", "commonsense", "empathy", "multicultural"]:
        indices = domain_indices.get(domain_name, [])
        if not indices:
            continue
        domain_attrs = [ATTRIBUTES[i] for i in indices]
        corr_matrix, n_samples = compute_domain_correlation_matrix(scores_by_attr, domain_attrs)
        if corr_matrix is None:
            continue

        pca_results = compute_pca_analysis(corr_matrix)

        print(f"\n  ### {domain_name.upper()} ({len(indices)} attributes → ? effective dims) ###")
        for pc in pca_results:
            marker = ""
            if pc["cumulative_pct"] >= 95 and (pc["component"] == 1 or pca_results[pc["component"] - 2]["cumulative_pct"] < 95):
                marker = "  ← 95% threshold"
            print(f"    PC{pc['component']}: eigenvalue={pc['eigenvalue']:.3f}  "
                  f"explains={pc['explained_pct']:.1f}%  cumulative={pc['cumulative_pct']:.1f}%{marker}")

        # Effective dimensionality: count components needed for 95% variance
        eff_dims = next((pc["component"] for pc in pca_results if pc["cumulative_pct"] >= 95), len(indices))
        print(f"  → {len(indices)} attributes contain ~{eff_dims} independent dimensions of information")

    # ── Section 6: Dimension dominance ──
    print(f"\n{'=' * 100}")
    print(f"DIMENSION DOMINANCE (pairs with |corr| > {args.threshold})")
    print(f"{'=' * 100}")
    counts = {}
    for r in high_corr:
        counts[r["a"]] = counts.get(r["a"], 0) + 1
        counts[r["b"]] = counts.get(r["b"], 0) + 1
    if counts:
        for attr, count in sorted(counts.items(), key=lambda x: -x[1]):
            idx = ATTRIBUTES.index(attr)
            print(f"  [{idx:2d}] {attr:<35} {count:>3} high-corr pairs")
    else:
        print("  No dominant dimensions found.")

    # ── Section 7: Length bias warning ──
    length_biased = [r for r in length_results if abs(r["corr"]) > args.threshold]
    if length_biased:
        print(f"\n{'=' * 100}")
        print("LENGTH BIAS WARNING")
        print(f"{'=' * 100}")
        for r in length_biased:
            idx = ATTRIBUTES.index(r["attr"])
            direction = "longer→higher" if r["corr"] > 0 else "shorter→higher"
            print(f"  [{idx:2d}] {r['attr']:<35} corr={r['corr']:>+.4f}  ({direction})")
        print(f"\n  These attributes correlate strongly with response length.")
        print(f"  Consider using --debiasing_dims with these indices.")

    # ── Section 8: Debiasing recommendations ──
    print(f"\n{'=' * 100}")
    print("DEBIASING RECOMMENDATIONS")
    print(f"{'=' * 100}")

    # Find low-variance attributes
    low_var = [s for s in attr_stats if s["n"] > 0 and s["std"] < 0.10]
    if low_var:
        print(f"\n  A. Low-variance attributes (std < 0.10) — near-constant, contribute noise:")
        for s in low_var:
            print(f"     [{s['idx']:2d}] {s['attr']:<35} std={s['std']:.4f}  range=[{s['min']:.2f}, {s['max']:.2f}]")

    # Find highly redundant pairs (corr > 0.85)
    very_redundant = [r for r in pairwise_results if abs(r["corr"]) > 0.85]
    if very_redundant:
        print(f"\n  B. Highly redundant pairs (|corr| > 0.85) — candidates for debiasing one of each pair:")
        for r in very_redundant:
            idx_a = ATTRIBUTES.index(r["a"])
            idx_b = ATTRIBUTES.index(r["b"])
            print(f"     [{idx_a:2d}] {r['a']:<30} ↔ [{idx_b:2d}] {r['b']:<30} corr={r['corr']:.3f}")

    # Length-biased dims
    strong_length = [r for r in length_results if abs(r["corr"]) > 0.30]
    if strong_length:
        print(f"\n  C. Length-biased attributes (|corr with length| > 0.30):")
        for r in strong_length:
            idx = ATTRIBUTES.index(r["attr"])
            print(f"     [{idx:2d}] {r['attr']:<35} length_corr={r['corr']:>+.4f}")

    # Summary recommendation
    recommended_dims = set()
    for s in low_var:
        recommended_dims.add(s["idx"])
    for r in length_results:
        if abs(r["corr"]) > 0.50:
            recommended_dims.add(ATTRIBUTES.index(r["attr"]))
    if recommended_dims:
        sorted_dims = sorted(recommended_dims)
        dim_str = " ".join(str(d) for d in sorted_dims)
        attr_str = ", ".join(f"{ATTRIBUTES[d]}" for d in sorted_dims)
        print(f"\n  → Suggested --debiasing_dims: {dim_str}")
        print(f"    ({attr_str})")


if __name__ == "__main__":
    main()
