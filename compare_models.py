"""
Compare evaluation results across multiple packaged models.

Loads pre-computed JSON results from model/<model_name>/results/eval.json
and produces:
  - Side-by-side comparison tables (preference accuracy, scoring metrics)
  - Plots saved to model/<model_name>/results/plots/

Workflow:
  1. Run evaluate.py for each model (saves results automatically).
  2. Run this script to compare all models and generate plots.

Usage:
    # Auto-discover all packaged models:
    python3 compare_models.py

    # Compare specific models:
    python3 compare_models.py --models multi-domain-rm-llama-3-8b-it multi-domain-rm-gemma-2-9b-it

    # Custom model directory:
    python3 compare_models.py --model_parent_dir model

    # Skip baselines (only compare trained models):
    python3 compare_models.py --no_baselines --models multi-domain-rm-llama-3-8b-it
"""

import csv
import json
import os
import re
import sys
from argparse import ArgumentParser

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from attributes import ATTRIBUTES, DOMAIN_PREFIXES

print(f"\n### Compare Models started at {datetime.now().isoformat()} ###")

DEFAULT_MODEL_PARENT_DIR = "model"

# Color palette for consistent model colors across plots.
MODEL_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#E91E63", "#795548"]


# ---------------------------------------------------------------------------
# Model discovery and loading
# ---------------------------------------------------------------------------

def discover_models(model_parent_dir="model"):
    """Auto-discover model directories (packaged models and baselines).

    A model is any subdirectory that has config.json (packaged model),
    results/eval.json, or results/eval_baseline.json.
    """
    models = []
    if not os.path.isdir(model_parent_dir):
        return models
    for name in sorted(os.listdir(model_parent_dir)):
        candidate = os.path.join(model_parent_dir, name)
        if not os.path.isdir(candidate):
            continue
        has_config = os.path.isfile(os.path.join(candidate, "config.json"))
        has_eval = os.path.isfile(os.path.join(candidate, "results", "eval.json"))
        has_baseline = os.path.isfile(os.path.join(candidate, "results", "eval_baseline.json"))
        if has_config or has_eval or has_baseline:
            models.append(name)
    return models


def _results_path(model_parent_dir, model_name):
    return os.path.join(model_parent_dir, model_name, "results", "eval.json")


def _baseline_results_path(model_parent_dir, model_name):
    return os.path.join(model_parent_dir, model_name, "results", "eval_baseline.json")


def _plots_dir(model_parent_dir, model_name):
    return os.path.join(model_parent_dir, model_name, "results", "plots")


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize: evaluate.py saves as scoring_100pct/scoring_80pct,
    # but compare_models expects "scoring". Use scoring_100pct.
    if "scoring" not in data and "scoring_100pct" in data:
        data["scoring"] = data["scoring_100pct"]
    return data


def canonical_model_name(model_name):
    """Strip run-specific suffixes appended after the canonical model name."""
    match = re.search(r"\s+\([^)]*\)$", model_name)
    display_suffix = match.group(0) if match else ""
    base_name = model_name[:match.start()] if match else model_name

    match = re.match(r"^(multi-domain-rm-.+?-it)(?:[-_ ].*)?$", base_name)
    if match:
        return f"{match.group(1)}{display_suffix}"
    return model_name


def short_name(model_name):
    """Display name for plots/tables."""
    return canonical_model_name(model_name)


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------

def print_preference_table(all_results):
    print(f"\n{'=' * 90}")
    print("  PREFERENCE ACCURACY COMPARISON")
    print(f"{'=' * 90}")

    names = [short_name(r["_name"]) for r in all_results]
    header = f"  {'':>20}" + "".join(f" {n:>15}" for n in names)
    print(header)
    print(f"  {'-' * (20 + 16 * len(names))}")

    row = f"  {'Overall':>20}"
    for r in all_results:
        acc = r.get("preference", {}).get("accuracy")
        row += f" {acc:>14.2f}%" if acc is not None else f" {'—':>15}"
    print(row)

    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("preference", {}).get("domains", {}).keys())
    for domain in sorted(all_domains):
        row = f"  {domain:>20}"
        for r in all_results:
            d = r.get("preference", {}).get("domains", {}).get(domain)
            row += f" {d['accuracy']:>14.2f}%" if d else f" {'—':>15}"
        print(row)

    all_diffs = set()
    for r in all_results:
        all_diffs.update(r.get("preference", {}).get("difficulty", {}).keys())
    if all_diffs:
        print(f"\n  {'--- By difficulty ---':>20}")
        for diff in sorted(all_diffs):
            row = f"  {diff:>20}"
            for r in all_results:
                d = r.get("preference", {}).get("difficulty", {}).get(diff)
                row += f" {d['accuracy']:>14.2f}%" if d else f" {'—':>15}"
            print(row)

    row = f"  {'Margin (mean)':>20}"
    for r in all_results:
        m = r.get("preference", {}).get("margin_mean")
        row += f" {m:>15.4f}" if m is not None else f" {'—':>15}"
    print(row)


def print_scoring_table(all_results):
    print(f"\n{'=' * 90}")
    print("  SCORING EVALUATION COMPARISON (Spearman / Pearson / MSE)")
    print(f"{'=' * 90}")

    names = [short_name(r["_name"]) for r in all_results]

    print(f"\n  {'Domain':<20}" + "".join(f" {n:>22}" for n in names))
    print(f"  {'-' * (20 + 23 * len(names))}")

    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("scoring", {}).get("domains", {}).keys())
    for domain in sorted(all_domains):
        row = f"  {domain:<20}"
        for r in all_results:
            d = r.get("scoring", {}).get("domains", {}).get(domain)
            row += f"  S={d['spearman']:.3f} P={d['pearson']:.3f}" if d else f" {'—':>22}"
        print(row)

    row = f"  {'AVERAGE':<20}"
    for r in all_results:
        avg = r.get("scoring", {}).get("average", {})
        if avg and avg.get("spearman") is not None:
            row += f"  S={avg['spearman']:.3f} P={avg['pearson']:.3f}"
        else:
            row += f" {'—':>22}"
    print(row)

    print(f"\n  {'Attribute':<35}" + "".join(f" {n:>15}" for n in names) + "  (Spearman)")
    print(f"  {'-' * (35 + 16 * len(names))}")
    for attr in ATTRIBUTES:
        row = f"  {attr:<35}"
        for r in all_results:
            a = r.get("scoring", {}).get("attributes", {}).get(attr)
            row += f" {a['spearman']:>15.4f}" if a else f" {'—':>15}"
        print(row)


def print_global_score_table(all_results):
    print(f"\n{'=' * 90}")
    print("  GLOBAL SCORE DISTRIBUTION")
    print(f"{'=' * 90}")

    names = [short_name(r["_name"]) for r in all_results]
    header = f"  {'':>15}" + "".join(f" {n:>15}" for n in names)
    print(header)
    print(f"  {'-' * (15 + 16 * len(names))}")

    for stat in ["mean", "std", "min", "max"]:
        row = f"  {stat:>15}"
        for r in all_results:
            val = r.get("scoring", {}).get("global_score", {}).get(stat)
            row += f" {val:>15.4f}" if val is not None else f" {'—':>15}"
        print(row)


def print_markdown_summary(all_results):
    names = [short_name(r["_name"]) for r in all_results]

    print(f"\n{'=' * 90}")
    print("  MARKDOWN SUMMARY (copy-paste ready)")
    print(f"{'=' * 90}\n")

    print("### Preference Accuracy\n")
    print("| Domain | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))

    row = "| **Overall** |"
    for r in all_results:
        acc = r.get("preference", {}).get("accuracy")
        row += f" {acc:.2f}% |" if acc is not None else " — |"
    print(row)

    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("preference", {}).get("domains", {}).keys())
    for domain in sorted(all_domains):
        row = f"| {domain} |"
        for r in all_results:
            d = r.get("preference", {}).get("domains", {}).get(domain)
            row += f" {d['accuracy']:.2f}% |" if d else " — |"
        print(row)

    print("\n### Scoring (Spearman)\n")
    print("| Domain | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))

    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("scoring", {}).get("domains", {}).keys())
    for domain in sorted(all_domains):
        row = f"| {domain} |"
        for r in all_results:
            d = r.get("scoring", {}).get("domains", {}).get(domain)
            row += f" {d['spearman']:.4f} |" if d else " — |"
        print(row)

    row = "| **Average** |"
    for r in all_results:
        avg = r.get("scoring", {}).get("average", {})
        row += f" {avg['spearman']:.4f} |" if avg and avg.get("spearman") is not None else " — |"
    print(row)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_preference_accuracy_by_domain(all_results, shared_plots_dir):
    """Bar chart: preference accuracy per domain, grouped by model."""
    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("preference", {}).get("domains", {}).keys())
    domains = sorted(all_domains)
    if not domains:
        return

    names = [short_name(r["_name"]) for r in all_results]
    x = np.arange(len(domains))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 2.5), 8))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("preference", {}).get("domains", {}).get(d)
            vals.append(dd["accuracy"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Preference Accuracy by Domain")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "preference_accuracy_by_domain.png"))


def plot_preference_accuracy_by_difficulty(all_results, shared_plots_dir):
    """Bar chart: preference accuracy per difficulty level."""
    all_diffs = set()
    for r in all_results:
        all_diffs.update(r.get("preference", {}).get("difficulty", {}).keys())
    diffs = sorted(all_diffs)
    if not diffs:
        return

    names = [short_name(r["_name"]) for r in all_results]
    x = np.arange(len(diffs))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(diffs) * 3), 8))
    for i, r in enumerate(all_results):
        vals = []
        for d in diffs:
            dd = r.get("preference", {}).get("difficulty", {}).get(d)
            vals.append(dd["accuracy"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Preference Accuracy by Difficulty")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(diffs)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "preference_accuracy_by_difficulty.png"))


def plot_scoring_spearman_by_domain(all_results, shared_plots_dir):
    """Bar chart: Spearman correlation per domain, grouped by model."""
    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("scoring", {}).get("domains", {}).keys())
    domains = sorted(all_domains)
    if not domains:
        return

    names = [short_name(r["_name"]) for r in all_results]
    x = np.arange(len(domains))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 2.5), 8))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("scoring", {}).get("domains", {}).get(d)
            vals.append(dd["spearman"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Scoring Spearman by Domain")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_spearman_by_domain.png"))


def plot_scoring_spearman_by_attribute(all_results, shared_plots_dir):
    """Horizontal bar chart: Spearman per attribute for each model."""
    names = [short_name(r["_name"]) for r in all_results]
    attrs_with_data = [a for a in ATTRIBUTES if any(
        r.get("scoring", {}).get("attributes", {}).get(a) for r in all_results
    )]
    if not attrs_with_data:
        return

    y = np.arange(len(attrs_with_data))
    height = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(12, max(8, len(attrs_with_data) * 0.5)))
    for i, r in enumerate(all_results):
        vals = []
        for a in attrs_with_data:
            ad = r.get("scoring", {}).get("attributes", {}).get(a)
            vals.append(ad["spearman"] if ad else 0)
        ax.barh(y + i * height, vals, height, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_xlabel("Spearman Correlation")
    ax.set_title("Scoring Spearman by Attribute")
    ax.set_yticks(y + height * (len(all_results) - 1) / 2)
    ax.set_yticklabels(attrs_with_data, fontsize=8)
    ax.legend(loc="upper right", fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_spearman_by_attribute.png"))


def plot_scoring_mse_by_domain(all_results, shared_plots_dir):
    """Bar chart: MSE per domain, grouped by model."""
    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("scoring", {}).get("domains", {}).keys())
    domains = sorted(all_domains)
    if not domains:
        return

    names = [short_name(r["_name"]) for r in all_results]
    x = np.arange(len(domains))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 2.5), 8))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("scoring", {}).get("domains", {}).get(d)
            vals.append(dd["mse"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Scoring MSE by Domain (lower is better)")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_mse_by_domain.png"))


def plot_overall_preference(all_results, shared_plots_dir):
    """Bar chart: overall preference accuracy per model."""
    names = [short_name(r["_name"]) for r in all_results]
    pref_accs = []
    for r in all_results:
        pa = r.get("preference", {}).get("accuracy")
        pref_accs.append(pa if pa is not None else 0)

    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 7))
    bars = ax.bar(names, pref_accs, color=colors)
    for bar, v in zip(bars, pref_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Preference Accuracy")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "overall_preference_accuracy.png"))


def plot_overall_scoring(all_results, shared_plots_dir):
    """Bar chart: average scoring Spearman per model."""
    names = [short_name(r["_name"]) for r in all_results]
    spear_avgs = []
    for r in all_results:
        sa = r.get("scoring", {}).get("average", {}).get("spearman")
        spear_avgs.append(sa if sa is not None else 0)

    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 7))
    bars = ax.bar(names, spear_avgs, color=colors)
    for bar, v in zip(bars, spear_avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Average Scoring Spearman")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "overall_scoring_spearman.png"))


def plot_scoring_100pct_vs_80pct(all_results, shared_plots_dir):
    """Grouped bar chart: Spearman for 100pct vs 80pct per model and domain."""
    # Only use non-baseline results that have both splits
    models = [r for r in all_results if not r.get("_is_baseline")
              and r.get("scoring_100pct") and r.get("scoring_80pct")]
    if not models:
        return

    all_domains = set()
    for r in models:
        all_domains.update(r["scoring_100pct"].get("domains", {}).keys())
        all_domains.update(r["scoring_80pct"].get("domains", {}).keys())
    domains = sorted(all_domains)
    if not domains:
        return

    names = [short_name(r["_name"]) for r in models]
    n_models = len(models)
    n_groups = len(domains)
    x = np.arange(n_groups)
    total_bars = n_models * 2
    width = 0.8 / max(total_bars, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2.5), 8))
    for i, r in enumerate(models):
        vals_100 = [r["scoring_100pct"].get("domains", {}).get(d, {}).get("spearman", 0) for d in domains]
        vals_80 = [r["scoring_80pct"].get("domains", {}).get(d, {}).get("spearman", 0) for d in domains]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.bar(x + (i * 2) * width, vals_100, width, label=f"{names[i]} 100%", color=color, alpha=1.0)
        ax.bar(x + (i * 2 + 1) * width, vals_80, width, label=f"{names[i]} 80%", color=color, alpha=0.5)

    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Scoring Spearman: 100% vs 80% Training Data")
    ax.set_xticks(x + width * (total_bars - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_100pct_vs_80pct.png"))


def plot_cultural_score_by_country(all_results, shared_plots_dir):
    """Bar chart: mean cultural score per country, grouped by model."""
    models = [r for r in all_results if r.get("cultural") and not r.get("_is_baseline")]
    if not models:
        return

    all_countries = set()
    for r in models:
        all_countries.update(r["cultural"].get("countries", {}).keys())
    countries = sorted(all_countries)
    if not countries:
        return

    names = [short_name(r["_name"]) for r in models]
    x = np.arange(len(countries))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(14, len(countries) * 1.5), 8))
    for i, r in enumerate(models):
        vals = [r["cultural"]["countries"].get(c, {}).get("score_mean", 0) for c in countries]
        ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_ylabel("Mean Score")
    ax.set_title("Cultural Evaluation — Mean Score by Country")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(countries, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=12)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_score_by_country.png"))



def plot_cultural_arousal(all_results, shared_plots_dir):
    """Bar chart: mean score per arousal level, grouped by model."""
    models = [r for r in all_results if r.get("cultural") and not r.get("_is_baseline")]
    if not models:
        return

    arousal_levels = sorted(models[0]["cultural"].get("arousal", {}).keys(), key=int)
    if not arousal_levels:
        return

    names = [short_name(r["_name"]) for r in models]
    x = np.arange(len(arousal_levels))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(models):
        vals = [r["cultural"]["arousal"].get(a, {}).get("mean", 0) for a in arousal_levels]
        ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Arousal Level")
    ax.set_title("Cultural Evaluation — Score by Arousal Level")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(arousal_levels)
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_score_by_arousal.png"))


def plot_scoring_spearman_heatmap(all_results, shared_plots_dir):
    """Heatmap: Spearman per attribute × model. Skip models with no scoring attributes."""
    # Filter out models that have no scoring attribute data (e.g. qwen3 baseline)
    filtered = [r for r in all_results if any(
        r.get("scoring", {}).get("attributes", {}).get(a) for a in ATTRIBUTES
    )]
    if not filtered:
        return
    names = [short_name(r["_name"]) for r in filtered]
    attrs_with_data = [a for a in ATTRIBUTES if any(
        r.get("scoring", {}).get("attributes", {}).get(a) for r in filtered
    )]
    if not attrs_with_data:
        return

    data = []
    for r in filtered:
        row = []
        for a in attrs_with_data:
            ad = r.get("scoring", {}).get("attributes", {}).get(a)
            row.append(ad["spearman"] if ad else 0)
        data.append(row)
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(max(12, len(attrs_with_data) * 0.6), max(4, len(names) * 0.8)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.9)
    ax.set_xticks(range(len(attrs_with_data)))
    ax.set_xticklabels(attrs_with_data, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Scoring Spearman by Attribute (Heatmap)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman Correlation")
    # Add text annotations
    for i in range(len(names)):
        for j in range(len(attrs_with_data)):
            v = data[i, j]
            color = "white" if v < 0.2 or v > 0.7 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_spearman_heatmap.png"))


def plot_scoring_100pct_vs_80pct_paired(all_results, shared_plots_dir):
    """Paired scatter plot: 100% vs 80% Spearman per domain, lines connecting pairs."""
    models = [r for r in all_results if not r.get("_is_baseline")
              and r.get("scoring_100pct") and r.get("scoring_80pct")]
    if not models:
        return

    all_domains = set()
    for r in models:
        all_domains.update(r["scoring_100pct"].get("domains", {}).keys())
    domains = sorted(all_domains)
    if not domains:
        return

    names = [short_name(r["_name"]) for r in models]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(models):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        for j, d in enumerate(domains):
            v100 = r["scoring_100pct"].get("domains", {}).get(d, {}).get("spearman", 0)
            v80 = r["scoring_80pct"].get("domains", {}).get(d, {}).get("spearman", 0)
            x_pos = j + i * 0.15 - 0.15
            ax.plot([x_pos, x_pos], [v80, v100], color=color, linewidth=2, alpha=0.7)
            ax.scatter(x_pos, v100, color=color, marker="o", s=60, zorder=5,
                       label=f"{names[i]} 100%" if j == 0 else "")
            ax.scatter(x_pos, v80, color=color, marker="^", s=60, zorder=5, alpha=0.5,
                       label=f"{names[i]} 80%" if j == 0 else "")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Scoring Spearman: 100% vs 80% (paired)")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_100pct_vs_80pct_paired.png"))


def plot_cultural_score_radar(all_results, shared_plots_dir):
    """Radar/spider chart: mean cultural score per country, one line per model."""
    models = [r for r in all_results if r.get("cultural") and not r.get("_is_baseline")]
    if not models:
        return

    all_countries = set()
    for r in models:
        all_countries.update(r["cultural"].get("countries", {}).keys())
    countries = sorted(all_countries)
    if not countries:
        return

    names = [short_name(r["_name"]) for r in models]
    n = len(countries)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i, r in enumerate(models):
        vals = [r["cultural"]["countries"].get(c, {}).get("score_mean", 0) for c in countries]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])
        ax.fill(angles, vals, alpha=0.1, color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(countries, fontsize=9)
    ax.set_title("Cultural Evaluation — Score by Country (Radar)", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_score_radar.png"))


def plot_cultural_arousal_lines(all_results, shared_plots_dir):
    """Line plot: mean score per arousal level, one line per model."""
    models = [r for r in all_results if r.get("cultural") and not r.get("_is_baseline")]
    if not models:
        return

    arousal_levels = sorted(models[0]["cultural"].get("arousal", {}).keys(), key=int)
    if not arousal_levels:
        return

    names = [short_name(r["_name"]) for r in models]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(models):
        vals = [r["cultural"]["arousal"].get(a, {}).get("mean", 0) for a in arousal_levels]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(arousal_levels, vals, marker="o", linewidth=2.5, markersize=8,
                label=names[i], color=color)

    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Arousal Level")
    ax.set_title("Cultural Evaluation — Score by Arousal Level (trend)")
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_score_by_arousal_lines.png"))


def _save_csv(rows, headers, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  Saved: {path}")


def export_csvs(all_results, output_dir):
    """Export comparison tables as CSV files."""
    print(f"\n{'=' * 90}")
    print("  EXPORTING CSVs")
    print(f"{'=' * 90}")

    names = [short_name(r["_name"]) for r in all_results]

    # Preference accuracy by domain
    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("preference", {}).get("domains", {}).keys())
    if all_domains:
        rows = []
        # Overall row
        row = ["Overall"]
        for r in all_results:
            acc = r.get("preference", {}).get("accuracy")
            row.append(f"{acc:.4f}" if acc is not None else "")
        rows.append(row)
        for domain in sorted(all_domains):
            row = [domain]
            for r in all_results:
                d = r.get("preference", {}).get("domains", {}).get(domain)
                row.append(f"{d['accuracy']:.4f}" if d else "")
            rows.append(row)
        _save_csv(rows, ["domain"] + names, os.path.join(output_dir, "preference_accuracy_by_domain.csv"))

    # Preference accuracy by difficulty
    all_diffs = set()
    for r in all_results:
        all_diffs.update(r.get("preference", {}).get("difficulty", {}).keys())
    if all_diffs:
        rows = []
        for diff in sorted(all_diffs):
            row = [diff]
            for r in all_results:
                d = r.get("preference", {}).get("difficulty", {}).get(diff)
                row.append(f"{d['accuracy']:.4f}" if d else "")
            rows.append(row)
        _save_csv(rows, ["difficulty"] + names, os.path.join(output_dir, "preference_accuracy_by_difficulty.csv"))

    # Scoring by domain (Spearman, Pearson, MSE)
    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("scoring", {}).get("domains", {}).keys())
    if all_domains:
        rows = []
        for domain in sorted(all_domains):
            for r in all_results:
                d = r.get("scoring", {}).get("domains", {}).get(domain)
                rows.append([
                    domain, short_name(r["_name"]),
                    f"{d['spearman']:.6f}" if d else "",
                    f"{d['pearson']:.6f}" if d else "",
                    f"{d['mse']:.6f}" if d else "",
                ])
        # Average row per model
        for r in all_results:
            avg = r.get("scoring", {}).get("average", {})
            rows.append([
                "AVERAGE", short_name(r["_name"]),
                f"{avg['spearman']:.6f}" if avg and avg.get("spearman") is not None else "",
                f"{avg['pearson']:.6f}" if avg and avg.get("pearson") is not None else "",
                f"{avg['mse']:.6f}" if avg and avg.get("mse") is not None else "",
            ])
        _save_csv(rows, ["domain", "model", "spearman", "pearson", "mse"],
                  os.path.join(output_dir, "scoring_by_domain.csv"))

    # Scoring by attribute (Spearman, Pearson, MSE)
    has_attrs = any(r.get("scoring", {}).get("attributes") for r in all_results)
    if has_attrs:
        rows = []
        for attr in ATTRIBUTES:
            for r in all_results:
                a = r.get("scoring", {}).get("attributes", {}).get(attr)
                rows.append([
                    attr, short_name(r["_name"]),
                    f"{a['spearman']:.6f}" if a else "",
                    f"{a['pearson']:.6f}" if a else "",
                    f"{a['mse']:.6f}" if a else "",
                ])
        _save_csv(rows, ["attribute", "model", "spearman", "pearson", "mse"],
                  os.path.join(output_dir, "scoring_by_attribute.csv"))

    # Global score distribution
    has_gs = any(r.get("scoring", {}).get("global_score") for r in all_results)
    if has_gs:
        rows = []
        for r in all_results:
            gs = r.get("scoring", {}).get("global_score", {})
            rows.append([
                short_name(r["_name"]),
                f"{gs.get('mean', '')}" if gs.get("mean") is not None else "",
                f"{gs.get('std', '')}" if gs.get("std") is not None else "",
                f"{gs.get('min', '')}" if gs.get("min") is not None else "",
                f"{gs.get('max', '')}" if gs.get("max") is not None else "",
            ])
        _save_csv(rows, ["model", "mean", "std", "min", "max"],
                  os.path.join(output_dir, "global_score_distribution.csv"))


def generate_plots(all_results, model_parent_dir):
    """Generate all plots and save to each model's results/plots/ dir."""
    # Use a shared dir for comparative plots (inside model_parent_dir).
    shared_plots_dir = os.path.join(model_parent_dir, "compare_models")

    print(f"\n{'=' * 90}")
    print("  GENERATING PLOTS")
    print(f"{'=' * 90}")

    has_pref = any("preference" in r for r in all_results)
    has_scoring = any("scoring" in r for r in all_results)

    if has_pref:
        plot_preference_accuracy_by_domain(all_results, shared_plots_dir)
        plot_preference_accuracy_by_difficulty(all_results, shared_plots_dir)
    if has_scoring:
        plot_scoring_spearman_by_domain(all_results, shared_plots_dir)
        plot_scoring_spearman_by_attribute(all_results, shared_plots_dir)
        plot_scoring_spearman_heatmap(all_results, shared_plots_dir)
        plot_scoring_mse_by_domain(all_results, shared_plots_dir)
    if has_pref:
        plot_overall_preference(all_results, shared_plots_dir)
    if has_scoring:
        plot_overall_scoring(all_results, shared_plots_dir)

    # Scoring 100pct vs 80pct
    has_both_splits = any(r.get("scoring_100pct") and r.get("scoring_80pct") for r in all_results)
    if has_both_splits:
        plot_scoring_100pct_vs_80pct(all_results, shared_plots_dir)
        plot_scoring_100pct_vs_80pct_paired(all_results, shared_plots_dir)

    # Cultural evaluation (data/test)
    has_cultural = any(r.get("cultural") for r in all_results)
    if has_cultural:
        plot_cultural_score_by_country(all_results, shared_plots_dir)
        plot_cultural_arousal(all_results, shared_plots_dir)

    # Per-model individual plots (baseline plots go in same dir with _baseline suffix).
    for r in all_results:
        base_name = r.get("_base_name", r["_name"])
        model_plots = _plots_dir(model_parent_dir, base_name)
        os.makedirs(model_plots, exist_ok=True)
        suffix = "_baseline" if r.get("_is_baseline") else ""
        _plot_single_model(r, model_plots, suffix=suffix)


def _plot_single_model(result, plots_dir, suffix=""):
    """Generate per-model plots (not comparative).

    Args:
        suffix: appended to filenames, e.g. "_baseline" → "preference_by_domain_baseline.png"
    """
    name = short_name(result["_name"])

    # Scoring: Spearman per attribute
    scoring = result.get("scoring", {})
    attrs_data = scoring.get("attributes", {})
    if attrs_data:
        attrs = [a for a in ATTRIBUTES if a in attrs_data]
        vals = [attrs_data[a]["spearman"] for a in attrs]
        if attrs:
            fig, ax = plt.subplots(figsize=(8, max(5, len(attrs) * 0.35)))
            colors = ["#4CAF50" if v >= 0.5 else "#FF9800" if v >= 0.3 else "#F44336" for v in vals]
            ax.barh(attrs, vals, color=colors)
            ax.set_xlabel("Spearman Correlation")
            ax.set_title(f"{name} — Spearman by Attribute")
            ax.invert_yaxis()
            ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
            ax.legend(fontsize=8)
            _save_fig(fig, os.path.join(plots_dir, f"spearman_by_attribute{suffix}.png"))

    # Preference: accuracy per domain
    pref = result.get("preference", {})
    domains_data = pref.get("domains", {})
    if domains_data:
        domains = sorted(domains_data.keys())
        accs = [domains_data[d]["accuracy"] for d in domains]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(domains, accs, color="#2196F3")
        for bar, v in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        overall = pref.get("accuracy")
        if overall is not None:
            ax.axhline(y=overall, color="red", linestyle="--", alpha=0.7, label=f"Overall: {overall:.1f}%")
            ax.legend()
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{name} — Preference Accuracy by Domain")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=30)
        _save_fig(fig, os.path.join(plots_dir, f"preference_by_domain{suffix}.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Compare evaluation results across packaged models.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to compare. Default: auto-discover from model/.")
    parser.add_argument("--model_parent_dir", type=str, default=DEFAULT_MODEL_PARENT_DIR,
                        help="Parent directory for packaged models.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip plot generation.")
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip loading eval_baseline.json for each model.")
    args = parser.parse_args()

    # Discover models
    model_names = args.models or discover_models(args.model_parent_dir)
    if not model_names:
        print("No packaged models found. Run stage-3_package_model.py first.")
        sys.exit(1)

    print(f"Models to compare: {model_names}")

    # Load cached results
    all_results = []
    for name in model_names:
        json_path = _results_path(args.model_parent_dir, name)
        if os.path.isfile(json_path):
            print(f"  Loading: {json_path}")
            r = load_results(json_path)
            r["_name"] = name
            all_results.append(r)
        else:
            print(f"  WARNING: No results for {name} at {json_path}, skipping.")
            print(f"           Run: python3 evaluate.py --model_name {name}")

        baseline_path = _baseline_results_path(args.model_parent_dir, name)
        if not args.no_baselines and os.path.isfile(baseline_path):
            print(f"  Loading: {baseline_path}")
            rb = load_results(baseline_path)
            rb["_name"] = f"{name} (baseline)"
            rb["_base_name"] = name
            rb["_is_baseline"] = True
            all_results.append(rb)

    if len(all_results) < 1:
        print("No results to compare. Run evaluate.py for each model first.")
        sys.exit(1)

    # Print comparison tables
    if any("preference" in r for r in all_results):
        print_preference_table(all_results)

    if any("scoring" in r for r in all_results):
        print_scoring_table(all_results)
        print_global_score_table(all_results)

    print_markdown_summary(all_results)

    # Export CSVs and generate plots
    output_dir = os.path.join(args.model_parent_dir, "compare_models")
    # Clean previous comparison outputs to avoid stale plots/CSVs
    if os.path.isdir(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    export_csvs(all_results, output_dir)

    if not args.no_plots:
        generate_plots(all_results, args.model_parent_dir)

    print(f"\n{'=' * 90}")
    print(f"  Results loaded from: {args.model_parent_dir}/<model>/results/eval.json")
    print(f"  CSVs & plots:        {output_dir}/")
    print(f"  Per-model plots:     {args.model_parent_dir}/<model>/results/plots/")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
