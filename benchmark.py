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
    python3 benchmark.py

    # Compare specific models:
    python3 benchmark.py --models multi-domain-rm-llama-3-8b-it multi-domain-rm-gemma-2-9b-it

    # Custom model directory:
    python3 benchmark.py --model_parent_dir model
"""

import csv
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attributes import ATTRIBUTES, DOMAIN_PREFIXES


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
        return json.load(f)


def short_name(model_name):
    """e.g. 'multi-domain-rm-llama-3-8b-it' -> 'llama-3-8b'"""
    return model_name.replace("multi-domain-rm-", "").replace("-it", "")


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

    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 1.5), 5))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("preference", {}).get("domains", {}).get(d)
            vals.append(dd["accuracy"] if dd else 0)
        bars = ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Preference Accuracy by Domain")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
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

    fig, ax = plt.subplots(figsize=(max(6, len(diffs) * 2), 5))
    for i, r in enumerate(all_results):
        vals = []
        for d in diffs:
            dd = r.get("preference", {}).get("difficulty", {}).get(d)
            vals.append(dd["accuracy"] if dd else 0)
        bars = ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Preference Accuracy by Difficulty")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(diffs)
    ax.legend()
    ax.set_ylim(0, 105)
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

    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 1.5), 5))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("scoring", {}).get("domains", {}).get(d)
            vals.append(dd["spearman"] if dd else 0)
        bars = ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Scoring Spearman by Domain")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend()
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

    fig, ax = plt.subplots(figsize=(10, max(6, len(attrs_with_data) * 0.4)))
    for i, r in enumerate(all_results):
        vals = []
        for a in attrs_with_data:
            ad = r.get("scoring", {}).get("attributes", {}).get(a)
            vals.append(ad["spearman"] if ad else 0)
        ax.barh(y + i * height, vals, height, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_xlabel("Spearman Correlation")
    ax.set_title("Scoring Spearman by Attribute")
    ax.set_yticks(y + height * (len(all_results) - 1) / 2)
    ax.set_yticklabels(attrs_with_data, fontsize=7)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
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

    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 1.5), 5))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("scoring", {}).get("domains", {}).get(d)
            vals.append(dd["mse"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_ylabel("MSE")
    ax.set_title("Scoring MSE by Domain (lower is better)")
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_mse_by_domain.png"))


def plot_overall_summary(all_results, shared_plots_dir):
    """Radar/overview chart: overall preference accuracy + avg Spearman per model."""
    names = [short_name(r["_name"]) for r in all_results]
    pref_accs = []
    spear_avgs = []
    for r in all_results:
        pa = r.get("preference", {}).get("accuracy")
        pref_accs.append(pa if pa is not None else 0)
        sa = r.get("scoring", {}).get("average", {}).get("spearman")
        spear_avgs.append(sa if sa is not None else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(names))]

    # Preference accuracy
    bars1 = ax1.bar(names, pref_accs, color=colors)
    for bar, v in zip(bars1, pref_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Overall Preference Accuracy")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="x", rotation=30)

    # Avg Spearman
    bars2 = ax2.bar(names, spear_avgs, color=colors)
    for bar, v in zip(bars2, spear_avgs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Spearman Correlation")
    ax2.set_title("Average Scoring Spearman")
    ax2.tick_params(axis="x", rotation=30)

    fig.suptitle("Model Comparison Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "overall_summary.png"))


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
    shared_plots_dir = os.path.join(model_parent_dir, "benchmark")

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
        plot_scoring_mse_by_domain(all_results, shared_plots_dir)
    if has_pref or has_scoring:
        plot_overall_summary(all_results, shared_plots_dir)

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
    args = parser.parse_args()

    # Discover models
    model_names = args.models or discover_models(args.model_parent_dir)
    if not model_names:
        print("No packaged models found. Run stage-3_package_model.py first.")
        sys.exit(1)

    print(f"Models to compare: {model_names}")

    # Load cached results (eval.json + eval_baseline.json)
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
        if os.path.isfile(baseline_path):
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
    output_dir = os.path.join(args.model_parent_dir, "benchmark")
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
