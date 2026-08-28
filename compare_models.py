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
    python3 compare_models.py --models multi-domain-rm-fsfairx-llama-3-8b-it multi-domain-rm-fsfairx-gemma-2-9b-it

    # Custom model directory:
    python3 compare_models.py --model_parent_dir model

    # Skip baselines (only compare trained models):
    python3 compare_models.py --no_baselines --models multi-domain-rm-fsfairx-llama-3-8b-it
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from datetime import datetime
from attributes import ATTRIBUTES, DOMAIN_PREFIXES
from config_utils import load_yaml_config, apply_section_overrides

print(f"\n### Compare Models started at {datetime.now().isoformat()} ###")

DEFAULT_MODEL_PARENT_DIR = "model"

# Color palette for consistent model colors across plots.
MODEL_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#E91E63", "#795548"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
BLUE_ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "blue_orange_accessible",
    ["#0072B2", "#E8ECF3", "#E69F00"],
)


def _split_display_suffix(model_name):
    match = re.search(r"\s+\([^)]*\)$", model_name)
    if not match:
        return model_name, ""
    return model_name[:match.start()], match.group(0)


def _canonical_base_name(model_name):
    base_name, _ = _split_display_suffix(model_name)
    match = re.match(r"^(multi-domain-rm-.+?-it)(?:[-_ ].*)?$", base_name)
    if match:
        return match.group(1)
    return base_name


def _lighten_hex(hex_color, amount=0.55):
    """Mix a hex color with white; amount=0 keeps original, amount=1 returns white."""
    color = hex_color.lstrip("#")
    rgb = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(channel + (255 - channel) * amount) for channel in rgb]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def _result_color_key(result):
    return _canonical_base_name(result.get("_base_name", result["_name"]))


def _result_color_map(results):
    color_map = {}
    for r in results:
        key = _result_color_key(r)
        if key not in color_map:
            color_map[key] = MODEL_COLORS[len(color_map) % len(MODEL_COLORS)]
    return color_map


def _result_color(result, color_map):
    base_color = color_map[_result_color_key(result)]
    if result.get("_is_baseline"):
        return _lighten_hex(base_color)
    return base_color


def _result_colors(results):
    color_map = _result_color_map(results)
    return [_result_color(r, color_map) for r in results]


def _ordered_difficulties(difficulties):
    ordered = [d for d in DIFFICULTY_ORDER if d in difficulties]
    ordered.extend(sorted(d for d in difficulties if d not in DIFFICULTY_ORDER))
    return ordered


def _attribute_prefix(attribute):
    return attribute.split("_", 1)[0] if "_" in attribute else attribute


def _attributes_with_group_gaps(attributes):
    grouped = []
    previous_prefix = None
    for attribute in attributes:
        prefix = _attribute_prefix(attribute)
        if previous_prefix is not None and prefix != previous_prefix:
            grouped.append(None)
        grouped.append(attribute)
        previous_prefix = prefix
    return grouped


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
    _, display_suffix = _split_display_suffix(model_name)
    base_name = _canonical_base_name(model_name)
    return f"{base_name.replace('multi-domain-rm-', '', 1)}{display_suffix}"


def short_name(model_name):
    """Display name for plots/tables."""
    return canonical_model_name(model_name)


def axis_model_label(model_name):
    """Display name for model names used as horizontal axis labels."""
    name = short_name(model_name)
    suffix = " (baseline)"
    if name.endswith(suffix):
        return f"{name[:-len(suffix)]}\nbaseline"
    return name


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
        for diff in _ordered_difficulties(all_diffs):
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


def print_scoring_attributes_table(all_results):
    print(f"\n{'=' * 90}")
    print("  SCORING ATTRIBUTES COMPARISON (Spearman)")
    print(f"{'=' * 90}")

    names = [short_name(r["_name"]) for r in all_results]

    print(f"\n  {'Attribute':<35}" + "".join(f" {n:>15}" for n in names) + "  (Spearman)")
    print(f"  {'-' * (35 + 16 * len(names))}")
    for attr in ATTRIBUTES:
        row = f"  {attr:<35}"
        for r in all_results:
            a = r.get("scoring", {}).get("attributes", {}).get(attr)
            row += f" {a['spearman']:>15.4f}" if a else f" {'—':>15}"
        print(row)


def print_scoring_domain_table(all_results):
    print(f"\n{'=' * 90}")
    print("  SCORING DOMAIN COMPARISON (Spearman / Pearson / MSE)")
    print(f"{'=' * 90}")

    names = [short_name(r["_name"]) for r in all_results]
    metric_col_width = 30

    print(f"\n  {'Domain':<20}" + "".join(f" {n:>{metric_col_width}}" for n in names))
    print(f"  {'-' * (20 + (metric_col_width + 1) * len(names))}")

    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("scoring", {}).get("domains", {}).keys())
    for domain in sorted(all_domains):
        row = f"  {domain:<20}"
        for r in all_results:
            d = r.get("scoring", {}).get("domains", {}).get(domain)
            row += f"  S={d['spearman']:.3f} P={d['pearson']:.3f} M={d['mse']:.3f}" if d else f" {'—':>{metric_col_width}}"
        print(row)

    row = f"  {'AVERAGE':<20}"
    for r in all_results:
        avg = r.get("scoring", {}).get("average", {})
        if avg and avg.get("spearman") is not None:
            row += f"  S={avg['spearman']:.3f} P={avg['pearson']:.3f} M={avg['mse']:.3f}"
        else:
            row += f" {'—':>{metric_col_width}}"
    print(row)


def print_scoring_table(all_results):
    print_scoring_attributes_table(all_results)
    print_scoring_domain_table(all_results)


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


def print_cultural_fairness_tables(all_results):
    models = [r for r in all_results if r.get("cultural") and not r.get("_is_baseline")]
    if not models:
        return

    names = [short_name(r["_name"]) for r in models]
    col_width = max(15, max(len(n) for n in names))

    print(f"\n{'=' * 90}")
    print("  GLOBAL MEAN PREFERENCE SCORES (UNCALIBRATED LOGITS)")
    print("  Stratified by Geopolitical Provenance. Stable scores across countries indicate geographic fairness.")
    print("  Source: data/test/human_scores_multicultural.jsonl")
    print(f"{'=' * 90}")

    countries = sorted({
        country
        for r in models
        for country in r.get("cultural", {}).get("countries", {}).keys()
    })
    if countries:
        country_width = max(18, max(len(country) for country in countries))
        print(f"\n  {'Country':<{country_width}} {'N':>5}" + "".join(f" {n:>{col_width}}" for n in names))
        print(f"  {'-' * (country_width + 6 + (col_width + 1) * len(names))}")
        for country in countries:
            n = next((
                r.get("cultural", {}).get("countries", {}).get(country, {}).get("n")
                for r in models
                if r.get("cultural", {}).get("countries", {}).get(country)
            ), None)
            row = f"  {country:<{country_width}} {n:>5}" if n is not None else f"  {country:<{country_width}} {'—':>5}"
            for r in models:
                val = r.get("cultural", {}).get("countries", {}).get(country, {}).get("score_mean")
                row += f" {val:>{col_width}.4f}" if val is not None else f" {'—':>{col_width}}"
            print(row)

    print(f"\n{'=' * 90}")
    print("  SCORE CORRELATION AGAINST EMOTIONAL AROUSAL (1-5 SCALE)")
    print("  Note: An optimal, unbiased model yields a correlation approaching 0.0,")
    print("        indicating complete independence from emotional intensity.")
    print("  Source: data/test/human_scores_multicultural.jsonl")
    print(f"{'=' * 90}")

    print(f"\n  {'Model':<{col_width}} {'Pearson':>10} {'Spearman':>10}")
    print(f"  {'-' * (col_width + 22)}")
    for r, name in zip(models, names):
        corr = r.get("cultural", {}).get("score_vs_arousal", {})
        pearson = corr.get("pearson")
        spearman = corr.get("spearman")
        p = f"{pearson:>10.4f}" if pearson is not None else f"{'—':>10}"
        s = f"{spearman:>10.4f}" if spearman is not None else f"{'—':>10}"
        print(f"  {name:<{col_width}} {p} {s}")


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
    colors = _result_colors(all_results)
    x = np.arange(len(domains))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 2.5), 8))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("preference", {}).get("domains", {}).get(d)
            vals.append(dd["accuracy"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=colors[i])

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
    diffs = _ordered_difficulties(all_diffs)
    if not diffs:
        return

    names = [short_name(r["_name"]) for r in all_results]
    colors = _result_colors(all_results)
    x = np.arange(len(diffs))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(diffs) * 3), 8))
    for i, r in enumerate(all_results):
        vals = []
        for d in diffs:
            dd = r.get("preference", {}).get("difficulty", {}).get(d)
            vals.append(dd["accuracy"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=colors[i])

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
    colors = _result_colors(all_results)
    x = np.arange(len(domains))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 2.5), 8))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("scoring", {}).get("domains", {}).get(d)
            vals.append(dd["spearman"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=colors[i])

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
    colors = _result_colors(all_results)
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
        ax.barh(y + i * height, vals, height, label=names[i], color=colors[i])

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
    colors = _result_colors(all_results)
    x = np.arange(len(domains))
    width = 0.8 / max(len(all_results), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 2.5), 8))
    for i, r in enumerate(all_results):
        vals = []
        for d in domains:
            dd = r.get("scoring", {}).get("domains", {}).get(d)
            vals.append(dd["mse"] if dd else 0)
        ax.bar(x + i * width, vals, width, label=names[i], color=colors[i])

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
    xlabels = [axis_model_label(r["_name"]) for r in all_results]
    pref_accs = []
    for r in all_results:
        pa = r.get("preference", {}).get("accuracy")
        pref_accs.append(pa if pa is not None else 0)

    colors = _result_colors(all_results)
    x = np.arange(len(all_results))

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2.1), 7))
    bars = ax.bar(x, pref_accs, color=colors)
    for bar, v in zip(bars, pref_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Preference Accuracy")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "overall_preference_accuracy.png"))


def plot_overall_scoring(all_results, shared_plots_dir):
    """Bar chart: average scoring Spearman per model."""
    names = [short_name(r["_name"]) for r in all_results]
    xlabels = [axis_model_label(r["_name"]) for r in all_results]
    spear_avgs = []
    for r in all_results:
        sa = r.get("scoring", {}).get("average", {}).get("spearman")
        spear_avgs.append(sa if sa is not None else 0)

    colors = _result_colors(all_results)
    x = np.arange(len(all_results))

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2.1), 7))
    bars = ax.bar(x, spear_avgs, color=colors)
    for bar, v in zip(bars, spear_avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Average Scoring Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "overall_scoring_spearman.png"))


def plot_scoring_80pct_vs_100pct(all_results, shared_plots_dir):
    """Grouped bar chart: Spearman for 80pct vs 100pct per model and domain."""
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
    colors = _result_colors(models)
    n_models = len(models)
    n_groups = len(domains)
    x = np.arange(n_groups)
    total_bars = n_models * 2
    width = 0.8 / max(total_bars, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2.5), 8))
    for i, r in enumerate(models):
        vals_80 = [r["scoring_80pct"].get("domains", {}).get(d, {}).get("spearman", 0) for d in domains]
        vals_100 = [r["scoring_100pct"].get("domains", {}).get(d, {}).get("spearman", 0) for d in domains]
        color = colors[i]
        ax.bar(x + (i * 2) * width, vals_80, width, label=f"{names[i]} 80%", color=color, alpha=0.5)
        ax.bar(x + (i * 2 + 1) * width, vals_100, width, label=f"{names[i]} 100%", color=color, alpha=1.0)

    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Scoring Spearman: 80% vs 100% Training Data")
    ax.set_xticks(x + width * (total_bars - 1) / 2)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_80pct_vs_100pct.png"))


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
    colors = _result_colors(models)
    x = np.arange(len(countries))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(14, len(countries) * 1.5), 8))
    for i, r in enumerate(models):
        vals = [r["cultural"]["countries"].get(c, {}).get("score_mean", 0) for c in countries]
        ax.bar(x + i * width, vals, width, label=names[i], color=colors[i])

    ax.set_ylabel("Mean Score")
    ax.set_title("Cultural Evaluation — Mean Score by Country")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(countries, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=12)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_score_by_country.png"))


def plot_cultural_country_score_violin(all_results, shared_plots_dir):
    """Violin plot: distribution of country-level cultural scores per model."""
    models = [r for r in all_results if r.get("cultural") and not r.get("_is_baseline")]
    if not models:
        return

    names = [short_name(r["_name"]) for r in models]
    colors = _result_colors(models)
    plot_rows = []
    for name, color, r in zip(names, colors, models):
        countries = r.get("cultural", {}).get("countries", {})
        vals = [
            c.get("score_mean")
            for c in countries.values()
            if c.get("score_mean") is not None
        ]
        if vals:
            plot_rows.append((name, color, vals))
    if not plot_rows:
        return

    plot_names = [row[0] for row in plot_rows]
    plot_colors = [row[1] for row in plot_rows]
    datasets = [row[2] for row in plot_rows]
    positions = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.8), 7))
    parts = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.75,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], plot_colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    for key in ["cmeans", "cmedians"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.4 if key == "cmedians" else 1.0)

    for i, vals in enumerate(datasets):
        offsets = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            positions[i] + offsets,
            vals,
            s=22,
            color=plot_colors[i],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.7)
    ax.set_ylabel("Mean Cultural Score by Country")
    ax.set_title("Distribution of Cultural Scores across Countries")
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_names, rotation=25, ha="right", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_country_score_violin.png"))



def plot_cultural_arousal(all_results, shared_plots_dir):
    """Bar chart: mean score per arousal level, grouped by model."""
    models = [r for r in all_results if r.get("cultural") and not r.get("_is_baseline")]
    if not models:
        return

    arousal_levels = sorted(models[0]["cultural"].get("arousal", {}).keys(), key=int)
    if not arousal_levels:
        return

    names = [short_name(r["_name"]) for r in models]
    colors = _result_colors(models)
    x = np.arange(len(arousal_levels))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(models):
        vals = [r["cultural"]["arousal"].get(a, {}).get("mean", 0) for a in arousal_levels]
        ax.bar(x + i * width, vals, width, label=names[i], color=colors[i])

    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Arousal Level")
    ax.set_title("Cultural Evaluation — Score by Arousal Level")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(arousal_levels)
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_score_by_arousal.png"))


def plot_cultural_country_deviation_heatmap(all_results, shared_plots_dir):
    """Heatmap: country score minus each model's mean score."""
    models = [r for r in all_results if r.get("cultural") and not r.get("_is_baseline")]
    if not models:
        return

    countries = sorted({
        country
        for r in models
        for country in r.get("cultural", {}).get("countries", {}).keys()
    })
    if not countries:
        return

    names = [short_name(r["_name"]) for r in models]
    data = []
    for country in countries:
        row = []
        for r in models:
            country_data = r.get("cultural", {}).get("countries", {})
            model_scores = [
                c.get("score_mean")
                for c in country_data.values()
                if c.get("score_mean") is not None
            ]
            country_score = country_data.get(country, {}).get("score_mean")
            if not model_scores or country_score is None:
                row.append(np.nan)
            else:
                row.append(country_score - float(np.mean(model_scores)))
        data.append(row)
    data = np.array(data, dtype=float)
    if np.all(np.isnan(data)):
        return

    max_abs = float(np.nanmax(np.abs(data)))
    if max_abs == 0:
        max_abs = 1.0
    cmap = BLUE_ORANGE_CMAP.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2.0), max(6, len(countries) * 0.45)))
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(countries)))
    ax.set_yticklabels(countries, fontsize=9)
    ax.set_title("Geopolitical Score Deviation by Country")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Country score - model mean")

    for i in range(len(countries)):
        for j in range(len(names)):
            value = data[i, j]
            if np.isnan(value):
                continue
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.45 else "black"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_country_score_deviation_heatmap.png"))


def plot_cultural_arousal_correlation_lollipop(all_results, shared_plots_dir):
    """Lollipop plot: reward correlation against emotional arousal."""
    models = [
        r for r in all_results
        if r.get("cultural", {}).get("score_vs_arousal") and not r.get("_is_baseline")
    ]
    if not models:
        return

    names = [short_name(r["_name"]) for r in models]
    pearson = [r["cultural"]["score_vs_arousal"].get("pearson") for r in models]
    spearman = [r["cultural"]["score_vs_arousal"].get("spearman") for r in models]
    if all(v is None for v in pearson + spearman):
        return

    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, max(5, len(models) * 0.55)))
    ax.axvspan(-0.05, 0.05, color="#E8ECF3", alpha=0.8, zorder=0)
    ax.axvline(0, color="black", linewidth=1)

    for values, offset, color, marker, label in [
        (pearson, -0.12, "#0072B2", "o", "Pearson"),
        (spearman, 0.12, "#E69F00", "s", "Spearman"),
    ]:
        for i, value in enumerate(values):
            if value is None:
                continue
            ax.hlines(y[i] + offset, 0, value, color=color, linewidth=2, alpha=0.8)
            ax.scatter(value, y[i] + offset, color=color, marker=marker, s=70, zorder=3,
                       label=label if i == 0 else "")
            ax.text(value, y[i] + offset, f" {value:+.3f}", va="center", fontsize=8,
                    ha="left" if value >= 0 else "right")

    max_abs = max(abs(v) for v in pearson + spearman if v is not None)
    xlim = max(0.1, max_abs * 1.35)
    ax.set_xlim(-xlim, xlim)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Correlation with emotional arousal")
    ax.set_title("Score Correlation against Emotional Arousal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "cultural_arousal_correlation_lollipop.png"))


def plot_scoring_spearman_heatmap(all_results, shared_plots_dir):
    """Heatmap: Spearman per attribute × model. Skip models with no scoring attributes."""
    # Filter out models that have no scoring attribute data (e.g. qwen3 baseline)
    filtered = [r for r in all_results if any(
        r.get("scoring", {}).get("attributes", {}).get(a) for a in ATTRIBUTES
    )]
    if not filtered:
        return
    names = [axis_model_label(r["_name"]) for r in filtered]
    attrs_with_data = [a for a in ATTRIBUTES if any(
        r.get("scoring", {}).get("attributes", {}).get(a) for r in filtered
    )]
    if not attrs_with_data:
        return

    display_attrs = _attributes_with_group_gaps(attrs_with_data)
    data = []
    for r in filtered:
        row = []
        for a in display_attrs:
            if a is None:
                row.append(np.nan)
                continue
            ad = r.get("scoring", {}).get("attributes", {}).get(a)
            row.append(ad["spearman"] if ad else 0)
        data.append(row)
    data = np.array(data, dtype=float)

    cmap = BLUE_ORANGE_CMAP.copy()
    cmap.set_bad(color="white")

    gap_width = 0.28
    x_edges = [0.0]
    x_centers = []
    group_ranges = []
    current_group = None
    group_start = None
    last_attr_right = None
    for a in display_attrs:
        left = x_edges[-1]
        width = gap_width if a is None else 1.0
        right = left + width
        x_edges.append(right)
        x_centers.append((left + right) / 2)

        if a is None:
            continue

        prefix = _attribute_prefix(a)
        if prefix != current_group:
            if current_group is not None:
                group_ranges.append((group_start, last_attr_right))
            current_group = prefix
            group_start = left
        last_attr_right = right
    if current_group is not None:
        group_ranges.append((group_start, last_attr_right))

    y_edges = np.arange(len(names) + 1)
    fig, ax = plt.subplots(figsize=(max(12, len(attrs_with_data) * 0.6), max(4, len(names) * 0.8)))
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(data),
        cmap=cmap,
        vmin=-0.1,
        vmax=0.9,
        edgecolors="none",
    )
    ax.invert_yaxis()

    tick_positions = [x_centers[i] for i, a in enumerate(display_attrs) if a is not None]
    tick_labels = [a for a in display_attrs if a is not None]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Scoring Spearman by Attribute (Heatmap)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for start, end in group_ranges:
        ax.add_patch(Rectangle((start, 0), end - start, len(names),
                               fill=False, edgecolor="black", linewidth=1.2))
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman Correlation")
    # Add text annotations
    for i in range(len(names)):
        for j, a in enumerate(display_attrs):
            if a is None:
                continue
            v = data[i, j]
            if np.isnan(v):
                continue
            rgba = im.cmap(im.norm(v))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.45 else "black"
            ax.text(x_centers[j], i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_spearman_heatmap.png"))


def plot_scoring_80pct_vs_100pct_paired(all_results, shared_plots_dir):
    """Paired scatter plot: 80% vs 100% Spearman per domain, lines connecting pairs."""
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
    colors = _result_colors(models)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(models):
        color = colors[i]
        for j, d in enumerate(domains):
            v80 = r["scoring_80pct"].get("domains", {}).get(d, {}).get("spearman", 0)
            v100 = r["scoring_100pct"].get("domains", {}).get(d, {}).get("spearman", 0)
            x_pos = j + i * 0.15 - 0.15
            ax.plot([x_pos, x_pos], [v80, v100], color=color, linewidth=2, alpha=0.7)
            ax.scatter(x_pos, v80, color=color, marker="^", s=60, zorder=5, alpha=0.5,
                       label=f"{names[i]} 80%" if j == 0 else "")
            ax.scatter(x_pos, v100, color=color, marker="o", s=60, zorder=5,
                       label=f"{names[i]} 100%" if j == 0 else "")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Scoring Spearman: 80% vs 100% (paired)")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, os.path.join(shared_plots_dir, "scoring_80pct_vs_100pct_paired.png"))


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
    colors = _result_colors(models)
    n = len(countries)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i, r in enumerate(models):
        vals = [r["cultural"]["countries"].get(c, {}).get("score_mean", 0) for c in countries]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=names[i], color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

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
    colors = _result_colors(models)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(models):
        vals = [r["cultural"]["arousal"].get(a, {}).get("mean", 0) for a in arousal_levels]
        color = colors[i]
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
        for diff in _ordered_difficulties(all_diffs):
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
        plot_scoring_80pct_vs_100pct(all_results, shared_plots_dir)
        plot_scoring_80pct_vs_100pct_paired(all_results, shared_plots_dir)

    # Cultural evaluation (data/test)
    has_cultural = any(r.get("cultural") for r in all_results)
    if has_cultural:
        plot_cultural_score_by_country(all_results, shared_plots_dir)
        plot_cultural_country_score_violin(all_results, shared_plots_dir)
        plot_cultural_arousal(all_results, shared_plots_dir)
        plot_cultural_country_deviation_heatmap(all_results, shared_plots_dir)
        plot_cultural_arousal_correlation_lollipop(all_results, shared_plots_dir)

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
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to compare. Default: auto-discover from model/.")
    parser.add_argument("--model_parent_dir", type=str, default=DEFAULT_MODEL_PARENT_DIR,
                        help="Parent directory for packaged models.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip plot generation.")
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip loading eval_baseline.json for each model.")
    args = parser.parse_args()
    config = load_yaml_config(args.config_path)
    args = apply_section_overrides(args, config.get("compare_models", {}))

    # Discover models
    model_names = args.models or discover_models(args.model_parent_dir)
    if not model_names:
        print("No packaged models found. Run stage-3_package_model.py first.")
        sys.exit(1)

    print(f"Models to compare: {model_names}")

    # Load cached results
    all_results = []
    for name in model_names:
        baseline_path = _baseline_results_path(args.model_parent_dir, name)
        if not args.no_baselines and os.path.isfile(baseline_path):
            print(f"  Loading: {baseline_path}")
            rb = load_results(baseline_path)
            rb["_name"] = f"{name} (baseline)"
            rb["_base_name"] = name
            rb["_is_baseline"] = True
            all_results.append(rb)

        json_path = _results_path(args.model_parent_dir, name)
        if os.path.isfile(json_path):
            print(f"  Loading: {json_path}")
            r = load_results(json_path)
            r["_name"] = name
            all_results.append(r)
        else:
            print(f"  WARNING: No results for {name} at {json_path}, skipping.")
            print(f"           Run: python3 evaluate.py --model_name {name}")

    if len(all_results) < 1:
        print("No results to compare. Run evaluate.py for each model first.")
        sys.exit(1)

    # Print comparison tables
    if any("scoring" in r for r in all_results):
        print_scoring_attributes_table(all_results)
        print_scoring_domain_table(all_results)

    if any("preference" in r for r in all_results):
        print_preference_table(all_results)

    if any("scoring" in r for r in all_results):
        print_global_score_table(all_results)

    if any(r.get("cultural") and not r.get("_is_baseline") for r in all_results):
        print_cultural_fairness_tables(all_results)

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
