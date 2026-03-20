"""
Compare evaluation results across multiple packaged models.

This script automates the process of evaluating all packaged models and
producing side-by-side comparison tables. It serves two purposes:

  1. Running evaluate.py for each packaged model found in model/ (or a
     user-specified list), saving per-model results as JSON files.

  2. Loading those JSON results and printing comparative tables covering:
     - Preference accuracy (overall, per-domain, per-difficulty)
     - Scoring regression quality (Spearman, Pearson, MSE per domain/attribute)
     - Global score distribution statistics
     - A copy-paste ready markdown summary

The JSON results are saved in results/<model_name>.json and can be reused
with --cached to skip re-evaluation on subsequent runs.

Workflow:
  1. Train and package models with stage-1/2/3 scripts.
  2. Run this script to evaluate and compare all of them.
  3. Use --cached to re-print tables without re-running evaluation.

Usage:
    # Auto-discover all packaged models and evaluate:
    python3 compare_models.py

    # Evaluate specific models:
    python3 compare_models.py --models multi-domain-rm-llama-3-8b-it multi-domain-rm-gemma-2-9b-it

    # Use cached results (skip re-evaluation):
    python3 compare_models.py --cached

    # Quick test with limited samples:
    python3 compare_models.py --max_samples 50
"""

import json
import os
import sys
from argparse import ArgumentParser
from glob import glob

import numpy as np

from attributes import ATTRIBUTES, DOMAIN_PREFIXES


# Default directory where per-model JSON results are saved.
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Model discovery and evaluation
# ---------------------------------------------------------------------------

def discover_models(model_parent_dir="model"):
    """
    Auto-discover packaged model directories.

    A "packaged model" is a subdirectory of model_parent_dir that contains
    a config.json file (created by stage-3_package_model.py). This excludes
    intermediate artifacts like embeddings/ and gating_network/.

    Returns a sorted list of model directory names (not full paths).
    """
    models = []
    if not os.path.isdir(model_parent_dir):
        return models
    for name in sorted(os.listdir(model_parent_dir)):
        candidate = os.path.join(model_parent_dir, name)
        # Only directories with config.json are packaged models.
        if os.path.isfile(os.path.join(candidate, "config.json")):
            models.append(name)
    return models


def run_evaluation(model_name, model_parent_dir, output_json, max_samples=None, max_length=4096):
    """
    Run evaluate.py as a subprocess for a single model.

    This invokes evaluate.py with --output_json so that the results are
    saved to a JSON file. The subprocess approach ensures each model gets
    a clean GPU memory state (important when comparing models that may
    not fit in memory simultaneously).

    Args:
        model_name:       Name of the packaged model directory.
        model_parent_dir: Parent directory (default: "model").
        output_json:      Path to save the JSON results.
        max_samples:      Optional sample cap for quick testing.
        max_length:       Max tokenization length.

    Returns the output_json path on success, None on failure.
    """
    import subprocess
    cmd = [
        sys.executable, "evaluate.py",
        "--model_parent_dir", model_parent_dir,
        "--model_name", model_name,
        "--output_json", output_json,
        "--max_length", str(max_length),
    ]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]

    print(f"\n{'#' * 70}")
    print(f"  Evaluating: {model_name}")
    print(f"{'#' * 70}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: evaluate.py failed for {model_name} (exit code {result.returncode})")
        return None
    return output_json


def load_results(path):
    """Load a previously saved JSON results file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def short_name(model_name):
    """
    Shorten model names for table columns.
    e.g. "multi-domain-rm-llama-3-8b-it" → "llama-3-8b"
    """
    return model_name.replace("multi-domain-rm-", "").replace("-it", "")


# ---------------------------------------------------------------------------
# Table printers
#
# Each function takes all_results: a list of dicts, one per model.
# Each dict contains the JSON output of evaluate.py plus "_name".
# ---------------------------------------------------------------------------

def print_preference_table(all_results):
    """
    Print preference accuracy comparison.

    Shows overall accuracy (chosen score > rejected score), broken down
    by domain (empathy, coherence, etc.) and difficulty level (easy,
    medium, hard). Also shows the mean margin (chosen_score - rejected_score),
    which indicates how confidently the model separates chosen from rejected.
    """
    print(f"\n{'=' * 90}")
    print("  PREFERENCE ACCURACY COMPARISON")
    print(f"{'=' * 90}")

    names = [short_name(r["_name"]) for r in all_results]
    header = f"  {'':>20}" + "".join(f" {n:>15}" for n in names)
    print(header)
    print(f"  {'-' * (20 + 16 * len(names))}")

    # Overall accuracy row.
    row = f"  {'Overall':>20}"
    for r in all_results:
        pref = r.get("preference", {})
        acc = pref.get("accuracy")
        row += f" {acc:>14.2f}%" if acc is not None else f" {'—':>15}"
    print(row)

    # Per-domain rows: one row per domain found across all results.
    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("preference", {}).get("domains", {}).keys())

    for domain in sorted(all_domains):
        row = f"  {domain:>20}"
        for r in all_results:
            d = r.get("preference", {}).get("domains", {}).get(domain)
            if d:
                row += f" {d['accuracy']:>14.2f}%"
            else:
                row += f" {'—':>15}"
        print(row)

    # Per-difficulty breakdown (easy, medium, hard).
    all_diffs = set()
    for r in all_results:
        all_diffs.update(r.get("preference", {}).get("difficulty", {}).keys())

    if all_diffs:
        print(f"\n  {'--- By difficulty ---':>20}")
        for diff in sorted(all_diffs):
            row = f"  {diff:>20}"
            for r in all_results:
                d = r.get("preference", {}).get("difficulty", {}).get(diff)
                if d:
                    row += f" {d['accuracy']:>14.2f}%"
                else:
                    row += f" {'—':>15}"
            print(row)

    # Mean margin: how far apart chosen vs rejected scores are on average.
    # A larger margin means the model is more confident in its preferences.
    row = f"  {'Margin (mean)':>20}"
    for r in all_results:
        m = r.get("preference", {}).get("margin_mean")
        row += f" {m:>15.4f}" if m is not None else f" {'—':>15}"
    print(row)


def print_scoring_table(all_results):
    """
    Print scoring regression quality comparison.

    Shows two levels of detail:
      1. Domain-level summary: average Spearman and Pearson correlations
         across attributes within each domain.
      2. Per-attribute detail: Spearman correlation for each of the 23
         attributes individually.

    Spearman measures how well the predicted attribute rankings match the
    ground-truth rankings. Pearson measures linear agreement. Both range
    from -1 to 1, with higher being better.
    """
    print(f"\n{'=' * 90}")
    print("  SCORING EVALUATION COMPARISON (Spearman / Pearson / MSE)")
    print(f"{'=' * 90}")

    names = [short_name(r["_name"]) for r in all_results]

    # ── Domain-level summary ──
    print(f"\n  {'Domain':<20}" + "".join(f" {n:>22}" for n in names))
    print(f"  {'-' * (20 + 23 * len(names))}")

    all_domains = set()
    for r in all_results:
        all_domains.update(r.get("scoring", {}).get("domains", {}).keys())

    for domain in sorted(all_domains):
        row = f"  {domain:<20}"
        for r in all_results:
            d = r.get("scoring", {}).get("domains", {}).get(domain)
            if d:
                row += f"  S={d['spearman']:.3f} P={d['pearson']:.3f}"
            else:
                row += f" {'—':>22}"
        print(row)

    # Average across all domains.
    row = f"  {'AVERAGE':<20}"
    for r in all_results:
        avg = r.get("scoring", {}).get("average", {})
        if avg and avg.get("spearman") is not None:
            row += f"  S={avg['spearman']:.3f} P={avg['pearson']:.3f}"
        else:
            row += f" {'—':>22}"
    print(row)

    # ── Per-attribute detail (Spearman only for readability) ──
    print(f"\n  {'Attribute':<35}" + "".join(f" {n:>15}" for n in names) + "  (Spearman)")
    print(f"  {'-' * (35 + 16 * len(names))}")

    for attr in ATTRIBUTES:
        row = f"  {attr:<35}"
        for r in all_results:
            a = r.get("scoring", {}).get("attributes", {}).get(attr)
            if a:
                row += f" {a['spearman']:>15.4f}"
            else:
                row += f" {'—':>15}"
        print(row)


def print_global_score_table(all_results):
    """
    Print global score distribution comparison.

    The "global score" is the final scalar reward output of the packaged
    model (gating_weights * transformed_rewards). Comparing distributions
    across models helps spot scale differences — a model with very narrow
    std may not discriminate well between good and bad responses.
    """
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
            gs = r.get("scoring", {}).get("global_score", {})
            val = gs.get(stat)
            row += f" {val:>15.4f}" if val is not None else f" {'—':>15}"
        print(row)


def print_markdown_summary(all_results):
    """
    Print compact markdown tables ready for copy-paste into documentation.

    Produces two tables:
      1. Preference accuracy per domain.
      2. Scoring Spearman correlation per domain.
    """
    names = [short_name(r["_name"]) for r in all_results]

    print(f"\n{'=' * 90}")
    print("  MARKDOWN SUMMARY (copy-paste ready)")
    print(f"{'=' * 90}\n")

    # ── Preference accuracy table ──
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

    # ── Scoring Spearman table ──
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Compare evaluation results across packaged models.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to evaluate. Default: auto-discover from model/.")
    parser.add_argument("--model_parent_dir", type=str, default="model",
                        help="Parent directory for packaged models.")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                        help="Directory to store/load JSON results.")
    parser.add_argument("--cached", action="store_true",
                        help="Use cached JSON results only (skip evaluation). "
                             "Useful for regenerating tables without GPU.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per evaluation (for quick testing).")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max sequence length for tokenization.")
    args = parser.parse_args()

    # ── Step 1: Discover which models to evaluate ──
    model_names = args.models or discover_models(args.model_parent_dir)
    if not model_names:
        print("No packaged models found. Run stage-3_package_model.py first.")
        sys.exit(1)

    print(f"Models to compare: {model_names}")
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Step 2: Evaluate each model (or load cached results) ──
    all_results = []
    for name in model_names:
        json_path = os.path.join(args.results_dir, f"{name}.json")

        if args.cached:
            # --cached mode: load existing JSON without running evaluation.
            if os.path.isfile(json_path):
                print(f"  Loading cached: {json_path}")
                r = load_results(json_path)
            else:
                print(f"  WARNING: No cached results for {name}, skipping.")
                continue
        else:
            # Normal mode: run evaluate.py as a subprocess.
            # Each model runs in its own process to ensure clean GPU state.
            out = run_evaluation(name, args.model_parent_dir, json_path,
                                max_samples=args.max_samples, max_length=args.max_length)
            if out is None:
                continue
            r = load_results(json_path)

        # Tag the result with the model name for table formatting.
        r["_name"] = name
        all_results.append(r)

    if len(all_results) < 1:
        print("No results to compare.")
        sys.exit(1)

    # ── Step 3: Print comparison tables ──
    if any("preference" in r for r in all_results):
        print_preference_table(all_results)

    if any("scoring" in r for r in all_results):
        print_scoring_table(all_results)
        print_global_score_table(all_results)

    # Markdown tables are always printed (useful for documentation).
    print_markdown_summary(all_results)

    print(f"\n{'=' * 90}")
    print(f"  JSON results saved in: {args.results_dir}/")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
