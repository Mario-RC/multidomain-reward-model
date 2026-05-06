"""Evaluate a packaged multi-domain reward model on test data.

Two evaluation modes:
  1. Scoring  – per-attribute regression quality (Multi-Domain-Data-Scoring)
  2. Preference – chosen-vs-rejected accuracy  (Multi-Domain-Data-Preference-Pairs)
"""

import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import ArgumentParser
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

from datetime import datetime
from modeling_custom import RewardModelWithGating
from config_utils import load_yaml_config, apply_section_overrides
from attributes import ATTRIBUTES, DOMAIN_PREFIXES
from utils import _resolve_inference_model_path, load_jsonl_test, _score_messages, load_cultural_test, parse_cultural_conversation


# ---------------------------------------------------------------------------
# Scoring evaluation  (Multi-Domain-Data-Scoring)
# ---------------------------------------------------------------------------

def evaluate_scoring(model, tokenizer, data_path, device, max_length, max_samples):
    records = load_jsonl_test(data_path)
    if not records:
        print(f"No test records found in {data_path}")
        return {}
    if max_samples and max_samples < len(records):
        random.seed(42)
        random.shuffle(records)
        records = records[:max_samples]

    print(f"\n{'=' * 70}")
    print(f"  SCORING EVALUATION — {len(records)} test samples")
    print(f"{'=' * 70}")

    # Per-attribute collectors (records only have scores for their domain).
    attr_pred: dict[str, list[float]] = {a: [] for a in ATTRIBUTES}
    attr_true: dict[str, list[float]] = {a: [] for a in ATTRIBUTES}
    all_scores: list[float] = []
    evaluated = 0
    skipped = 0

    for record in tqdm(records, desc="Scoring"):
        messages = record.get("messages")
        scores = record.get("scores", {})
        if not messages or not scores:
            skipped += 1
            continue

        # Require at least one valid attribute score.
        valid_attrs = [(i, a) for i, a in enumerate(ATTRIBUTES) if scores.get(a) is not None]
        if not valid_attrs:
            skipped += 1
            continue

        try:
            out = _score_messages(model, tokenizer, messages, device, max_length)
            rewards = out.rewards.cpu().float().squeeze(0).numpy()
            all_scores.append(out.score.cpu().float().item())
            for idx, attr in valid_attrs:
                attr_pred[attr].append(float(rewards[idx]))
                attr_true[attr].append(float(scores[attr]))
            evaluated += 1
        except Exception:
            skipped += 1
            continue

    if skipped:
        print(f"  Skipped: {skipped}")
    print(f"  Evaluated: {evaluated}")

    if evaluated == 0:
        print("  No valid samples evaluated.")
        return {}

    # Per-attribute metrics
    header = f"  {'Attribute':<42} {'N':>6} {'MSE':>8} {'Pearson':>8} {'Spearman':>9}"
    print(f"\n{header}")
    print(f"  {'-' * 78}")

    mses, pearsons, spearmans = [], [], []
    domain_metrics: dict[str, list[tuple[float, float, float]]] = {}
    results_attr = {}

    for attr in ATTRIBUTES:
        p = np.array(attr_pred[attr])
        t = np.array(attr_true[attr])
        n = len(p)
        if n < 2:
            print(f"  {attr:<42} {n:>6}      —        —         —")
            continue
        mse = float(np.mean((p - t) ** 2))
        r_p = pearsonr(p, t).statistic if np.std(p) > 0 and np.std(t) > 0 else 0.0
        r_s = spearmanr(p, t).statistic if np.std(p) > 0 and np.std(t) > 0 else 0.0
        mses.append(mse)
        pearsons.append(r_p)
        spearmans.append(r_s)
        results_attr[attr] = {"n": n, "mse": round(mse, 6), "pearson": round(float(r_p), 6), "spearman": round(float(r_s), 6)}
        print(f"  {attr:<42} {n:>6} {mse:>8.4f} {r_p:>8.4f} {r_s:>9.4f}")

        for domain_name, prefix in DOMAIN_PREFIXES.items():
            if attr.startswith(prefix):
                domain_metrics.setdefault(domain_name, []).append((mse, r_p, r_s))

    if mses:
        print(f"  {'-' * 78}")
        print(f"  {'AVERAGE':<42} {'':>6} {np.mean(mses):>8.4f} {np.mean(pearsons):>8.4f} {np.mean(spearmans):>9.4f}")

    # Per-domain summary
    results_domain = {}
    if domain_metrics:
        print(f"\n  {'Domain':<20} {'MSE':>8} {'Pearson':>8} {'Spearman':>9}")
        print(f"  {'-' * 49}")
        for domain_name in sorted(domain_metrics):
            vals = domain_metrics[domain_name]
            dm = float(np.mean([v[0] for v in vals]))
            dp = float(np.mean([v[1] for v in vals]))
            ds = float(np.mean([v[2] for v in vals]))
            results_domain[domain_name] = {"mse": round(dm, 6), "pearson": round(dp, 6), "spearman": round(ds, 6)}
            print(f"  {domain_name:<20} {dm:>8.4f} {dp:>8.4f} {ds:>9.4f}")

    # Global score distribution
    scores_arr = np.array(all_scores)
    print(f"\n  Global score stats — mean: {scores_arr.mean():.4f}  std: {scores_arr.std():.4f}"
          f"  min: {scores_arr.min():.4f}  max: {scores_arr.max():.4f}")

    return {
        "evaluated": evaluated,
        "skipped": skipped,
        "attributes": results_attr,
        "domains": results_domain,
        "average": {
            "mse": round(float(np.mean(mses)), 6) if mses else None,
            "pearson": round(float(np.mean(pearsons)), 6) if pearsons else None,
            "spearman": round(float(np.mean(spearmans)), 6) if spearmans else None,
        },
        "global_score": {
            "mean": round(float(scores_arr.mean()), 6),
            "std": round(float(scores_arr.std()), 6),
            "min": round(float(scores_arr.min()), 6),
            "max": round(float(scores_arr.max()), 6),
        },
    }


# ---------------------------------------------------------------------------
# Preference evaluation  (Multi-Domain-Data-Preference-Pairs)
# ---------------------------------------------------------------------------

def evaluate_preference(model, tokenizer, data_path, device, max_length, max_samples):
    records = load_jsonl_test(data_path)
    if not records:
        print(f"No test records found in {data_path}")
        return {}
    if max_samples and max_samples < len(records):
        random.seed(42)
        random.shuffle(records)
        records = records[:max_samples]

    print(f"\n{'=' * 70}")
    print(f"  PREFERENCE EVALUATION — {len(records)} test pairs")
    print(f"{'=' * 70}")

    correct = 0
    ties = 0
    total = 0
    domain_stats: dict[str, list[int, int, int]] = {}
    difficulty_stats: dict[str, list[int, int, int]] = {}
    skipped = 0
    margins: list[float] = []

    for record in tqdm(records, desc="Preference"):
        messages = record.get("messages", [])
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not messages or not chosen or not rejected:
            skipped += 1
            continue

        # chosen / rejected are lists of message dicts
        chosen_msgs = messages + (chosen if isinstance(chosen, list) else [{"role": "assistant", "content": chosen}])
        rejected_msgs = messages + (rejected if isinstance(rejected, list) else [{"role": "assistant", "content": rejected}])

        try:
            c_score = _score_messages(model, tokenizer, chosen_msgs, device, max_length).score.cpu().float().item()
            r_score = _score_messages(model, tokenizer, rejected_msgs, device, max_length).score.cpu().float().item()
        except Exception:
            skipped += 1
            continue

        is_correct = c_score > r_score
        is_tie = c_score == r_score
        correct += int(is_correct)
        ties += int(is_tie)
        total += 1
        margins.append(c_score - r_score)

        metadata = record.get("metadata", {})
        domain = metadata.get("domain", "unknown")
        difficulty = metadata.get("difficulty", "unknown")

        for bucket, key in [(domain_stats, domain), (difficulty_stats, difficulty)]:
            if key not in bucket:
                bucket[key] = [0, 0, 0]
            bucket[key][0] += int(is_correct)
            bucket[key][1] += 1
            bucket[key][2] += int(is_tie)

    if skipped:
        print(f"  Skipped: {skipped}")

    if total == 0:
        print("  No valid pairs evaluated.")
        return {}

    margins_arr = np.array(margins)
    print(f"\n  Overall accuracy: {correct}/{total}  ({100 * correct / total:.2f}%)")
    print(f"  Ties (chosen == rejected): {ties}")
    print(f"  Margin stats — mean: {margins_arr.mean():.4f}  std: {margins_arr.std():.4f}")

    # Per-domain
    results_domain = {}
    if domain_stats:
        print(f"\n  {'Domain':<25} {'Accuracy':>10} {'Correct':>9} {'Total':>7} {'Ties':>6}")
        print(f"  {'-' * 61}")
        for d in sorted(domain_stats):
            c, t, ti = domain_stats[d]
            results_domain[d] = {"accuracy": round(100 * c / t, 4), "correct": c, "total": t, "ties": ti}
            print(f"  {d:<25} {100 * c / t:>9.2f}% {c:>9} {t:>7} {ti:>6}")

    # Per-difficulty
    results_difficulty = {}
    if difficulty_stats:
        print(f"\n  {'Difficulty':<25} {'Accuracy':>10} {'Correct':>9} {'Total':>7} {'Ties':>6}")
        print(f"  {'-' * 61}")
        for d in sorted(difficulty_stats):
            c, t, ti = difficulty_stats[d]
            results_difficulty[d] = {"accuracy": round(100 * c / t, 4), "correct": c, "total": t, "ties": ti}
            print(f"  {d:<25} {100 * c / t:>9.2f}% {c:>9} {t:>7} {ti:>6}")

    return {
        "total": total,
        "correct": correct,
        "ties": ties,
        "skipped": skipped,
        "accuracy": round(100 * correct / total, 4),
        "margin_mean": round(float(margins_arr.mean()), 6),
        "margin_std": round(float(margins_arr.std()), 6),
        "domains": results_domain,
        "difficulty": results_difficulty,
    }


# ---------------------------------------------------------------------------
# Cultural evaluation  (data/test/*.json)
# ---------------------------------------------------------------------------

MU_ATTRIBUTES = [a for a in ATTRIBUTES if a.startswith("mu_")]
MU_INDICES = [ATTRIBUTES.index(a) for a in MU_ATTRIBUTES]


def evaluate_cultural(model, tokenizer, data_dir, device, max_length):
    """Score cultural conversations and report per-country / per-arousal statistics."""
    records = load_cultural_test(data_dir)
    if not records:
        print(f"No cultural test records found in {data_dir}")
        return {}

    print(f"\n{'=' * 70}")
    print(f"  CULTURAL EVALUATION — {len(records)} conversations")
    print(f"{'=' * 70}")

    country_scores: dict[str, list[float]] = {}
    country_mu: dict[str, dict[str, list[float]]] = {}
    arousal_scores: dict[int, list[float]] = {}
    all_scores: list[float] = []
    all_arousal: list[int] = []
    skipped = 0

    for record in tqdm(records, desc="Cultural"):
        messages = record.get("messages")
        if not messages or not isinstance(messages, list):
            messages = parse_cultural_conversation(record)
        if len(messages) < 2:
            skipped += 1
            continue

        try:
            out = _score_messages(model, tokenizer, messages, device, max_length)
            score = out.score.cpu().float().item()
            rewards = out.rewards.cpu().float().squeeze(0).numpy()
        except Exception:
            skipped += 1
            continue

        dm = record.get("domain_metadata") or {}
        country = dm.get("country_1", record.get("country_1", "unknown"))
        arousal = dm.get("arousal_score", record.get("arousal_score"))
        if isinstance(arousal, str):
            arousal = int(arousal) if arousal.isdigit() else None

        all_scores.append(score)
        country_scores.setdefault(country, []).append(score)

        if country not in country_mu:
            country_mu[country] = {a: [] for a in MU_ATTRIBUTES}
        for a, idx in zip(MU_ATTRIBUTES, MU_INDICES):
            country_mu[country][a].append(float(rewards[idx]))

        if arousal is not None:
            arousal_scores.setdefault(arousal, []).append(score)
            all_arousal.append(arousal)

    if skipped:
        print(f"  Skipped: {skipped}")
    print(f"  Evaluated: {len(all_scores)}")

    if not all_scores:
        return {}

    scores_arr = np.array(all_scores)
    print(f"\n  Global score — mean: {scores_arr.mean():.4f}  std: {scores_arr.std():.4f}"
          f"  min: {scores_arr.min():.4f}  max: {scores_arr.max():.4f}")

    # Per-country table
    results_country = {}
    print(f"\n  {'Country':<12} {'N':>4} {'Score':>8} {'Std':>8}", end="")
    for a in MU_ATTRIBUTES:
        print(f" {a.replace('mu_',''):>12}", end="")
    print()
    print(f"  {'-' * (36 + 13 * len(MU_ATTRIBUTES))}")

    for c in sorted(country_scores):
        cs = np.array(country_scores[c])
        row = {"n": len(cs), "score_mean": round(float(cs.mean()), 4), "score_std": round(float(cs.std()), 4)}
        print(f"  {c:<12} {len(cs):>4} {cs.mean():>8.4f} {cs.std():>8.4f}", end="")
        mu_means = {}
        for a in MU_ATTRIBUTES:
            vals = np.array(country_mu[c][a])
            m = float(vals.mean())
            mu_means[a] = round(m, 4)
            print(f" {m:>12.4f}", end="")
        row["mu_attributes"] = mu_means
        results_country[c] = row
        print()

    # Per-arousal level
    results_arousal = {}
    if arousal_scores:
        print(f"\n  {'Arousal':>8} {'N':>5} {'Score Mean':>12} {'Score Std':>11}")
        print(f"  {'-' * 40}")
        for level in sorted(arousal_scores):
            vals = np.array(arousal_scores[level])
            results_arousal[level] = {"n": len(vals), "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)}
            print(f"  {level:>8} {len(vals):>5} {vals.mean():>12.4f} {vals.std():>11.4f}")

    # Correlation score vs arousal
    corr_info = {}
    if len(all_arousal) >= 3:
        a_arr = np.array(all_arousal, dtype=float)
        s_arr = np.array(all_scores[:len(all_arousal)])
        r_p = pearsonr(a_arr, s_arr).statistic
        r_s = spearmanr(a_arr, s_arr).statistic
        corr_info = {"pearson": round(float(r_p), 4), "spearman": round(float(r_s), 4)}
        print(f"\n  Score vs Arousal — Pearson: {r_p:.4f}  Spearman: {r_s:.4f}")

    return {
        "evaluated": len(all_scores),
        "skipped": skipped,
        "global_score": {
            "mean": round(float(scores_arr.mean()), 4),
            "std": round(float(scores_arr.std()), 4),
            "min": round(float(scores_arr.min()), 4),
            "max": round(float(scores_arr.max()), 4),
        },
        "countries": results_country,
        "arousal": results_arousal,
        "score_vs_arousal": corr_info,
    }


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _generate_plots(results, plots_dir):
    """Generate per-model plots from evaluation results."""
    model_name = os.path.basename(results.get("model", "").rstrip("/"))

    # Scoring: Spearman per attribute (use 100pct if available, else 80pct)
    scoring = results.get("scoring_100pct") or results.get("scoring_80pct") or {}
    attrs_data = scoring.get("attributes", {})
    if attrs_data:
        attrs = [a for a in ATTRIBUTES if a in attrs_data]
        vals = [attrs_data[a]["spearman"] for a in attrs]
        if attrs:
            fig, ax = plt.subplots(figsize=(8, max(5, len(attrs) * 0.35)))
            colors = ["#4CAF50" if v >= 0.5 else "#FF9800" if v >= 0.3 else "#F44336" for v in vals]
            ax.barh(attrs, vals, color=colors)
            ax.set_xlabel("Spearman Correlation")
            ax.set_title(f"{model_name} — Spearman by Attribute")
            ax.invert_yaxis()
            ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
            ax.legend(fontsize=8)
            _save_fig(fig, os.path.join(plots_dir, "spearman_by_attribute.png"))

    # Cultural: attributes by country (data/test)
    cultural = results.get("cultural", {})
    countries_data = cultural.get("countries", {})
    if countries_data:
        countries = sorted(countries_data.keys())
        sample = next(iter(countries_data.values()), {})
        mu_attrs = sorted(sample.get("mu_attributes", {}).keys())
        if mu_attrs:
            x = np.arange(len(countries))
            width = 0.8 / max(len(mu_attrs), 1)
            fig, ax = plt.subplots(figsize=(max(10, len(countries) * 1.2), 5))
            for j, attr in enumerate(mu_attrs):
                vals = [countries_data.get(c, {}).get("mu_attributes", {}).get(attr, 0) for c in countries]
                ax.bar(x + j * width, vals, width, label=attr.replace("mu_", ""))
            ax.set_ylabel("Attribute Score")
            ax.set_title(f"{model_name} — Cultural Attributes by Country")
            ax.set_xticks(x + width * (len(mu_attrs) - 1) / 2)
            ax.set_xticklabels(countries, rotation=45, ha="right", fontsize=8)
            ax.legend(fontsize=7)
            fig.tight_layout()
            _save_fig(fig, os.path.join(plots_dir, "cultural_attributes_by_country.png"))

    # Preference: accuracy per domain
    pref = results.get("preference", {})
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
        ax.set_title(f"{model_name} — Preference Accuracy by Domain")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=30)
        _save_fig(fig, os.path.join(plots_dir, "preference_by_domain.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n### Evaluate started at {datetime.now().isoformat()} ###")
    parser = ArgumentParser(description="Evaluate packaged multi-domain reward model on test data.")

    # Model
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Override for packaged model path.")
    parser.add_argument("--model_parent_dir", type=str, default="model", help="Packaged model parent directory.")
    parser.add_argument("--model_name", type=str, default=None, help="Packaged model directory name.")

    # Data
    parser.add_argument("--scoring_data_path", type=str, default=None, help="Path to Multi-Domain-Data-Scoring[.jsonl]. Default: from config or data/dataset/Multi-Domain-Data-Scoring.")
    parser.add_argument("--preference_data_path", type=str, default=None, help="Path to Multi-Domain-Data-Preference-Pairs[.jsonl]. Default: from config or data/dataset/Multi-Domain-Data-Preference-Pairs.")

    # Eval control
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length for tokenization.")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap per evaluation (for quick debugging).")
    parser.add_argument("--skip_scoring", action="store_true", help="Skip scoring evaluation.")
    parser.add_argument("--skip_preference", action="store_true", help="Skip preference evaluation.")
    parser.add_argument("--eval", type=str, default=None, help="Path to test data directory or JSONL file (e.g. data/test). Evaluates with scoring metrics.")
    parser.add_argument("--output_json", type=str, default=None, help="Save evaluation results to a JSON file.")

    args = parser.parse_args()
    config = load_yaml_config(args.config_path)
    args = apply_section_overrides(
        args, config.get("inference", {}),
        skip_keys={"model_path", "model_parent_dir", "model_name"},
    )

    # Resolve data paths from config fallbacks
    if not args.scoring_data_path:
        s1_cfg = config.get("stage_1_prepare", {})
        args.scoring_data_path = s1_cfg.get("dataset_path", "data/dataset/Multi-Domain-Data-Scoring")

    if not args.preference_data_path:
        args.preference_data_path = "data/dataset/Multi-Domain-Data-Preference-Pairs"

    # Resolve single model path
    model_path = _resolve_inference_model_path(config, args.model_path, args.model_parent_dir, args.model_name)
    model_label = args.model_name or os.path.basename(model_path.rstrip("/"))

    # Load training metadata saved by stage-3 (discover stage-1 / stage-2 paths).
    metadata_file = os.path.join(model_path, "training_metadata.json")
    if os.path.isfile(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            training_metadata = json.load(f)
        print(f"  Training metadata:")
        print(f"    Base model:      {training_metadata.get('base_model_path')}")
        print(f"    Stage-1 weights: {training_metadata.get('stage_1_weights_path')}")
        print(f"    Stage-2 weights: {training_metadata.get('stage_2_weights_path')}")
    else:
        training_metadata = {}

    # Derive stage-1 weight paths (100pct and 80pct) from metadata.
    stage1_100pct_path = training_metadata.get("stage_1_weights_path")
    stage1_80pct_path = stage1_100pct_path.replace("_100pct.pt", "_80pct.pt") if stage1_100pct_path else None

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    print(f"\n{'#' * 70}")
    print(f"  Evaluating: {model_label}")
    print(f"  Model: {model_path}")
    print(f"{'#' * 70}")

    print(f"Loading model: {model_path}")
    model = RewardModelWithGating.from_pretrained(
        model_path,
        device_map={"": 0} if use_cuda else None,
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model.eval()
    device = next(model.parameters()).device

    results = {"model": model_path}
    if training_metadata:
        results["training_metadata"] = training_metadata

    # ------------------------------------------------------------------
    # Scoring evaluation — stage-1 regression weights (80pct then 100pct)
    # Evaluate 80pct first, then load 100pct so it stays for stage-3.
    # ------------------------------------------------------------------
    if not args.skip_scoring:
        swapped_to_80 = False

        # --- 80pct scoring ---
        if stage1_80pct_path and os.path.isfile(stage1_80pct_path):
            print(f"\n  Loading 80pct weights: {stage1_80pct_path}")
            payload_80 = torch.load(stage1_80pct_path, map_location="cpu", weights_only=True)
            w80 = payload_80["weight"] if isinstance(payload_80, dict) and "weight" in payload_80 else payload_80
            model.regression_layer.weight.data.copy_(w80.to(model.regression_layer.weight.dtype))
            swapped_to_80 = True
            results["scoring_80pct"] = evaluate_scoring(
                model, tokenizer, args.scoring_data_path, device, args.max_length, args.max_samples
            )
        elif stage1_80pct_path:
            print(f"  WARNING: 80pct weights not found at {stage1_80pct_path}, skipping 80pct scoring.")

        # --- 100pct scoring (stays loaded for stage-3) ---
        if swapped_to_80:
            # Reload 100pct from disk since we swapped to 80pct.
            print(f"\n  Loading 100pct weights: {stage1_100pct_path}")
            payload_100 = torch.load(stage1_100pct_path, map_location="cpu", weights_only=True)
            w100 = payload_100["weight"] if isinstance(payload_100, dict) and "weight" in payload_100 else payload_100
            model.regression_layer.weight.data.copy_(w100.to(model.regression_layer.weight.dtype))
        else:
            print(f"\n  Stage-1 scoring with 100pct weights (packaged)")
        results["scoring_100pct"] = evaluate_scoring(
            model, tokenizer, args.scoring_data_path, device, args.max_length, args.max_samples
        )

    # ------------------------------------------------------------------
    # Preference evaluation — full stage-3 model (regression + gating)
    # ------------------------------------------------------------------
    if not args.skip_preference:
        results["preference"] = evaluate_preference(
            model, tokenizer, args.preference_data_path, device, args.max_length, args.max_samples
        )

    # ------------------------------------------------------------------
    # Cultural evaluation — full stage-3 model (regression + gating)
    # ------------------------------------------------------------------
    if args.eval:
        results["cultural"] = evaluate_cultural(
            model, tokenizer, args.eval, device, args.max_length
        )

    # Generate per-model plots
    _generate_plots(results, os.path.join(model_path, "results", "plots"))

    # Save results
    auto_json = os.path.join(model_path, "results", "eval.json")
    os.makedirs(os.path.dirname(auto_json) or ".", exist_ok=True)
    with open(auto_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {auto_json}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Combined results saved to {args.output_json}")

    del model
    if use_cuda:
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    main()
