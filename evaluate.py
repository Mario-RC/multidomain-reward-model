"""Evaluate a packaged multi-domain reward model on test data.

Two evaluation modes:
  1. Scoring  – per-attribute regression quality (Multi-Domain-Data-Scoring)
  2. Preference – chosen-vs-rejected accuracy  (Multi-Domain-Data-Preference-Pairs)
"""

import json
import os
import random

import numpy as np
import torch
from argparse import ArgumentParser
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

from datetime import datetime
from modeling_custom import RewardModelWithGating
from config_utils import load_yaml_config
from attributes import ATTRIBUTES, DOMAIN_PREFIXES
from utils import _resolve_inference_model_path, _resolve_jsonl_path, load_jsonl_test, _score_messages


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
    parser.add_argument("--scoring_data_path", type=str, default=None, help="Path to Multi-Domain-Data-Scoring[.jsonl]. Default: from config or data/Multi-Domain-Data-Scoring.")
    parser.add_argument("--preference_data_path", type=str, default=None, help="Path to Multi-Domain-Data-Preference-Pairs[.jsonl]. Default: from config or data/Multi-Domain-Data-Preference-Pairs.")

    # Eval control
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length for tokenization.")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap per evaluation (for quick debugging).")
    parser.add_argument("--skip_scoring", action="store_true", help="Skip scoring evaluation.")
    parser.add_argument("--skip_preference", action="store_true", help="Skip preference evaluation.")
    parser.add_argument("--output_json", type=str, default=None, help="Save evaluation results to a JSON file.")

    args = parser.parse_args()
    config = load_yaml_config(args.config_path)

    # Resolve data paths from config fallbacks
    if not args.scoring_data_path:
        s1_cfg = config.get("stage_1_prepare", {})
        args.scoring_data_path = s1_cfg.get("dataset_path", "data/Multi-Domain-Data-Scoring")

    if not args.preference_data_path:
        args.preference_data_path = "data/Multi-Domain-Data-Preference-Pairs"

    # Load model
    path = _resolve_inference_model_path(config, args.model_path, args.model_parent_dir, args.model_name)
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    print(f"Loading model: {path}")
    model = RewardModelWithGating.from_pretrained(
        path,
        device_map={"": 0} if use_cuda else None,
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model.eval()
    device = next(model.parameters()).device

    # Run evaluations
    results = {"model": path}

    if not args.skip_scoring:
        results["scoring"] = evaluate_scoring(model, tokenizer, args.scoring_data_path, device, args.max_length, args.max_samples)

    if not args.skip_preference:
        results["preference"] = evaluate_preference(model, tokenizer, args.preference_data_path, device, args.max_length, args.max_samples)

    # Save results to JSON — always save to model results dir, optionally also to custom path.
    output_paths = []
    if args.output_json:
        output_paths.append(args.output_json)

    # Auto-save to model/<model_name>/results/eval.json
    model_name = args.model_name
    if not model_name:
        # Try to infer from the resolved path
        model_name = os.path.basename(path.rstrip("/"))
    auto_json = os.path.join(args.model_parent_dir, model_name, "results", "eval.json")
    if auto_json not in output_paths:
        output_paths.append(auto_json)

    for out_path in output_paths:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
