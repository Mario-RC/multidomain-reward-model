"""Evaluate base reward models (as-is from HuggingFace) on test data.

Evaluation modes:
  1. Scoring   – native reward correlated against per-attribute labels
  2. Preference – chosen-vs-rejected accuracy
  3. Cultural  – score cultural conversations (per-country / per-arousal)
  4. Generative – BRRM-style 2-turn judge for preference
"""

import json
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import ArgumentParser
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from datetime import datetime
from attributes import ATTRIBUTES, DOMAIN_PREFIXES
from config_utils import load_yaml_config, apply_section_overrides
from utils import load_jsonl_test, _requires_remote_code, load_cultural_test, parse_cultural_conversation


@torch.no_grad()
def _get_reward_score(model, tokenizer, messages, device, max_length, pad_token_id):
    """Get the native scalar reward from a sequence-classification reward model."""
    encoding = tokenizer.apply_chat_template(
        messages, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    )
    if isinstance(encoding, torch.Tensor):
        input_ids = encoding.to(device)
        attention_mask = None
    else:
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else None

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # SequenceClassifierOutput: logits shape (batch, num_labels) — typically (1, 1) for RM
    reward = outputs.logits.cpu().float().squeeze().item()
    return reward


def evaluate_scoring(model, tokenizer, records, device, max_length, pad_token_id):
    """Evaluate native reward correlated against per-attribute labels."""
    print(f"\n{'=' * 70}")
    print(f"  BASELINE SCORING EVALUATION — {len(records)} test samples")
    print(f"{'=' * 70}")

    attr_pred: dict[str, list[float]] = {a: [] for a in ATTRIBUTES}
    attr_true: dict[str, list[float]] = {a: [] for a in ATTRIBUTES}
    evaluated = 0
    skipped = 0

    for record in tqdm(records, desc="Scoring"):
        messages = record.get("messages")
        scores = record.get("scores", {})
        if not messages or not scores:
            skipped += 1
            continue

        valid_attrs = [(i, a) for i, a in enumerate(ATTRIBUTES) if scores.get(a) is not None]
        if not valid_attrs:
            skipped += 1
            continue

        try:
            reward = _get_reward_score(model, tokenizer, messages, device, max_length, pad_token_id)
            for idx, attr in valid_attrs:
                attr_pred[attr].append(reward)
                attr_true[attr].append(float(scores[attr]))
            evaluated += 1
        except Exception as e:
            if skipped < 3:
                print(f"  [SKIP] {type(e).__name__}: {e}")
            skipped += 1
            continue

    if skipped:
        print(f"  Skipped: {skipped}")
    print(f"  Evaluated: {evaluated}")

    if evaluated == 0:
        print("  No valid samples evaluated.")
        return {}

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
    }


def evaluate_preference(model, tokenizer, records, device, max_length, pad_token_id):
    """Evaluate chosen-vs-rejected accuracy on Multi-Domain-Data-Preference-Pairs test split."""
    print(f"\n{'=' * 70}")
    print(f"  PREFERENCE EVALUATION — {len(records)} test pairs")
    print(f"{'=' * 70}")

    correct = 0
    ties = 0
    total = 0
    domain_stats: dict[str, list[int]] = {}
    difficulty_stats: dict[str, list[int]] = {}
    skipped = 0
    margins: list[float] = []

    for record in tqdm(records, desc="Preference"):
        messages = record.get("messages", [])
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not messages or not chosen or not rejected:
            skipped += 1
            continue

        chosen_msgs = messages + (chosen if isinstance(chosen, list) else [{"role": "assistant", "content": chosen}])
        rejected_msgs = messages + (rejected if isinstance(rejected, list) else [{"role": "assistant", "content": rejected}])

        try:
            c_score = _get_reward_score(model, tokenizer, chosen_msgs, device, max_length, pad_token_id)
            r_score = _get_reward_score(model, tokenizer, rejected_msgs, device, max_length, pad_token_id)
        except Exception as e:
            if skipped < 3:
                print(f"  [SKIP] {type(e).__name__}: {e}")
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

    results_domain = {}
    if domain_stats:
        print(f"\n  {'Domain':<25} {'Accuracy':>10} {'Correct':>9} {'Total':>7} {'Ties':>6}")
        print(f"  {'-' * 61}")
        for d in sorted(domain_stats):
            c, t, ti = domain_stats[d]
            results_domain[d] = {"accuracy": round(100 * c / t, 4), "correct": c, "total": t, "ties": ti}
            print(f"  {d:<25} {100 * c / t:>9.2f}% {c:>9} {t:>7} {ti:>6}")

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


BRRM_TURN1_TEMPLATE = """You are a response quality evaluator. Given the context of the conversation (the last turn is the User's query) and two responses from the Assistant, you should compare the difference of two model responses, select the most important cognitive abilities for this query, and analyze critical issues in each response.

**Context:**
[The Begin of Context]
{context}
[The End of Context]

**Responses:**
[The Begin of Response 1]
{response1}
[The End of Response 1]

[The Begin of Response 2]
{response2}
[The End of Response 2]

**Output Format:**
[Quality Assessment Focus]
Choose 1-3 abilities: Information Accuracy, Computational Precision, Logical Reasoning, Implementation Capability, Safety Awareness, Response Completeness, Instruction Adherence, Communication Clarity.
[End of Quality Assessment Focus]

[Quality Analysis for Response 1]
- Critical Issues: [Focus on chosen abilities, list specific errors/concerns, or "None identified"]
  * Information Accuracy: factual errors, source reliability, misinformation
  * Computational Precision: calculation errors, formula mistakes, step validity
  * Logical Reasoning: conclusion correctness (CRITICAL), logical flaws, reasoning gaps
  * Implementation Capability: functional errors (CRITICAL), security issues, inefficiency
  * Safety Awareness: harmful content (CRITICAL), inappropriate refusals, bias
  * Instruction Adherence: constraint violations, format errors, requirement misses
  * Response Completeness: missing content, insufficient detail, incomplete coverage
[End of Quality Analysis for Response 1]

[Quality Analysis for Response 2]
- Critical Issues: [Same format as above]
[End of Quality Analysis for Response 2]"""

BRRM_TURN2_TEMPLATE = r"""You are making final comparative judgments using established evaluation priorities. You have the conversation context, two responses to compare, and a detailed quality analysis from a previous evaluation.
Before scoring, analyze step by step. Different query types require different evaluation hierarchies. Please strictly follow the required output format.

**Evaluation Hierarchies:**
- **Accuracy-Critical** (factual, computational, technical): Correctness > Process > Presentation
- **Creative/Open-Ended** (writing, discussion): User Intent > Content Quality > Creativity
- **Instruction-Following** (constrained tasks): Adherence > Content > Clarity

#### Output Format Requirements ####
[The Begin of Analysis on Response 1]
[Apply appropriate evaluation hierarchy to the quality analysis findings]
[The End of Analysis on Response 1]

[The Begin of Analysis on Response 2]
[Apply appropriate evaluation hierarchy to the quality analysis findings]
[The End of Analysis on Response 2]

[The Begin of Ranking Score]
\boxed{1 or 2} (response that better meets the appropriate evaluation hierarchy)
[The End of Ranking Score]"""

BRRM_RANKING_PATTERN = re.compile(
    r"\[The Begin of Ranking Score\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Ranking Score\]",
    re.DOTALL,
)


def _format_context(messages: list[dict]) -> str:
    """Format a list of chat messages into a plain-text context string for BRRM."""
    parts = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        parts.append(f"{role}: {m.get('content', '')}")
    return "\n".join(parts)


def _extract_response_text(response: list[dict] | str) -> str:
    if isinstance(response, str):
        return response
    return "\n".join(m.get("content", "") for m in response)


@torch.no_grad()
def _brrm_judge(model, tokenizer, context_str, response1, response2, device, max_gen_tokens=8192):
    """Run BRRM 2-turn generation and return the preferred response index (1 or 2), or 0 on parse failure."""
    turn1_content = BRRM_TURN1_TEMPLATE.format(
        context=context_str, response1=response1, response2=response2,
    )
    messages = [{"role": "user", "content": turn1_content}]

    # Turn 1: quality assessment
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    )
    input_ids = (encoded.input_ids if hasattr(encoded, "input_ids") else encoded).to(device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=max_gen_tokens,
        temperature=1.0, top_p=0.95, top_k=20,
        do_sample=True, pad_token_id=tokenizer.eos_token_id,
    )
    turn1_response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)

    # Turn 2: scoring
    messages.append({"role": "assistant", "content": turn1_response})
    messages.append({"role": "user", "content": BRRM_TURN2_TEMPLATE})
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    )
    input_ids = (encoded.input_ids if hasattr(encoded, "input_ids") else encoded).to(device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids, attention_mask=attention_mask, max_new_tokens=max_gen_tokens,
        temperature=1.0, top_p=0.95, top_k=20,
        do_sample=True, pad_token_id=tokenizer.eos_token_id,
    )
    turn2_response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)

    match = BRRM_RANKING_PATTERN.search(turn2_response)
    if match:
        val = match.group(1).strip()
        if val in ("1", "2"):
            return int(val)
    return 0  # parse failure


def evaluate_preference_generative(model, tokenizer, records, device, max_gen_tokens=8192, seed=42):
    """Evaluate preference accuracy using a generative judge model (e.g. BRRM)."""
    print(f"\n{'=' * 70}")
    print(f"  GENERATIVE PREFERENCE EVALUATION — {len(records)} test pairs")
    print(f"{'=' * 70}")

    rng = random.Random(seed)
    correct = 0
    total = 0
    parse_failures = 0
    skipped = 0
    domain_stats: dict[str, list[int]] = {}
    difficulty_stats: dict[str, list[int]] = {}

    for record in tqdm(records, desc="Gen-Preference"):
        messages = record.get("messages", [])
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not messages or not chosen or not rejected:
            skipped += 1
            continue

        context_str = _format_context(messages)
        chosen_text = _extract_response_text(chosen)
        rejected_text = _extract_response_text(rejected)

        # Randomize order to avoid position bias
        chosen_is_first = rng.random() < 0.5
        if chosen_is_first:
            r1, r2 = chosen_text, rejected_text
        else:
            r1, r2 = rejected_text, chosen_text

        try:
            preference = _brrm_judge(model, tokenizer, context_str, r1, r2, device, max_gen_tokens)
        except Exception as e:
            if skipped < 3:
                print(f"  [SKIP] {type(e).__name__}: {e}")
            skipped += 1
            continue

        if preference == 0:
            parse_failures += 1
            continue

        # Map back: if chosen was Response 1, correct answer is 1
        correct_answer = 1 if chosen_is_first else 2
        is_correct = preference == correct_answer
        correct += int(is_correct)
        total += 1

        metadata = record.get("metadata", {})
        domain = metadata.get("domain", "unknown")
        difficulty = metadata.get("difficulty", "unknown")

        for bucket, key in [(domain_stats, domain), (difficulty_stats, difficulty)]:
            if key not in bucket:
                bucket[key] = [0, 0]
            bucket[key][0] += int(is_correct)
            bucket[key][1] += 1

    if skipped:
        print(f"  Skipped: {skipped}")
    if parse_failures:
        print(f"  Parse failures: {parse_failures}")

    if total == 0:
        print("  No valid pairs evaluated.")
        return {}

    print(f"\n  Overall accuracy: {correct}/{total}  ({100 * correct / total:.2f}%)")
    print(f"  Parse failure rate: {parse_failures}/{total + parse_failures}  ({100 * parse_failures / (total + parse_failures):.1f}%)")

    results_domain = {}
    if domain_stats:
        print(f"\n  {'Domain':<25} {'Accuracy':>10} {'Correct':>9} {'Total':>7}")
        print(f"  {'-' * 55}")
        for d in sorted(domain_stats):
            c, t = domain_stats[d]
            results_domain[d] = {"accuracy": round(100 * c / t, 4), "correct": c, "total": t}
            print(f"  {d:<25} {100 * c / t:>9.2f}% {c:>9} {t:>7}")

    results_difficulty = {}
    if difficulty_stats:
        print(f"\n  {'Difficulty':<25} {'Accuracy':>10} {'Correct':>9} {'Total':>7}")
        print(f"  {'-' * 55}")
        for d in sorted(difficulty_stats):
            c, t = difficulty_stats[d]
            results_difficulty[d] = {"accuracy": round(100 * c / t, 4), "correct": c, "total": t}
            print(f"  {d:<25} {100 * c / t:>9.2f}% {c:>9} {t:>7}")

    return {
        "total": total,
        "correct": correct,
        "parse_failures": parse_failures,
        "skipped": skipped,
        "accuracy": round(100 * correct / total, 4),
        "domains": results_domain,
        "difficulty": results_difficulty,
    }


def evaluate_cultural_baseline(model, tokenizer, data_dir, device, max_length, pad_token_id):
    """Score cultural conversations with the base model's native reward and report per-country / per-arousal stats."""
    records = load_cultural_test(data_dir)
    if not records:
        print(f"No cultural test records found in {data_dir}")
        return {}

    print(f"\n{'=' * 70}")
    print(f"  BASELINE CULTURAL EVALUATION — {len(records)} conversations")
    print(f"{'=' * 70}")

    country_scores: dict[str, list[float]] = {}
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
            score = _get_reward_score(model, tokenizer, messages, device, max_length, pad_token_id)
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
    print(f"\n  {'Country':<12} {'N':>4} {'Score':>8} {'Std':>8}")
    print(f"  {'-' * 36}")

    for c in sorted(country_scores):
        cs = np.array(country_scores[c])
        row = {"n": len(cs), "score_mean": round(float(cs.mean()), 4), "score_std": round(float(cs.std()), 4)}
        print(f"  {c:<12} {len(cs):>4} {cs.mean():>8.4f} {cs.std():>8.4f}")
        results_country[c] = row

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
    """Generate per-model plots from baseline evaluation results."""
    model_name = os.path.basename(results.get("model", "").rstrip("/"))

    # Scoring: Spearman per attribute
    scoring = results.get("scoring", {})
    attrs_data = scoring.get("attributes", {})
    if attrs_data:
        attrs = [a for a in ATTRIBUTES if a in attrs_data]
        vals = [attrs_data[a]["spearman"] for a in attrs]
        if attrs:
            fig, ax = plt.subplots(figsize=(8, max(5, len(attrs) * 0.35)))
            colors = ["#4CAF50" if v >= 0.5 else "#FF9800" if v >= 0.3 else "#F44336" for v in vals]
            ax.barh(attrs, vals, color=colors)
            ax.set_xlabel("Spearman Correlation")
            ax.set_title(f"{model_name} (baseline) — Spearman by Attribute")
            ax.invert_yaxis()
            ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
            ax.legend(fontsize=8)
            _save_fig(fig, os.path.join(plots_dir, "spearman_by_attribute_baseline.png"))

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
        ax.set_title(f"{model_name} (baseline) — Preference Accuracy by Domain")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=30)
        _save_fig(fig, os.path.join(plots_dir, "preference_by_domain_baseline.png"))


def _save_results(results, args):
    """Save results JSON to model_parent_dir/<model_name>/results/eval_baseline.json."""
    if not args.model_name:
        print("\n  WARNING: --model_name not set, skipping auto-save. Use --model_name to save results.")
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n  Results saved to {args.output_json}")
        return

    output_paths = []
    if args.output_json:
        output_paths.append(args.output_json)
    auto_json = os.path.join(args.model_parent_dir, args.model_name, "results", "eval_baseline.json")
    if auto_json not in output_paths:
        output_paths.append(auto_json)
    for out_path in output_paths:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {out_path}")

    # Generate per-model plots
    plots_dir = os.path.join(args.model_parent_dir, args.model_name, "results", "plots")
    _generate_plots(results, plots_dir)


def main() -> None:
    print(f"\n### Evaluate baseline started at {datetime.now().isoformat()} ###")
    parser = ArgumentParser(description="Evaluate base reward model on scoring and preference data.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Base model HF ID or local path.")
    parser.add_argument("--scoring_data_path", type=str, default="data/dataset/Multi-Domain-Data-Scoring", help="Path to scoring dataset JSONL.")
    parser.add_argument("--preference_data_path", type=str, default="data/dataset/Multi-Domain-Data-Preference-Pairs", help="Path to preference pairs dataset JSONL.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_scoring", action="store_true", help="Skip scoring evaluation.")
    parser.add_argument("--skip_preference", action="store_true", help="Skip preference evaluation.")
    parser.add_argument("--eval", type=str, default=None, help="Path to cultural test JSON directory (e.g. data/test). Enables cultural evaluation.")
    parser.add_argument("--generative_judge", action="store_true", help="Use generative judge (e.g. BRRM) for preference evaluation. Loads model as CausalLM.")
    parser.add_argument("--max_gen_tokens", type=int, default=8192, help="Max new tokens per generation turn (generative judge mode).")
    parser.add_argument("--model_name", type=str, default=None, help="Packaged model dir name (e.g. multi-domain-rm-llama-3-8b-it). Results saved to model/<model_name>/results/eval_baseline.json.")
    parser.add_argument("--model_parent_dir", type=str, default="model", help="Parent directory for saving results.")
    parser.add_argument("--output_json", type=str, default=None, help="Custom path to save results JSON.")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    apply_section_overrides(args, config.get("evaluate_baseline", {}))

    if not args.model_path:
        parser.error("--model_path is required (via CLI or config.yaml evaluate_baseline.model_path)")

    generative = args.generative_judge

    print(f"Base model:          {args.model_path}")
    if generative:
        print(f"Mode:                generative judge (preference only)")
    else:
        print(f"Mode:                native reward score")
    print(f"Scoring data:        {args.scoring_data_path}")
    print(f"Preference data:     {args.preference_data_path}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    trust_remote = _requires_remote_code(args.model_path)

    # --- Generative mode (BRRM): preference only ---
    if generative:
        print(f"\nLoading generative model on {device} ({dtype})...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={"": 0} if use_cuda else None,
            dtype=dtype,
            trust_remote_code=trust_remote,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=trust_remote)
        model.eval()

        results = {"model": args.model_path, "type": "baseline_generative"}
        pref_records = load_jsonl_test(args.preference_data_path)
        if not pref_records:
            print("No preference test records found.")
        else:
            if args.max_samples and args.max_samples < len(pref_records):
                random.seed(42)
                random.shuffle(pref_records)
                pref_records = pref_records[:args.max_samples]
            results["preference"] = evaluate_preference_generative(model, tokenizer, pref_records, device, args.max_gen_tokens)

        _save_results(results, args)
        return

    # --- Scalar RM mode ---
    print(f"\nLoading base model on {device} ({dtype})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        device_map={"": 0} if use_cuda else None,
        dtype=dtype,
        trust_remote_code=trust_remote,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=trust_remote)
    model.eval()
    pad_token_id = tokenizer.pad_token_id

    results = {"model": args.model_path, "type": "baseline"}

    # Scoring evaluation
    if not args.skip_scoring:
        scoring_records = load_jsonl_test(args.scoring_data_path)
        if not scoring_records:
            print("No scoring test records found.")
        else:
            if args.max_samples and args.max_samples < len(scoring_records):
                random.seed(42)
                random.shuffle(scoring_records)
                scoring_records = scoring_records[:args.max_samples]
            results["scoring"] = evaluate_scoring(model, tokenizer, scoring_records, device, args.max_length, pad_token_id)

    if not args.skip_preference:
        pref_records = load_jsonl_test(args.preference_data_path)
        if not pref_records:
            print("No preference test records found.")
        else:
            if args.max_samples and args.max_samples < len(pref_records):
                random.seed(42)
                random.shuffle(pref_records)
                pref_records = pref_records[:args.max_samples]
            results["preference"] = evaluate_preference(model, tokenizer, pref_records, device, args.max_length, pad_token_id)

    if args.eval:
        results["cultural"] = evaluate_cultural_baseline(
            model, tokenizer, args.eval, device, args.max_length, pad_token_id,
        )

    _save_results(results, args)


if __name__ == "__main__":
    main()
