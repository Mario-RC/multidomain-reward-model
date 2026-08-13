# stage-2_train.py

import os
import re
import sys
import time
import tempfile
import torch
import numpy as np
from safetensors.torch import load_file
from argparse import ArgumentParser, BooleanOptionalAction
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import pandas as pd
from glob import glob
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
import datasets
import traceback  # Used for detailed error traces
from config_utils import load_yaml_config, apply_model_registry, apply_section_overrides

from datetime import datetime
from utils import shared_gate_checkpoint_filename

# Enable TF32 for better throughput on Ampere+ GPUs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"\n### Stage 2: Train started at {datetime.now().isoformat()} ###")

from attributes import (
    ATTRIBUTES as attributes,
    ATTRIBUTE_SUBSETS,
    resolve_active_attributes,
    DOMAIN_ATTRIBUTE_INDICES,
    DOMAIN_NAMES,
)

# ----------------------------
# MODEL
# ----------------------------
class GatingNetwork(nn.Module):
    """
    Lightweight MLP that outputs objective-mixing weights.

    The network consumes prompt embeddings and predicts one weight per reward
    objective. Outputs are temperature-scaled and normalized with softmax.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 10,
        logit_scale: float = 1.0,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
        dropout: float = 0.0,
        learnable_logit_scale: bool = False,
        active_attribute_indices=None,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature
        self.logit_scale = nn.Parameter(
            torch.ones(1) * logit_scale,
            requires_grad=learnable_logit_scale,
        )
        self.dropout_prob = dropout
        active_mask = torch.ones(out_features, dtype=torch.bool)
        if active_attribute_indices is not None:
            if not active_attribute_indices:
                raise ValueError("At least one attribute must be active.")
            if min(active_attribute_indices) < 0 or max(active_attribute_indices) >= out_features:
                raise ValueError("active_attribute_indices contains an out-of-range index.")
            active_mask.zero_()
            active_mask[list(active_attribute_indices)] = True
        # Reconstructed from training_config; keep legacy state_dicts compatible.
        self.register_buffer("active_attribute_mask", active_mask, persistent=False)
        layers = []
        last_dim = in_features
        for _ in range(n_hidden):
            layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through the gating network."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Hidden layers: ReLU + optional dropout.
                x = F.relu(x)
                if self.dropout_prob > 0 and self.training:  # Dropout only in training mode.
                    x = F.dropout(x, p=self.dropout_prob)
        # Normalize objective weights with temperature-scaled softmax.
        logits = x / self.temperature
        mask = self.active_attribute_mask.to(device=logits.device)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        x = F.softmax(logits, dim=-1)
        return x * self.logit_scale  # Learnable global output scaling.

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------
def find_debiasing_penalties(cluster_V, debiasing_dim=4, corr_threshold=0.028):
    """
    Find per-dimension penalties that decorrelate all other reward dimensions
    from a chosen target dimension.

    For each dimension d != debiasing_dim, searches signed penalty
    candidates until the absolute Spearman correlation between the adjusted d
    and the raw debiasing_dim falls below `corr_threshold`.  The adjusted
    value is: V_d' = V_d - penalty * V_debiasing_dim.

    Args:
        cluster_V (np.ndarray): Array of shape [N, K] containing multi-objective rewards.
        debiasing_dim (int): Index of the dimension to decorrelate from.
        corr_threshold (float): Maximum allowed absolute Spearman correlation.
    Returns:
        dict: Contains penalty and final-correlation arrays plus any unresolved dimensions.
    """
    magnitudes = [
        0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25,
        0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0,
    ]
    penalty_candidates = sorted({value for magnitude in magnitudes for value in (magnitude, -magnitude)})
    K = cluster_V.shape[1]
    candidate_dims = set(range(K))

    if debiasing_dim not in candidate_dims:
        print(f"Warning: debiasing_dim {debiasing_dim} is out of bounds (0-{K-1}). Skipping debiasing.")
        return {"penalty": np.ones(K), "corr": np.ones(K), "unresolved_dims": [debiasing_dim]}
    candidate_dims.remove(debiasing_dim)

    dimwise_penalties = np.zeros(K)  # Initialize with no penalty.
    # Track final (or best observed) correlation for each dimension.
    dimwise_corr_final = {dim: 1.0 for dim in range(K)}

    for penalty in penalty_candidates:
        if not candidate_dims:
            break  # Stop once all dimensions satisfy the threshold.
        # Apply candidate penalty to the target dimension's contribution.
        V_adjusted = cluster_V - penalty * cluster_V[:, [debiasing_dim]]
        dims_to_remove = set()
        for dim in candidate_dims:
            # Compute Spearman correlation; guard against degenerate inputs.
            adjusted_values = V_adjusted[:, dim]
            target_values = cluster_V[:, debiasing_dim]
            if np.std(adjusted_values) == 0 or np.std(target_values) == 0:
                corr = 0.0
            else:
                try:
                    corr, _ = spearmanr(adjusted_values, target_values)
                    if np.isnan(corr):
                        corr = 0.0
                except ValueError:
                    corr = 0.0

            if abs(corr) <= corr_threshold:
                dims_to_remove.add(dim)
                dimwise_penalties[dim] = penalty  # First penalty that satisfies threshold.
                dimwise_corr_final[dim] = corr  # Correlation at acceptance time.
            else:
                # Keep the best (smallest absolute) correlation seen so far.
                dimwise_corr_final[dim] = min(dimwise_corr_final.get(dim, 1.0), abs(corr))

        candidate_dims -= dims_to_remove  # Skip already-satisfied dimensions.

    # Build final per-dimension correlation summary.
    final_corr_array = np.array([dimwise_corr_final.get(dim, 1.0) for dim in range(K)])
    final_corr_array[debiasing_dim] = 1.0  # Self-correlation.

    return {"penalty": dimwise_penalties, "corr": final_corr_array, "unresolved_dims": sorted(candidate_dims)}


def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """Calculate weighted average scores for each section of the RewardBench."""
    section_scores = {}
    for section, tests in subset_mapping.items():
        valid_tests = [test for test in tests if test in metrics and test in example_counts]
        total_weighted_score = sum(metrics[test] * example_counts[test] for test in valid_tests)
        total_examples = sum(example_counts[test] for test in valid_tests)
        section_scores[section] = 100 * total_weighted_score / total_examples if total_examples > 0 else 0.0
    return section_scores

def eval_reward_bench(df_examples, acc_column="correct"):
    """
    Evaluate RewardBench using precomputed correctness flags.

    Returns section scores plus per-subset metrics.
    """
    categories = {
        "chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
        "chat-hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual"],
        "safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond", "donotanswer"],
        "reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
    }

    all_rows = []
    for category, subsets in categories.items():
        for subset in subsets:
            df_subset = df_examples[df_examples["subset"] == subset]
            # Use `nanmean` to be robust when labels contain NaNs.
            acc = np.nanmean(df_subset[acc_column].values) if len(df_subset) > 0 else 0.0
            row = {"category": category, "subset": subset, "n": len(df_subset), "accuracy": acc}
            all_rows.append(row)

    df_acc = pd.DataFrame.from_records(all_rows) if all_rows else pd.DataFrame(columns=["category", "subset", "n", "accuracy"])

    EXAMPLE_COUNTS = { "alpacaeval-easy": 100, "alpacaeval-length": 95, "alpacaeval-hard": 95, "mt-bench-easy": 28, "mt-bench-med": 40, "mt-bench-hard": 37, "math-prm": 984, "refusals-dangerous": 100, "refusals-offensive": 100, "llmbar-natural": 100, "llmbar-adver-neighbor": 134, "llmbar-adver-GPTInst": 92, "llmbar-adver-GPTOut": 47, "llmbar-adver-manual": 46, "xstest-should-refuse": 250, "xstest-should-respond": 154, "donotanswer": 136, "hep-cpp": 164, "hep-go": 164, "hep-java": 164, "hep-js": 164, "hep-python": 164, "hep-rust": 164 }
    SUBSET_MAPPING = { "Chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"], "Chat Hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual"], "Safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond", "donotanswer"], "Reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"] }

    # Build a subset -> accuracy dictionary.
    metrics = {row['subset']: row['accuracy'] for _, row in df_acc.iterrows() if pd.notna(row['accuracy'])}

    scores_per_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, metrics)
    score_weights = {"Chat": 1, "Chat Hard": 1, "Safety": 1, "Reasoning": 1}

    # Compute final weighted score with divide-by-zero protection.
    total_score = sum(scores_per_section.get(k, 0) * score_weights.get(k, 0) for k in score_weights)
    total_weight = sum(score_weights.get(k, 0) for k in score_weights if k in scores_per_section and scores_per_section.get(k) is not None)
    scores_per_section["Score"] = round(total_score / total_weight, 2) if total_weight > 0 else 0.0

    return scores_per_section, metrics


def load_embeddings(embedding_path_pattern, require_routing_metadata=False):
    """
    Load embedding pairs from `.safetensors` files.

    Returns concatenated tensors on CPU with basic integrity checks.
    """
    file_paths = sorted(glob(embedding_path_pattern))
    if not file_paths:
        raise ValueError(f"No embedding files found matching pattern: {embedding_path_pattern}")

    embeddings_list, prompt_embeddings_list = [], []
    difficulties_list, domains_list, group_ids_list = [], [], []
    format_versions = []
    print(f"Loading {len(file_paths)} embedding file(s) matching pattern: ...{os.path.basename(embedding_path_pattern)}")  # Keep log compact.

    for embedding_path in file_paths:
        try:
            embeddings_data = load_file(embedding_path)  # Loaded on CPU by default.
            # Validate required keys.
            if "embeddings" not in embeddings_data or "prompt_embeddings" not in embeddings_data:
                print(f"Warning: Skipping file {embedding_path} due to missing keys 'embeddings' or 'prompt_embeddings'.")
                continue
            embeddings_list.append(embeddings_data["embeddings"])
            prompt_embeddings_list.append(embeddings_data["prompt_embeddings"])
            if "difficulties" in embeddings_data:
                difficulties_list.append(embeddings_data["difficulties"])
            if "domains" in embeddings_data:
                domains_list.append(embeddings_data["domains"])
            if "group_ids" in embeddings_data:
                group_ids_list.append(embeddings_data["group_ids"])
            if "format_version" in embeddings_data:
                format_versions.append(int(embeddings_data["format_version"].reshape(-1)[0].item()))
        except Exception as e:
            print(f"Warning: Failed to load or process file {embedding_path}: {e}")
            continue  # Skip corrupted or unreadable files.

    if not embeddings_list or not prompt_embeddings_list:
         raise ValueError(f"No valid embeddings could be loaded from {embedding_path_pattern}. Check file integrity and ensure keys ('embeddings', 'prompt_embeddings') exist.")

    # Concatenate on CPU and cast to float32 for consistency.
    embeddings_cpu = torch.cat(embeddings_list, dim=0).float()
    prompt_embeddings_cpu = torch.cat(prompt_embeddings_list, dim=0).float()
    difficulties_cpu = torch.cat(difficulties_list, dim=0) if len(difficulties_list) == len(embeddings_list) else None
    domains_cpu = torch.cat(domains_list, dim=0).long() if len(domains_list) == len(embeddings_list) else None
    group_ids_cpu = torch.cat(group_ids_list, dim=0).long() if len(group_ids_list) == len(embeddings_list) else None

    row_count = len(embeddings_cpu)
    for name, tensor in (
        ("prompt_embeddings", prompt_embeddings_cpu),
        ("difficulties", difficulties_cpu),
        ("domains", domains_cpu),
        ("group_ids", group_ids_cpu),
    ):
        if tensor is not None and len(tensor) != row_count:
            raise ValueError(f"{name} has {len(tensor)} rows; expected {row_count}.")
    if require_routing_metadata:
        if len(format_versions) != len(embeddings_list) or any(version != 2 for version in format_versions):
            raise ValueError(
                "Stage-2 embeddings must explicitly declare format_version=2 in every shard."
            )
        if domains_cpu is not None and (
            (domains_cpu < 0).any() or (domains_cpu >= len(DOMAIN_NAMES)).any()
        ):
            raise ValueError("Domain labels contain unknown/out-of-range values.")
        problems = []
        if prompt_embeddings_cpu.ndim != 2:
            problems.append(
                f"prompt_embeddings must be [N,H], got {tuple(prompt_embeddings_cpu.shape)}"
            )
        if domains_cpu is None:
            problems.append("missing 'domains'")
        if group_ids_cpu is None:
            problems.append("missing 'group_ids'")
        if problems:
            raise ValueError(
                "Legacy or invalid Stage-2 embeddings (" + ", ".join(problems) + "). "
                "Re-run stage-2_prepare.py with the shared-prompt format."
            )
    elif prompt_embeddings_cpu.ndim != 2:
        print(
            "WARNING: legacy candidate-conditioned prompt embeddings detected; "
            "they cannot be used to train or evaluate shared-prompt gating."
        )
    print(f"Successfully loaded a total of {len(embeddings_cpu)} embedding pairs into CPU RAM.")
    return embeddings_cpu, prompt_embeddings_cpu, difficulties_cpu, domains_cpu, group_ids_cpu


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    """Main function to parse arguments, load data, train the model, and evaluate."""
    training_started = time.perf_counter()
    parser = ArgumentParser(description="Train ArmoRM Gating Network")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--base_data_dir", type=str, default=None, help="Override model artifact root (useful for isolated tests).")
    parser.add_argument("--model_key", type=str, default=None, help="Model key defined in config.yaml:model_registry.")
    parser.add_argument("--model_path", type=str, default=None, help="Path or HF ID of the base Reward Model")
    parser.add_argument("--multi_objective_dataset_name", type=str, default=None, help="Dataset name from stage-1_prepare output (e.g., 'stage_1').")
    parser.add_argument("--preference_dataset_name", type=str, default=None, help="Preference dataset folder name (matches stage-2_prepare output_dataset_name). Required.")
    parser.add_argument("--validation_preference_dataset_name", type=str, default=None, help="Optional full V2 dataset that defines a fixed grouped validation set while training on a selected subset.")
    parser.add_argument("--reference_dataset_name", type=str, default=None, help="Reference dataset used only for debiasing. Set to null to disable debiasing.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Split suffix used by stage-2_prepare outputs (e.g., train, all, val, test)."    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for AdamW optimizer")
    parser.add_argument("--n_steps", type=int, default=30000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--debiasing_dims", type=int, nargs="+", default=[-1], help="Indices (0-based) of attribute dimensions to decorrelate from all others. Set to -1 to disable debiasing. Multiple values supported, e.g. --debiasing_dims 21 18.")
    parser.add_argument("--corr_threshold", type=float, default=0.04, help="Maximum allowed absolute Spearman correlation for debiasing")
    parser.add_argument("--model_family", type=str, default="llama3", choices=["llama3", "gemma2", "qwen3", "mistral", "auto"], help="Model family for token pattern matching during embedding extraction (if applicable, less relevant here)")
    parser.add_argument("--eval", type=str, default=None, help="Eval dataset name (e.g. reward-bench). Requires embeddings from stage-2_prepare.")
    parser.add_argument("--eval_split", type=str, default="filtered", help="Split suffix for the eval dataset (default: filtered).")
    parser.add_argument("--logit_scale", type=float, default=2.0, help="Scaling factor applied after softmax in the gating network")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for softmax scaling in the gating network")
    parser.add_argument("--n_hidden", type=int, default=1, help="Number of hidden layers in the gating network MLP")
    parser.add_argument("--hidden_size", type=int, default=64, help="Dimension of hidden layers in the gating network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability in the gating network's hidden layers")
    parser.add_argument("--learnable_logit_scale", action=BooleanOptionalAction, default=False, help="Allow the global gate scale to train (off by default).")
    parser.add_argument("--domain_loss_weight", type=float, default=0.25, help="Weight of supervised domain-mass routing loss.")
    parser.add_argument("--entropy_weight", type=float, default=0.02, help="Weight of per-example anti-collapse entropy loss.")
    parser.add_argument("--entropy_floor_fraction", type=float, default=0.35, help="Minimum gate entropy as a fraction of log(active attributes).")
    parser.add_argument("--load_balance_weight", type=float, default=0.05, help="Weight of batch-level attribute balancing loss.")
    parser.add_argument("--balance_domains", action=BooleanOptionalAction, default=True, help="Sample batches uniformly across domains.")
    parser.add_argument("--balance_difficulties", action=BooleanOptionalAction, default=False, help="Balance non-empty domain-by-difficulty cells instead of domains only.")
    parser.add_argument("--attribute_subset", choices=sorted(ATTRIBUTE_SUBSETS), default="full", help="Reversible Stage-2 attribute ablation; Stage 1 remains 23-dimensional.")
    parser.add_argument("--exclude_attributes", nargs="*", default=[], help="Additional exact attribute names to mask before gate softmax.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction of prompt groups used for validation.")
    parser.add_argument("--train_on_all", action=BooleanOptionalAction, default=False, help="Refit selected hyperparameters on all available training groups for a fixed number of steps.")
    parser.add_argument("--max_samples", type=int, default=None, help="Load only the first N samples from datasets (for debugging RAM issues)")
    parser.add_argument("--eval_every", type=int, default=200, help="Evaluate on validation set every N steps")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (number of evaluations without improvement)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_tag", type=str, default=None, help="Optional safe tag appended to checkpoint names (letters, digits, underscore, hyphen).")
    parser.add_argument("--stage_1_weights_path", type=str, default=None, help="Optional override for Stage 1 regression weights path (default: auto-resolved _100pct.pt)")
    parser.add_argument("--curriculum", action="store_true", default=False, help="Enable phased curriculum learning: easy → easy+medium → all")
    parser.add_argument("--curriculum_phase1_frac", type=float, default=0.20, help="Fraction of n_steps for easy-only phase (default: 0.20)")
    parser.add_argument("--curriculum_phase2_frac", type=float, default=0.50, help="Fraction of n_steps to end easy+medium phase (default: 0.50)")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    args = apply_section_overrides(args, config.get("stage_2_train", {}))
    try:
        args = apply_model_registry(args, config)
    except ValueError as error:
        parser.error(str(error))
    if not args.model_path:
        parser.error("--model_path is required via CLI, stage_2_train, or --model_key.")
    if not args.multi_objective_dataset_name:
        parser.error("--multi_objective_dataset_name is required.")
    if args.n_steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.patience < 1:
        parser.error("--n_steps, --batch_size, --eval_every, and --patience must be >= 1.")
    if not 0 < args.val_size < 1:
        parser.error("--val_size must be strictly between 0 and 1.")
    if args.temperature <= 0 or args.logit_scale <= 0:
        parser.error("--temperature and --logit_scale must be positive.")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1).")
    if args.corr_threshold < 0 or args.corr_threshold > 1:
        parser.error("--corr_threshold must be in [0, 1].")
    if not 0 <= args.curriculum_phase1_frac <= args.curriculum_phase2_frac <= 1:
        parser.error("Curriculum fractions must satisfy 0 <= phase1 <= phase2 <= 1.")
    try:
        active_attribute_indices, active_attribute_names, excluded_attribute_names = (
            resolve_active_attributes(args.attribute_subset, args.exclude_attributes)
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"Attribute subset: {args.attribute_subset} ({len(active_attribute_names)}/{len(attributes)} active)")
    print(f"Excluded attributes: {list(excluded_attribute_names) or 'none'}")
    invalid_debiasing_dims = sorted({
        dimension for dimension in args.debiasing_dims
        if dimension < -1 or dimension >= len(attributes)
    })
    if invalid_debiasing_dims:
        parser.error(
            f"--debiasing_dims contains out-of-range values: {invalid_debiasing_dims}; "
            f"valid values are -1 or 0..{len(attributes) - 1}."
        )
    if -1 in args.debiasing_dims and any(dimension >= 0 for dimension in args.debiasing_dims):
        parser.error("--debiasing_dims -1 cannot be combined with active dimensions.")
    if not 0 <= args.entropy_floor_fraction <= 1:
        parser.error("--entropy_floor_fraction must be in [0, 1].")
    if args.checkpoint_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.checkpoint_tag):
        parser.error("--checkpoint_tag accepts only letters, digits, underscore, and hyphen.")

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    # Seed RNGs for reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Resolve local base paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DATA_DIR = args.base_data_dir or os.path.join(script_dir, "model")
    # ----------------------------------

    # Validate preference_dataset_name (required).
    if not args.preference_dataset_name:
        print("FATAL ERROR: --preference_dataset_name is required (set stage_2_train.preference_dataset_name in config.yaml or pass --preference_dataset_name).")
        sys.exit(1)

    # A null reference truly disables debiasing. It must not silently fall back
    # to preference data because that invalidates the no-debiasing ablation.
    _ref_is_null = args.reference_dataset_name is None or str(args.reference_dataset_name).lower() == "null"
    if _ref_is_null and any(d >= 0 for d in args.debiasing_dims):
        print("NOTE: reference_dataset_name=null; disabling requested debiasing dimensions.")
        args.debiasing_dims = [-1]

    # Extract short names used in filesystem paths.
    args.model_name = args.model_path.split("/")[-1]
    # Match stage-2_prepare output naming convention: <dataset>-<dataset_split>.
    pref_base = args.preference_dataset_name
    validation_pref_base = args.validation_preference_dataset_name
    ref_base = "null" if _ref_is_null else args.reference_dataset_name
    args.preference_dataset_name = f"{pref_base}-{args.dataset_split}"
    args.validation_preference_dataset_name = (
        f"{validation_pref_base}-{args.dataset_split}" if validation_pref_base else None
    )
    args.reference_dataset_name = None if _ref_is_null else f"{ref_base}-{args.dataset_split}"

    # --- Define load paths ---
    # Preference embeddings path pattern (inside dataset-split folder).
    preference_embedding_path_pattern = os.path.join(
        BASE_DATA_DIR, "embeddings", args.model_name, args.preference_dataset_name, "*.safetensors"
    )
    validation_embedding_path_pattern = None
    if validation_pref_base and validation_pref_base != pref_base:
        validation_embedding_path_pattern = os.path.join(
            BASE_DATA_DIR, "embeddings", args.model_name,
            args.validation_preference_dataset_name, "*.safetensors",
        )
    # Regression weights file path.
    if args.stage_1_weights_path:
        fname = args.stage_1_weights_path
        if os.path.sep not in fname:
            # Auto-append _100pct suffix when bare filename lacks it.
            if fname.endswith(".pt") and not (fname.endswith("_100pct.pt") or fname.endswith("_80pct.pt")):
                fname = fname[:-3] + "_100pct.pt"
            regression_layer_path = os.path.join(BASE_DATA_DIR, "regression_weights", fname)
        else:
            regression_layer_path = fname
    else:
        regression_layer_path = os.path.join(
            BASE_DATA_DIR, "regression_weights", f"{args.model_name}_{args.multi_objective_dataset_name}_100pct.pt"
        )
    # Eval dataset embeddings path pattern.
    eval_embedding_path_pattern = None
    if args.eval:
        eval_folder_name = f"{args.eval}-{args.eval_split}"
        eval_embedding_path_pattern = os.path.join(
            BASE_DATA_DIR, "embeddings", args.model_name, eval_folder_name, "*.safetensors"
        )
    reference_embedding_path_pattern = None
    if not _ref_is_null:
        reference_embedding_path_pattern = os.path.join(
            BASE_DATA_DIR, "embeddings", args.model_name, args.reference_dataset_name, "*.safetensors"
        )
    # -------------------------

    # Print paths.
    print(f"Preference Embedding Path Pattern: {preference_embedding_path_pattern}")
    print(f"Fixed Validation Embedding Path Pattern: {validation_embedding_path_pattern or 'same as preference data'}")
    print(f"Regression Layer Path: {regression_layer_path}")
    print(f"Reference Embedding Path Pattern: {reference_embedding_path_pattern or 'disabled'}")
    if eval_embedding_path_pattern:
        print(f"Eval Embedding Path Pattern: {eval_embedding_path_pattern}")

    # Load data to CPU with robust error handling.
    try:
        print("Loading preference embeddings (to CPU RAM)...")
        embeddings_cpu, prompt_embeddings_cpu, difficulties_cpu, domains_cpu, group_ids_cpu = load_embeddings(
            preference_embedding_path_pattern, require_routing_metadata=True
        )

        if args.max_samples is not None and args.max_samples < len(embeddings_cpu):
            print(f"NOTE: Subsetting preference data to first {args.max_samples} samples.")
            indices = torch.arange(args.max_samples)
            embeddings_cpu = embeddings_cpu[indices]
            prompt_embeddings_cpu = prompt_embeddings_cpu[indices]
            if difficulties_cpu is not None:
                difficulties_cpu = difficulties_cpu[indices]
            domains_cpu = domains_cpu[indices]
            group_ids_cpu = group_ids_cpu[indices]

        validation_embeddings_cpu = validation_prompt_embeddings_cpu = None
        validation_difficulties_cpu = validation_domains_cpu = validation_group_ids_cpu = None
        if validation_embedding_path_pattern:
            print("Loading fixed full validation source (to CPU RAM)...")
            (
                validation_embeddings_cpu, validation_prompt_embeddings_cpu,
                validation_difficulties_cpu, validation_domains_cpu, validation_group_ids_cpu,
            ) = load_embeddings(validation_embedding_path_pattern, require_routing_metadata=True)
            if args.max_samples is not None and args.max_samples < len(validation_embeddings_cpu):
                validation_indices = torch.arange(args.max_samples)
                validation_embeddings_cpu = validation_embeddings_cpu[validation_indices]
                validation_prompt_embeddings_cpu = validation_prompt_embeddings_cpu[validation_indices]
                if validation_difficulties_cpu is not None:
                    validation_difficulties_cpu = validation_difficulties_cpu[validation_indices]
                validation_domains_cpu = validation_domains_cpu[validation_indices]
                validation_group_ids_cpu = validation_group_ids_cpu[validation_indices]

        print("Loading regression layer (to device)...")
        regression_layer = torch.load(regression_layer_path, map_location=device, weights_only=True)["weight"].float()
        n_attributes, hidden_size = regression_layer.shape
        if n_attributes != len(attributes):
            raise ValueError(
                f"Stage-1 head has {n_attributes} outputs, but the canonical "
                f"taxonomy has {len(attributes)}."
            )

        ref_embeddings_cpu = None
        _debiasing_requested = any(d >= 0 for d in args.debiasing_dims)
        if _debiasing_requested:
            print("Loading reference embeddings for debiasing (to CPU RAM)...")
            ref_embeddings_cpu, _, _, _, _ = load_embeddings(reference_embedding_path_pattern)

            if args.max_samples is not None and args.max_samples < len(ref_embeddings_cpu):
                print(f"NOTE: Subsetting reference data to first {args.max_samples} samples.")
                indices_ref = torch.arange(args.max_samples)
                ref_embeddings_cpu = ref_embeddings_cpu[indices_ref]
        else:
            if not _ref_is_null:
                print(f"WARNING: reference_dataset_name '{ref_base}' was provided but debiasing is disabled, "
                      f"so the reference dataset will NOT be used. "
                      f"Set --debiasing_dims to enable debiasing, "
                      f"or pass --reference_dataset_name null to silence this warning.")
            print("Debiasing disabled. Skipping reference embeddings.")

    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"FATAL ERROR: Failed during data loading: {e}.")
        print("Please ensure:")
        print("1. All necessary `stage-1` and `stage-2_prepare` scripts ran successfully.")
        print("2. The file paths printed above point to existing files/folders.")
        print(f"3. Regression weights file ({regression_layer_path}) contains the 'weight' key.")
        sys.exit(1)

    # Calculate debiasing penalties for each requested dimension.
    debiasing_dims = [d for d in args.debiasing_dims if 0 <= d < n_attributes]
    debiasing_enabled = len(debiasing_dims) > 0

    all_penalties = {}  # dim -> penalties_tensor
    if debiasing_enabled:
        print(f"Calculating debiasing penalties for dimensions: {debiasing_dims}...")
        ref_embeddings_for_debiasing = ref_embeddings_cpu.to('cpu')
        if ref_embeddings_for_debiasing is not None and ref_embeddings_for_debiasing.shape[0] > 0:
            regression_layer_cpu = regression_layer.to('cpu')
            try:
                pairwise_rewards = ref_embeddings_for_debiasing @ regression_layer_cpu.T
                rewards = pairwise_rewards.reshape(-1, n_attributes) if pairwise_rewards.numel() > 0 else np.array([])

                if rewards.shape[0] > 0:
                    for dim in debiasing_dims:
                        penalties = find_debiasing_penalties(
                            rewards.numpy(), debiasing_dim=dim, corr_threshold=args.corr_threshold
                        )
                        unresolved_dims = penalties["unresolved_dims"]
                        if unresolved_dims:
                            raise RuntimeError(
                                f"Debiasing dimension {dim} did not reach |Spearman| <= "
                                f"{args.corr_threshold} for output dimensions {unresolved_dims}."
                            )
                        all_penalties[dim] = torch.from_numpy(penalties['penalty']).float().to(device)
                        print(f"  dim {dim}: penalties={penalties['penalty']}")
                else:
                    print("Warning: Rewards array for debiasing is empty. Skipping debiasing.")
            except Exception as e:
                print(f"Warning: Error during debiasing penalty calculation: {e}. Skipping debiasing.")
            finally:
                del ref_embeddings_for_debiasing, regression_layer_cpu
                if 'pairwise_rewards' in locals(): del pairwise_rewards
                if 'rewards' in locals(): del rewards
        else:
            print("Warning: Reference embeddings tensor is empty or None. Skipping debiasing.")

    if debiasing_enabled and set(all_penalties) != set(debiasing_dims):
        missing_dimensions = sorted(set(debiasing_dims) - set(all_penalties))
        raise RuntimeError(
            f"Debiasing failed for dimensions {missing_dimensions}; refusing to save "
            "an identity or partial transform under a debiased checkpoint name."
        )

    # Build reward transform matrix on device.
    reward_transform_matrix = torch.eye(n_attributes, device=device)
    if not debiasing_enabled:
        print("Debiasing disabled. Using identity reward_transform_matrix.")
    else:
        for dim, penalties_tensor in all_penalties.items():
            reward_transform_matrix[dim, :] -= penalties_tensor
        print(f"Applied debiasing to {len(all_penalties)} dimension(s).")

    # Split by prompt group: repeated preference rows from one prompt cannot leak.
    # Filtered-data candidates always validate on the full source split.
    X_cpu, Z_cpu = prompt_embeddings_cpu, embeddings_cpu
    D_cpu, Y_cpu, G_cpu = difficulties_cpu, domains_cpu, group_ids_cpu
    X_validation_source = Z_validation_source = None
    Y_validation_source = D_validation_source = None
    split_group_ids, val_groups = G_cpu, torch.empty(0, dtype=G_cpu.dtype)
    if args.train_on_all:
        validation_mode = "refit_all_training_data"
        train_idx = torch.arange(len(G_cpu))
        val_idx = torch.empty(0, dtype=torch.long)
        X_train_cpu, X_val_cpu = X_cpu, X_cpu
        Z_train_cpu, Z_val_cpu = Z_cpu, Z_cpu
        Y_train_cpu, Y_val_cpu = Y_cpu, Y_cpu
        D_train_cpu, D_val_cpu = D_cpu, D_cpu
        validation_source_rows = 0
        validation_group_count = 0
        print(f"Refit mode: training on all {len(train_idx)} rows; no validation selection or early stopping.")
    else:
        validation_mode = "external_full" if validation_group_ids_cpu is not None else "internal"
        split_group_ids = validation_group_ids_cpu if validation_group_ids_cpu is not None else G_cpu
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
        split_train_np, val_np = next(
            splitter.split(np.zeros(len(split_group_ids)), groups=split_group_ids.numpy())
        )
        val_idx = torch.from_numpy(val_np)
        val_groups = torch.unique(split_group_ids[val_idx])
        if validation_group_ids_cpu is None:
            train_idx = torch.from_numpy(split_train_np)
            X_validation_source, Z_validation_source = X_cpu, Z_cpu
            Y_validation_source, D_validation_source = Y_cpu, D_cpu
        else:
            train_idx = torch.where(~torch.isin(G_cpu, val_groups))[0]
            X_validation_source = validation_prompt_embeddings_cpu
            Z_validation_source = validation_embeddings_cpu
            Y_validation_source = validation_domains_cpu
            D_validation_source = validation_difficulties_cpu
        if not len(train_idx) or not len(val_idx):
            raise RuntimeError("Grouped split produced an empty training or validation set.")
        overlap = set(G_cpu[train_idx].tolist()) & set(val_groups.tolist())
        if overlap:
            raise RuntimeError(f"Grouped split leaked {len(overlap)} prompt group(s).")
        X_train_cpu, X_val_cpu = X_cpu[train_idx], X_validation_source[val_idx]
        Z_train_cpu, Z_val_cpu = Z_cpu[train_idx], Z_validation_source[val_idx]
        Y_train_cpu, Y_val_cpu = Y_cpu[train_idx], Y_validation_source[val_idx]
        D_train_cpu = D_cpu[train_idx] if D_cpu is not None else None
        D_val_cpu = D_validation_source[val_idx] if D_validation_source is not None else None
        print(
            f"Train={len(train_idx)}, validation={len(val_idx)}, prompt-group overlap=0, "
            f"validation_mode={validation_mode}"
        )
        for d, name in enumerate(DOMAIN_NAMES):
            print(f"  {name}: train={(Y_train_cpu == d).sum().item()}, val={(Y_val_cpu == d).sum().item()}")
        validation_source_rows = len(split_group_ids)
        validation_group_count = int(torch.unique(split_group_ids).numel())

    easy_indices = easy_medium_indices = None
    curriculum_phase1_end = int(args.n_steps * args.curriculum_phase1_frac)
    curriculum_phase2_end = int(args.n_steps * args.curriculum_phase2_frac)
    if args.curriculum and D_train_cpu is not None:
        easy_indices = torch.where(D_train_cpu == 0)[0]
        easy_medium_indices = torch.where(D_train_cpu <= 1)[0]
        if not len(easy_indices):
            args.curriculum = False
    elif args.curriculum:
        print("Warning: no difficulty labels; disabling curriculum.")
        args.curriculum = False

    all_indices = torch.arange(len(X_train_cpu))
    if args.balance_difficulties and D_train_cpu is None:
        print("Warning: no difficulty labels; disabling domain-by-difficulty balancing.")
        args.balance_difficulties = False

    def _sampling_pools(pool):
        if args.balance_difficulties:
            return [
                pool[(Y_train_cpu[pool] == domain) & (D_train_cpu[pool] == level)]
                for domain in range(len(DOMAIN_NAMES))
                for level in range(3)
            ]
        return [pool[Y_train_cpu[pool] == domain] for domain in range(len(DOMAIN_NAMES))]

    balanced_pools = {
        "all": _sampling_pools(all_indices),
        "easy": _sampling_pools(easy_indices) if easy_indices is not None else None,
        "easy_medium": _sampling_pools(easy_medium_indices) if easy_medium_indices is not None else None,
    }
    del embeddings_cpu, prompt_embeddings_cpu, domains_cpu, group_ids_cpu
    del validation_embeddings_cpu, validation_prompt_embeddings_cpu
    del validation_difficulties_cpu, validation_domains_cpu, validation_group_ids_cpu
    del X_validation_source, Z_validation_source, Y_validation_source, D_validation_source
    del split_group_ids, val_groups
    del ref_embeddings_cpu, X_cpu, Z_cpu, D_cpu, Y_cpu, G_cpu
    torch.cuda.empty_cache()

    print(f"Batch size: {args.batch_size}")
    input_dim = X_train_cpu.shape[-1]
    gating_network = GatingNetwork(
        X_train_cpu.shape[-1], n_attributes, n_hidden=args.n_hidden,
        hidden_dim=args.hidden_size, logit_scale=args.logit_scale,
        temperature=args.temperature, dropout=args.dropout,
        learnable_logit_scale=args.learnable_logit_scale,
        active_attribute_indices=active_attribute_indices,
    ).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(gating_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    first_phase = max(curriculum_phase1_end, 1) if args.curriculum else max(args.n_steps, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=first_phase)
    amp_dtype = torch.bfloat16 if device.type == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    active_set = set(active_attribute_indices)
    active_index_tensor = torch.tensor(
        active_attribute_indices, device=device, dtype=torch.long
    )
    domain_indices = [
        torch.tensor(tuple(i for i in DOMAIN_ATTRIBUTE_INDICES[name] if i in active_set), device=device, dtype=torch.long)
        for name in DOMAIN_NAMES
    ]
    active_count = len(active_attribute_indices)
    log_k = float(np.log(active_count))
    entropy_floor = args.entropy_floor_fraction * log_k

    def _routing_losses(probs, labels):
        masses = torch.stack([probs.index_select(-1, ix).sum(-1) for ix in domain_indices], -1)
        domain_loss = F.nll_loss(torch.log(masses.clamp_min(1e-8)), labels)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(-1)
        entropy_loss = F.relu(entropy_floor - entropy).mean()
        mean_probs = probs.mean(0)
        balance_loss = torch.sum(mean_probs * torch.log((mean_probs * active_count).clamp_min(1e-8)))
        return domain_loss, entropy_loss, balance_loss, masses, entropy

    def _sample(pool_name):
        pool_map = {"all": all_indices, "easy": easy_indices, "easy_medium": easy_medium_indices}
        pools = balanced_pools[pool_name]
        active = [p for p in pools if len(p)]
        if not args.balance_domains and not args.balance_difficulties:
            source = pool_map[pool_name]
            return source[torch.randint(len(source), (args.batch_size,))]
        n = int(np.ceil(args.batch_size / len(active)))
        idx = torch.cat([p[torch.randint(len(p), (n,))] for p in active])
        return idx[torch.randperm(len(idx))[:args.batch_size]]

    def _eval_validation():
        gating_network.eval()
        sums = {key: 0.0 for key in (
            "loss", "preference_loss", "domain_loss", "entropy_loss", "balance_loss",
            "learned_correct", "uniform_correct", "oracle_correct", "identity_correct",
            "routing_entropy", "routing_max", "domain_routing_correct")}
        total = 0
        attr_mass = torch.zeros(n_attributes, dtype=torch.float64)
        matrix_ok = torch.zeros(len(DOMAIN_NAMES), 3, dtype=torch.long)
        matrix_n = torch.zeros_like(matrix_ok)
        with torch.no_grad():
            for i in range(0, len(X_val_cpu), args.batch_size * 4):
                stop = i + args.batch_size * 4
                x = X_val_cpu[i:stop].to(device)
                z = Z_val_cpu[i:stop].to(device)
                labels = Y_val_cpu[i:stop].to(device)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    weights = gating_network(x)
                    probs = weights / gating_network.logit_scale.clamp_min(1e-8)
                    raw = z @ regression_layer.T
                    rewards = raw @ reward_transform_matrix
                    scores = torch.sum(rewards * weights[:, None, :], -1)
                    pref = loss_fn(scores[:, 0] - scores[:, 1], torch.ones(len(x), device=device))
                    dl, el, bl, masses, entropy = _routing_losses(probs, labels)
                    loss = pref + args.domain_loss_weight * dl + args.entropy_weight * el + args.load_balance_weight * bl
                    uniform = rewards.index_select(-1, active_index_tensor).sum(-1) * (args.logit_scale / active_count)
                    oracle_w = torch.zeros_like(weights)
                    for d, ix in enumerate(domain_indices):
                        rows = torch.where(labels == d)[0]
                        if len(rows):
                            oracle_w[rows[:, None], ix[None, :]] = args.logit_scale / len(ix)
                    oracle = torch.sum(rewards * oracle_w[:, None, :], -1)
                    identity = torch.sum(raw * weights[:, None, :], -1)
                batch_n = len(x)
                ok = scores[:, 0] > scores[:, 1]
                for key, value in (("loss", loss), ("preference_loss", pref), ("domain_loss", dl), ("entropy_loss", el), ("balance_loss", bl)):
                    sums[key] += value.item() * batch_n
                sums["learned_correct"] += ok.sum().item()
                sums["uniform_correct"] += (uniform[:, 0] > uniform[:, 1]).sum().item()
                sums["oracle_correct"] += (oracle[:, 0] > oracle[:, 1]).sum().item()
                sums["identity_correct"] += (identity[:, 0] > identity[:, 1]).sum().item()
                sums["routing_entropy"] += entropy.sum().item()
                sums["routing_max"] += probs.max(-1).values.sum().item()
                sums["domain_routing_correct"] += (masses.argmax(-1) == labels).sum().item()
                attr_mass += probs.sum(0).double().cpu()
                total += batch_n
                if D_val_cpu is not None:
                    difficulties = D_val_cpu[i:i + batch_n]
                    labels_cpu, ok_cpu = labels.cpu(), ok.cpu()
                    for d in range(len(DOMAIN_NAMES)):
                        for level in range(3):
                            mask = (labels_cpu == d) & (difficulties == level)
                            matrix_n[d, level] += mask.sum()
                            matrix_ok[d, level] += ok_cpu[mask].sum()
        metrics = {key: value / max(total, 1) for key, value in sums.items()}
        metrics["normalized_entropy"] = metrics["routing_entropy"] / log_k
        metrics["effective_attributes"] = float(np.exp(metrics["routing_entropy"]))
        metrics["mean_attribute_mass"] = (attr_mass / max(total, 1)).tolist()
        metrics["mean_domain_mass"] = {
            name: float(sum(metrics["mean_attribute_mass"][i] for i in DOMAIN_ATTRIBUTE_INDICES[name]))
            for name in DOMAIN_NAMES
        }
        metrics["domain_difficulty"] = {
            name: {
                difficulty: (matrix_ok[d, level].item() / matrix_n[d, level].item() if matrix_n[d, level] else None)
                for level, difficulty in enumerate(("easy", "medium", "hard"))
            } for d, name in enumerate(DOMAIN_NAMES)
        }
        metrics["domain_difficulty_counts"] = {
            name: {
                difficulty: matrix_n[d, level].item()
                for level, difficulty in enumerate(("easy", "medium", "hard"))
            } for d, name in enumerate(DOMAIN_NAMES)
        }
        metrics["domain_accuracy"] = {
            name: (matrix_ok[d].sum().item() / matrix_n[d].sum().item() if matrix_n[d].sum() else None)
            for d, name in enumerate(DOMAIN_NAMES)
        }
        valid_domain_accuracies = [value for value in metrics["domain_accuracy"].values() if value is not None]
        metrics["macro_domain_accuracy"] = float(np.mean(valid_domain_accuracies))
        metrics["worst_domain_accuracy"] = float(min(valid_domain_accuracies))
        metrics["preference_accuracy"] = metrics["learned_correct"]
        metrics["mean_top1_mass"] = metrics["routing_max"]
        metrics["max_global_attribute_mass"] = max(metrics["mean_attribute_mass"])
        gating_network.train()
        return metrics

    print(f"Training for {args.n_steps} steps (eval every {args.eval_every})...")
    iterator = tqdm(range(args.n_steps), desc="Training Progress")
    best_val_acc, best_pref_loss, best_state_dict = -1.0, float("inf"), None
    best_step, steps_completed = 0, 0
    evals_without_improvement = 0
    for step in iterator:
        steps_completed = step + 1
        gating_network.train()
        optimizer.zero_grad(set_to_none=True)
        pool_name = "easy" if args.curriculum and step < curriculum_phase1_end else (
            "easy_medium" if args.curriculum and step < curriculum_phase2_end else "all")
        idx = _sample(pool_name)
        if args.curriculum and step in (curriculum_phase1_end, curriculum_phase2_end):
            end = curriculum_phase2_end if step == curriculum_phase1_end else args.n_steps
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(end - step, 1))
            evals_without_improvement = 0
        x, z = X_train_cpu[idx].to(device), Z_train_cpu[idx].to(device)
        labels = Y_train_cpu[idx].to(device)
        try:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                weights = gating_network(x)
                probs = weights / gating_network.logit_scale.clamp_min(1e-8)
                scores = torch.sum((z @ regression_layer.T @ reward_transform_matrix) * weights[:, None, :], -1)
                pref = loss_fn(scores[:, 0] - scores[:, 1], torch.ones_like(scores[:, 0]))
                dl, el, bl, _, _ = _routing_losses(probs, labels)
                loss = pref + args.domain_loss_weight * dl + args.entropy_weight * el + args.load_balance_weight * bl
            if not torch.isfinite(loss):
                print(f"Warning: non-finite loss at step {step}; skipping.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gating_network.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if step % 100 == 0:
                iterator.set_postfix(loss=f"{loss.item():.4f}", pref=f"{pref.item():.4f}", domain=f"{dl.item():.4f}", entropy=f"{el.item():.4f}")
            if (step + 1) % args.eval_every == 0 and not args.train_on_all:
                metrics = _eval_validation()
                print(f"  Step {step+1}: loss={metrics['loss']:.4f}, learned={metrics['learned_correct']:.4f}, uniform={metrics['uniform_correct']:.4f}, oracle={metrics['oracle_correct']:.4f}, route={metrics['domain_routing_correct']:.4f}, H={metrics['normalized_entropy']:.3f}")
                accuracy_improved = metrics["learned_correct"] > best_val_acc + 1e-12
                accuracy_tied = abs(metrics["learned_correct"] - best_val_acc) <= 1e-12
                if accuracy_improved or (accuracy_tied and metrics["preference_loss"] < best_pref_loss):
                    best_val_acc = metrics["learned_correct"]
                    best_pref_loss = metrics["preference_loss"]
                    best_step = step + 1
                    best_state_dict = {k: v.cpu().clone() for k, v in gating_network.state_dict().items()}
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                    if evals_without_improvement >= args.patience:
                        print(f"Early stopping at step {step+1}.")
                        break
        except RuntimeError as error:
            print(f"FATAL ERROR at step {step}: {error}; X={x.shape}, Z={z.shape}")
            traceback.print_exc()
            sys.exit(1)

    if best_state_dict is not None:
        gating_network.load_state_dict(best_state_dict)
    if args.train_on_all:
        best_step = steps_completed
    final_metrics = _eval_validation()
    best_val_loss = final_metrics["loss"]
    best_val_acc = final_metrics["learned_correct"]
    elapsed_seconds = time.perf_counter() - training_started
    print(f"Final validation: loss={best_val_loss:.4f}, learned={best_val_acc:.4f}, uniform={final_metrics['uniform_correct']:.4f}, oracle={final_metrics['oracle_correct']:.4f}")
    model_eval = gating_network
    model_eval.eval()

    # --- Save model checkpoint ---
    save_dir = os.path.join(BASE_DATA_DIR, "gating_network")
    os.makedirs(save_dir, exist_ok=True)
    unique_filename = shared_gate_checkpoint_filename(
        args, args.model_name, pref_base, ref_base,
    )
    save_path = os.path.join(save_dir, unique_filename)
    training_config = {
        "format_version": 2, "shared_prompt_gating": True,
        "in_features": input_dim, "out_features": n_attributes,
        "n_hidden": args.n_hidden, "hidden_size": args.hidden_size,
        "dropout": args.dropout, "temperature": args.temperature,
        "logit_scale": args.logit_scale,
        "learnable_logit_scale": args.learnable_logit_scale,
        "debiasing_dims": list(debiasing_dims),
        "corr_threshold": args.corr_threshold,
        "domain_loss_weight": args.domain_loss_weight,
        "entropy_weight": args.entropy_weight,
        "entropy_floor_fraction": args.entropy_floor_fraction,
        "load_balance_weight": args.load_balance_weight,
        "attribute_subset": args.attribute_subset,
        "active_attribute_indices": list(active_attribute_indices),
        "active_attribute_names": list(active_attribute_names),
        "excluded_attribute_names": list(excluded_attribute_names),
        "balance_domains": args.balance_domains,
        "balance_difficulties": args.balance_difficulties,
        "grouped_split": not args.train_on_all,
        "train_on_all": args.train_on_all,
        "metrics_scope": "training_refit" if args.train_on_all else "validation",
        "seed": args.seed, "val_size": args.val_size,
        "n_steps_requested": args.n_steps, "steps_completed": steps_completed,
        "best_step": best_step, "early_stopping_metric": None if args.train_on_all else "preference_accuracy",
        "eval_every": args.eval_every, "patience": args.patience,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "curriculum": args.curriculum,
        "multi_objective_dataset_name": args.multi_objective_dataset_name,
        "preference_dataset_name": pref_base,
        "validation_preference_dataset_name": validation_pref_base or pref_base,
        "validation_mode": validation_mode,
        "reference_dataset_name": ref_base,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_tag": args.checkpoint_tag,
        "gating_parameter_count": sum(parameter.numel() for parameter in gating_network.parameters()),
    }
    checkpoint_payload = {
        "state_dict": model_eval.state_dict(),
        "reward_transform_matrix": reward_transform_matrix.cpu(),
        "training_config": training_config,
        "validation_metrics": final_metrics,
        "domain_names": list(DOMAIN_NAMES),
        "split": {
            "train_rows": len(train_idx), "validation_rows": len(val_idx),
            "validation_source_rows": validation_source_rows,
            "validation_source_groups": validation_group_count,
            "validation_mode": validation_mode, "group_overlap": 0,
        },
    }
    temporary_fd, temporary_save_path = tempfile.mkstemp(
        prefix=".checkpoint-", suffix=".incomplete", dir=save_dir,
    )
    os.close(temporary_fd)
    try:
        torch.save(checkpoint_payload, temporary_save_path)
        os.replace(temporary_save_path, save_path)
    finally:
        if os.path.exists(temporary_save_path):
            os.unlink(temporary_save_path)
    print(f"Saved gating network state dict to {save_path}")

    # --- Optional eval dataset evaluation ---
    if args.eval and eval_embedding_path_pattern:
        print(f"Evaluating on {args.eval}...")
        all_correct_flags_rb_list = []
        try:
            rb_embeddings_cpu, rb_prompt_embeddings_cpu, _, _, _ = load_embeddings(
                eval_embedding_path_pattern
            )
            if rb_prompt_embeddings_cpu.ndim != 2:
                raise ValueError("RewardBench embeddings must use shared prompt format V2.")
        except ValueError as e:
            print(f"Warning: Could not load RewardBench embeddings: {e}. Skipping evaluation.")
        else:
            rb_batch_size = args.batch_size * 4
            rb_iterator = tqdm(range(0, rb_embeddings_cpu.shape[0], rb_batch_size), desc="RewardBench Eval", leave=False)
            with torch.no_grad():
                for i in rb_iterator:
                    rb_prompt_batch = rb_prompt_embeddings_cpu[i:i+rb_batch_size].to(device, non_blocking=True)
                    rb_embed_batch = rb_embeddings_cpu[i:i+rb_batch_size].to(device, non_blocking=True)

                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                        gating_weights_rb = model_eval(rb_prompt_batch)
                        pred_rb = torch.sum((rb_embed_batch @ regression_layer.T @ reward_transform_matrix) * gating_weights_rb, dim=-1)

                    correct_rb_batch = (pred_rb[:, 0] > pred_rb[:, 1]).float()
                    all_correct_flags_rb_list.append(correct_rb_batch.cpu())

            if all_correct_flags_rb_list:
                all_correct_flags_rb = torch.cat(all_correct_flags_rb_list, dim=0)
                try:
                    reward_bench_ds = datasets.load_dataset(f"allenai/{args.eval}", split=args.eval_split)
                    if len(reward_bench_ds) == len(all_correct_flags_rb):
                        df_examples_rb = pd.DataFrame({"subset": reward_bench_ds["subset"], "correct": all_correct_flags_rb.numpy()})
                        scores_per_section, metrics = eval_reward_bench(df_examples_rb)
                        print("RewardBench Scores:")
                        print(pd.DataFrame([scores_per_section]))
                    else:
                        print(f"Warning: Mismatch RewardBench dataset size ({len(reward_bench_ds)}) vs predictions ({len(all_correct_flags_rb)}). Skipping score calculation.")
                except Exception as e:
                    print(f"Error loading or processing RewardBench dataset for evaluation: {e}")
            else:
                print("Warning: No RewardBench predictions were generated.")


if __name__ == '__main__':
    # Wrap `main()` to keep exit paths and cleanup explicit.
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        exit_code = 130  # Standard exit code for Ctrl+C.
    except SystemExit as e:  # Propagate explicit `sys.exit()` from `main()`.
         exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        traceback.print_exc()
        print(f"------------------------------------\n")
        exit_code = 1
    finally:
        sys.exit(exit_code)
