# stage-2_train.py

import os
import sys
import torch    
import numpy as np
from safetensors.torch import load_file
from argparse import ArgumentParser
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import pandas as pd
from glob import glob
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import datasets
import traceback  # Used for detailed error traces
from config_utils import load_yaml_config, apply_section_overrides

from datetime import datetime

# Enable TF32 for better throughput on Ampere+ GPUs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"\n### Stage 2: Train started at {datetime.now().isoformat()} ###")

from attributes import ATTRIBUTES as attributes

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
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        self.dropout_prob = dropout
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
        x = F.softmax(x / self.temperature, dim=-1)  # Use `dim=-1` for shape generality.
        return x * self.logit_scale  # Learnable global output scaling.

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------
def find_debiasing_penalties(cluster_V, debiasing_dim=4, corr_threshold=0.028):
    """
    Find per-dimension penalties that decorrelate all other reward dimensions
    from a chosen target dimension.

    For each dimension d != debiasing_dim, iteratively increases a penalty
    factor until the absolute Spearman correlation between the adjusted d
    and the raw debiasing_dim falls below `corr_threshold`.  The adjusted
    value is: V_d' = V_d - penalty * V_debiasing_dim.

    Args:
        cluster_V (np.ndarray): Array of shape [N, K] containing multi-objective rewards.
        debiasing_dim (int): Index of the dimension to decorrelate from.
        corr_threshold (float): Maximum allowed absolute Spearman correlation.
    Returns:
        dict: Contains 'penalty' (np.ndarray of penalties per dim) and 'corr' (np.ndarray of final correlations).
    """
    penalty_candidates = sorted([
        0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    ])
    K = cluster_V.shape[1]
    candidate_dims = set(range(K))

    if debiasing_dim not in candidate_dims:
        print(f"Warning: debiasing_dim {debiasing_dim} is out of bounds (0-{K-1}). Skipping debiasing.")
        return {"penalty": np.ones(K), "corr": np.ones(K)}
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
            try:
                corr, p_value = spearmanr(V_adjusted[:, dim], cluster_V[:, debiasing_dim])
                if np.isnan(corr):
                    corr = 0.0  # Treat NaN as zero correlation.
            except ValueError:  # Handles constant-valued arrays.
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

    return {"penalty": dimwise_penalties, "corr": final_corr_array}


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


def load_embeddings(embedding_path_pattern):
    """
    Load embedding pairs from `.safetensors` files.

    Returns concatenated tensors on CPU with basic integrity checks.
    """
    file_paths = sorted(glob(embedding_path_pattern))
    if not file_paths:
        raise ValueError(f"No embedding files found matching pattern: {embedding_path_pattern}")

    embeddings_list, prompt_embeddings_list, difficulties_list = [], [], []
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
        except Exception as e:
            print(f"Warning: Failed to load or process file {embedding_path}: {e}")
            continue  # Skip corrupted or unreadable files.

    if not embeddings_list or not prompt_embeddings_list:
         raise ValueError(f"No valid embeddings could be loaded from {embedding_path_pattern}. Check file integrity and ensure keys ('embeddings', 'prompt_embeddings') exist.")

    # Concatenate on CPU and cast to float32 for consistency.
    embeddings_cpu = torch.cat(embeddings_list, dim=0).float()
    prompt_embeddings_cpu = torch.cat(prompt_embeddings_list, dim=0).float()
    difficulties_cpu = torch.cat(difficulties_list, dim=0) if len(difficulties_list) == len(embeddings_list) else None
    print(f"Successfully loaded a total of {len(embeddings_cpu)} embedding pairs into CPU RAM.")
    return embeddings_cpu, prompt_embeddings_cpu, difficulties_cpu


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    """Main function to parse arguments, load data, train the model, and evaluate."""
    parser = ArgumentParser(description="Train ArmoRM Gating Network")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_key", type=str, default=None, help="Model key defined in config.yaml:model:registry.")
    parser.add_argument("--model_path", type=str, default=None, help="Path or HF ID of the base Reward Model")
    parser.add_argument("--multi_objective_dataset_name", type=str, default=None, help="Dataset name from stage-1_prepare output (e.g., 'stage_1').")
    parser.add_argument("--preference_dataset_name", type=str, default=None, help="Preference dataset folder name (matches stage-2_prepare output_dataset_name). Required.")
    parser.add_argument("--reference_dataset_name", type=str, default=None, help="Reference dataset folder name (matches stage-2_prepare output_dataset_name). If null, uses preference_dataset_name.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Split suffix used by stage-2_prepare outputs (e.g., train, all, val, test)."    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for AdamW optimizer")
    parser.add_argument("--n_steps", type=int, default=2000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--debiasing_dims", type=int, nargs="+", default=[-1], help="Indices (0-based) of attribute dimensions to decorrelate from all others. Set to -1 to disable debiasing. Multiple values supported, e.g. --debiasing_dims 21 18.")
    parser.add_argument("--corr_threshold", type=float, default=0.03, help="Maximum allowed absolute Spearman correlation for debiasing")
    parser.add_argument("--model_family", type=str, default="llama3", choices=["llama3", "gemma2", "qwen3", "auto"], help="Model family for token pattern matching during embedding extraction (if applicable, less relevant here)")
    parser.add_argument("--eval", type=str, default=None, help="Eval dataset name (e.g. reward-bench). Requires embeddings from stage-2_prepare.")
    parser.add_argument("--eval_split", type=str, default="filtered", help="Split suffix for the eval dataset (default: filtered).")
    parser.add_argument("--logit_scale", type=float, default=1.0, help="Scaling factor applied after softmax in the gating network")
    parser.add_argument("--temperature", type=float, default=10.0, help="Temperature for softmax scaling in the gating network")
    parser.add_argument("--n_hidden", type=int, default=3, help="Number of hidden layers in the gating network MLP")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Dimension of hidden layers in the gating network")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability in the gating network's hidden layers")
    parser.add_argument("--max_samples", type=int, default=None, help="Load only the first N samples from datasets (for debugging RAM issues)")
    parser.add_argument("--eval_every", type=int, default=200, help="Evaluate on validation set every N steps")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (number of evaluations without improvement)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--curriculum", action="store_true", default=False, help="Enable phased curriculum learning: easy → easy+medium → all")
    parser.add_argument("--curriculum_phase1_frac", type=float, default=0.20, help="Fraction of n_steps for easy-only phase (default: 0.20)")
    parser.add_argument("--curriculum_phase2_frac", type=float, default=0.50, help="Fraction of n_steps to end easy+medium phase (default: 0.50)")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    args = apply_section_overrides(args, config.get("stage_2_train", {}))

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    # Seed RNGs for reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Resolve local base paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DATA_DIR = os.path.join(script_dir, "model")
    # ----------------------------------

    # Validate preference_dataset_name (required).
    if not args.preference_dataset_name:
        print("FATAL ERROR: --preference_dataset_name is required (set stage_2_train.preference_dataset_name in config.yaml or pass --preference_dataset_name).")
        sys.exit(1)

    # Resolve reference_dataset_name (fallback: preference_dataset_name for embeddings).
    # Pass --reference_dataset_name null to use preference data for debiasing.
    _ref_is_null = args.reference_dataset_name is None or args.reference_dataset_name.lower() == "null"
    if _ref_is_null:
        args.reference_dataset_name = args.preference_dataset_name
        print(f"NOTE: No reference_dataset_name specified. Using preference_dataset_name ({args.preference_dataset_name}) for debiasing.")

    # Extract short names used in filesystem paths.
    args.model_name = args.model_path.split("/")[-1]
    # Match stage-2_prepare output naming convention: <dataset>-<dataset_split>.
    pref_base = args.preference_dataset_name
    ref_base = "null" if _ref_is_null else args.reference_dataset_name
    args.preference_dataset_name = f"{pref_base}-{args.dataset_split}"
    args.reference_dataset_name = f"{ref_base}-{args.dataset_split}" if not _ref_is_null else f"{args.reference_dataset_name}-{args.dataset_split}"

    # --- Define load paths ---
    # Preference embeddings path pattern (inside dataset-split folder).
    preference_embedding_path_pattern = os.path.join(
        BASE_DATA_DIR, "embeddings", args.model_name, args.preference_dataset_name, "*.safetensors"
    )
    # Regression weights file path.
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
    # Reference embeddings path pattern.
    reference_embedding_path_pattern = os.path.join(
        BASE_DATA_DIR, "embeddings", args.model_name, args.reference_dataset_name, "*.safetensors"
    )
    # -------------------------

    # Print paths.
    print(f"Preference Embedding Path Pattern: {preference_embedding_path_pattern}")
    print(f"Regression Layer Path: {regression_layer_path}")
    print(f"Reference Embedding Path Pattern: {reference_embedding_path_pattern}")
    if eval_embedding_path_pattern:
        print(f"Eval Embedding Path Pattern: {eval_embedding_path_pattern}")

    # Load data to CPU with robust error handling.
    try:
        print("Loading preference embeddings (to CPU RAM)...")
        embeddings_cpu, prompt_embeddings_cpu, difficulties_cpu = load_embeddings(preference_embedding_path_pattern)

        if args.max_samples is not None and args.max_samples < len(embeddings_cpu):
            print(f"NOTE: Subsetting preference data to first {args.max_samples} samples.")
            indices = torch.arange(args.max_samples)
            embeddings_cpu = embeddings_cpu[indices]
            prompt_embeddings_cpu = prompt_embeddings_cpu[indices]
            if difficulties_cpu is not None:
                difficulties_cpu = difficulties_cpu[indices]

        print("Loading regression layer (to device)...")
        regression_layer = torch.load(regression_layer_path, map_location=device, weights_only=True)["weight"].float()
        n_attributes, hidden_size = regression_layer.shape

        ref_embeddings_cpu = None
        _debiasing_requested = any(d >= 0 for d in args.debiasing_dims)
        if _debiasing_requested:
            print("Loading reference embeddings for debiasing (to CPU RAM)...")
            ref_embeddings_cpu, _, _ = load_embeddings(reference_embedding_path_pattern)

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

    # Build reward transform matrix on device.
    reward_transform_matrix = torch.eye(n_attributes, device=device)
    if not debiasing_enabled:
        print("Debiasing disabled. Using identity reward_transform_matrix.")
    else:
        for dim, penalties_tensor in all_penalties.items():
            reward_transform_matrix[dim, :] -= penalties_tensor
        print(f"Applied debiasing to {len(all_penalties)} dimension(s).")

    # Keep dataset tensors on CPU; move mini-batches on demand.
    X_cpu = prompt_embeddings_cpu
    Z_cpu = embeddings_cpu
    D_cpu = difficulties_cpu  # may be None if embeddings were generated without difficulty labels

    # Split train/validation sets on CPU.
    print("Splitting data into train/validation sets (CPU)...")
    if D_cpu is not None:
        X_train_cpu, X_val_cpu, Z_train_cpu, Z_val_cpu, D_train_cpu, _ = train_test_split(
            X_cpu, Z_cpu, D_cpu, test_size=0.2, random_state=args.seed, shuffle=True
        )
    else:
        X_train_cpu, X_val_cpu, Z_train_cpu, Z_val_cpu = train_test_split(
            X_cpu, Z_cpu, test_size=0.2, random_state=args.seed, shuffle=True
        )
        D_train_cpu = None
    print(f"Train size: {len(X_train_cpu)}, Validation size: {len(X_val_cpu)}")

    # Build curriculum phase index pools and phase boundaries.
    easy_indices = None
    easy_medium_indices = None
    curriculum_phase1_end = int(args.n_steps * args.curriculum_phase1_frac)
    curriculum_phase2_end = int(args.n_steps * args.curriculum_phase2_frac)
    if args.curriculum:
        if D_train_cpu is not None:
            easy_indices = torch.where(D_train_cpu == 0)[0]
            easy_medium_indices = torch.where(D_train_cpu <= 1)[0]
            print(f"Curriculum enabled — easy: {len(easy_indices)}, easy+medium: {len(easy_medium_indices)}, all: {len(X_train_cpu)}")
            if len(easy_indices) == 0:
                print("Warning: No 'easy' examples found in training set. Disabling curriculum.")
                args.curriculum = False
        else:
            print("Warning: No difficulty labels in embeddings. Disabling curriculum (re-run stage-2_prepare to generate them).")
            args.curriculum = False

    # Release original large CPU tensors after split.
    del embeddings_cpu, prompt_embeddings_cpu, ref_embeddings_cpu, X_cpu, Z_cpu, D_cpu
    torch.cuda.empty_cache()  # Hint CUDA allocator to release cached blocks.

    print(f"Batch size: {args.batch_size}")

    # Initialize gating network on selected device.
    print("Initializing gating network...")
    input_dim = X_train_cpu.shape[-1]  # Input feature dimension.
    gating_network = GatingNetwork(
        input_dim, n_attributes, n_hidden=args.n_hidden, hidden_dim=args.hidden_size,
        logit_scale=args.logit_scale, temperature=args.temperature, dropout=args.dropout,
    ).to(device)

    # Optimizer, loss, and scheduler.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(gating_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.curriculum:
        # Per-phase cosine schedule with warm restarts at phase boundaries.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(curriculum_phase1_end, 1))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)
    # Choose AMP dtype based on hardware support.
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    print(f"Using Automatic Mixed Precision (AMP) with dtype: {amp_dtype}")

    # --- Helper: compute validation loss and accuracy ---
    def _eval_validation():
        gating_network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        vbs = args.batch_size * 4
        with torch.no_grad():
            for i in range(0, X_val_cpu.shape[0], vbs):
                X_vb = X_val_cpu[i:i+vbs].to(device, non_blocking=True)
                Z_vb = Z_val_cpu[i:i+vbs].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    gw = gating_network(X_vb)
                    pred = torch.sum((Z_vb @ regression_layer.T @ reward_transform_matrix) * gw, dim=-1)
                    batch_loss = loss_fn(pred[:, 0] - pred[:, 1], torch.ones(pred.shape[0], device=device))
                total_loss += batch_loss.item() * X_vb.shape[0]
                correct += ((pred[:, 0] - pred[:, 1]) > 0).sum().item()
                total += X_vb.shape[0]
        gating_network.train()
        val_loss = total_loss / total if total > 0 else float('inf')
        val_acc = correct / total if total > 0 else 0.0
        return val_loss, val_acc

    # --- Training loop with early stopping (based on validation loss) ---
    print(f"Starting training for {args.n_steps} steps (eval every {args.eval_every}, patience {args.patience})...")
    iterator = tqdm(range(args.n_steps), desc="Training Progress")

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state_dict = None
    evals_without_improvement = 0

    for step in iterator:
        gating_network.train()  # Enable training behavior (e.g., dropout).
        optimizer.zero_grad(set_to_none=True)

        # Sample CPU indices for this step.
        if args.curriculum:
            if step < curriculum_phase1_end:
                pool = easy_indices
            elif step < curriculum_phase2_end:
                pool = easy_medium_indices
            else:
                pool = None
            if pool is not None:
                idx = pool[torch.randint(0, len(pool), (args.batch_size,), device="cpu")]
            else:
                idx = torch.randint(0, X_train_cpu.shape[0], (args.batch_size,), device="cpu")
        else:
            idx = torch.randint(0, X_train_cpu.shape[0], (args.batch_size,), device="cpu")

        # Warm restart LR at phase transitions.
        if args.curriculum:
            if step == curriculum_phase1_end:
                phase2_steps = max(curriculum_phase2_end - curriculum_phase1_end, 1)
                for pg in optimizer.param_groups:
                    pg['lr'] = args.learning_rate
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_steps)
                evals_without_improvement = 0
                print(f"  Phase 2 started at step {step} — LR warm restart to {args.learning_rate}")
            elif step == curriculum_phase2_end:
                phase3_steps = max(args.n_steps - curriculum_phase2_end, 1)
                for pg in optimizer.param_groups:
                    pg['lr'] = args.learning_rate
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase3_steps)
                evals_without_improvement = 0
                print(f"  Phase 3 started at step {step} — LR warm restart to {args.learning_rate}")

        # Transfer only the current mini-batch to device.
        X_batch = X_train_cpu[idx].to(device, non_blocking=True)
        Z_batch = Z_train_cpu[idx].to(device, non_blocking=True)

        try:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                gating_weights = gating_network(X_batch)
                # Predicted score per candidate: sum((Z @ W^T) * gating_weights).
                # Apply debiasing transform before combining with gating weights.
                pred_scores = torch.sum((Z_batch @ regression_layer.T @ reward_transform_matrix) * gating_weights, dim=-1)
                # Pairwise preference loss: chosen should score higher than rejected.
                loss = loss_fn(pred_scores[:, 0] - pred_scores[:, 1], torch.ones_like(pred_scores[:, 0]))

            # Guard against NaN/Inf before backward pass.
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss ({loss.item()}) at step {step}. Skipping update.")
                continue  # Skip this optimization step.

            # Scaled backward/update for AMP.
            scaler.scale(loss).backward()
            # Add gradient clipping here if training becomes unstable.
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 100 == 0:
                 current_lr = scheduler.get_last_lr()[0]
                 iterator.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{current_lr:.1e}"})

            # --- Periodic validation & early stopping (based on val loss) ---
            if (step + 1) % args.eval_every == 0:
                val_loss, val_acc = _eval_validation()
                print(f"  Step {step+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f} (best_loss={best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_state_dict = {k: v.cpu().clone() for k, v in gating_network.state_dict().items()}
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                    if evals_without_improvement >= args.patience:
                        print(f"  Early stopping at step {step+1} (no improvement for {args.patience} evals). Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
                        break

        except RuntimeError as e:
            # Surface detailed tensor shapes for common runtime failures.
            print(f"FATAL ERROR during training step {step}: {e}")
            print(f"Shapes - X_batch: {X_batch.shape}, Z_batch: {Z_batch.shape}, W.T: {regression_layer.T.shape}, Gating: {gating_weights.shape if 'gating_weights' in locals() else 'N/A'}")
            traceback.print_exc()
            sys.exit(1)

    # --- Restore best checkpoint ---
    if best_state_dict is not None:
        gating_network.load_state_dict(best_state_dict)
        print(f"\nRestored best checkpoint (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
    else:
        # No eval was run (n_steps < eval_every), evaluate now
        best_val_loss, best_val_acc = _eval_validation()
        print(f"\nFinal Validation Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.4f}")

    acc_val = best_val_acc
    model_eval = gating_network
    model_eval.eval()

    # --- Save model checkpoint ---
    save_dir = os.path.join(BASE_DATA_DIR, "gating_network")
    os.makedirs(save_dir, exist_ok=True)
    # Build checkpoint name — include all hyperparams to guarantee unique filenames
    _hp_defaults = {"learning_rate": 0.001, "weight_decay": 0.0, "n_hidden": 3, "hidden_size": 1024, "dropout": 0.2, "batch_size": 1024, "corr_threshold": 0.03, "logit_scale": 1.0}
    _hp_suffix = "".join(
        f"_{k[:2]}{getattr(args, k)}" for k in _hp_defaults
    )
    _curr_suffix = "_cv" if args.curriculum else ""
    unique_name = (
        f"gating_network_{args.model_name}_mo_{args.multi_objective_dataset_name}_"
        f"pref_{pref_base}_ref_{ref_base}_t{args.temperature:.1f}_n{args.n_steps}_seed{args.seed}{_hp_suffix}{_curr_suffix}"
    )
    save_path = os.path.join(save_dir, f"{unique_name}.pt")
    torch.save({
        "state_dict": model_eval.state_dict(),
        "reward_transform_matrix": reward_transform_matrix.cpu(),
    }, save_path)
    print(f"Saved gating network state dict to {save_path}")

    # --- Optional eval dataset evaluation ---
    if args.eval and eval_embedding_path_pattern:
        print(f"Evaluating on {args.eval}...")
        all_correct_flags_rb_list = []
        try:
            rb_embeddings_cpu, rb_prompt_embeddings_cpu, _ = load_embeddings(eval_embedding_path_pattern)
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