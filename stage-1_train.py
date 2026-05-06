# stage-1_train.py

import os
import sys
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from safetensors.torch import load_file
from argparse import ArgumentParser
import traceback  # For error logging
from datetime import datetime
from config_utils import load_yaml_config, apply_section_overrides

print(f"\n### Stage 1: Train started at {datetime.now().isoformat()} ###")

"""
Perform multi-objective linear regression on precomputed embeddings.
Loads shard embeddings and attribute labels, trains Ridge models per attribute,
selects the best regularization, and saves the resulting weights.
"""

# ---------------------------
# Argument Parsing
# ---------------------------
parser = ArgumentParser(description="Stage 1 Train: Linear probing on precomputed embeddings")
parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
parser.add_argument("--model_key", type=str, default=None, help="Model key defined in config.yaml:model:registry.")
parser.add_argument("--model_path", type=str, default=None, help="Path or HF ID of the base reward model (used for naming output).")
parser.add_argument("--multi_objective_dataset_name", type=str, default=None, help="Dataset base name produced by stage-1_prepare (e.g., 'stage_1').")
parser.add_argument("--dataset_split", type=str, default="train", help="Split tag used by stage-1_prepare for folder/filename suffix (e.g., train, all).")
parser.add_argument("--embeddings_dir", type=str, default=None, help="Optional override for the embeddings root. Defaults to ./model/embeddings/.")
parser.add_argument("--output_dir", type=str, default=None, help="Optional override for saving regression weights. Defaults to ./model/regression_weights/.")
parser.add_argument("--model_family", type=str, default="llama3", help="Model family (llama3, gemma2, qwen3, mistral, auto).")
args = parser.parse_args()

config = load_yaml_config(args.config_path)
args = apply_section_overrides(args, config.get("stage_1_train", {}))


if not args.multi_objective_dataset_name:
    print("FATAL ERROR: --multi_objective_dataset_name is required (or set stage_1_train.multi_objective_dataset_name in config.yaml).")
    sys.exit(1)

# Derive a short model name for paths
args.model_name = args.model_path.split("/")[-1]

# ---------------------------
# Configuration and setup
# ---------------------------
from attributes import ATTRIBUTES as attributes
print(f"Using {len(attributes)} custom attributes for regression.")

# Base paths
script_dir = os.path.dirname(os.path.abspath(__file__))
default_base_data_dir = os.path.join(script_dir, "model")

# Determine embeddings directory
if args.embeddings_dir:
    embeddings_base_dir = args.embeddings_dir
else:
    embeddings_base_dir = os.path.join(default_base_data_dir, "embeddings", args.model_name)

# Specific dataset folder matches Stage 1 prepare naming (<dataset>-<split>)
dataset_folder = f"{args.multi_objective_dataset_name}-{args.dataset_split}"
embeddings_folder_path = os.path.join(embeddings_base_dir, dataset_folder)
print(f"Looking for embedding files inside: {embeddings_folder_path}")

# Collect shard files
embedding_files = sorted(glob(os.path.join(embeddings_folder_path, "*.safetensors")))

if not embedding_files:
    print(f"FATAL ERROR: No embedding files (*.safetensors) found inside: {embeddings_folder_path}")
    print("Ensure stage-1_prepare.py created this folder, and --multi_objective_dataset_name matches that folder.")
    sys.exit(1)

# ---------------------------
# Loading embeddings and labels
# ---------------------------
embeddings_list = []
labels_list = []
print(f"Loading embeddings and labels from {len(embedding_files)} Safetensors file(s)...")
for file_path in tqdm(embedding_files, desc="Loading embedding shards"):
    try:
        data = load_file(file_path)  # Loads to CPU
        if "embeddings" not in data or "labels" not in data:
            print(f"Warning: Skipping file {file_path} - missing 'embeddings' or 'labels'.")
            continue
        embeddings_list.append(data["embeddings"])
        labels_list.append(data["labels"])
    except Exception as e:
        print(f"Warning: Failed to load file {file_path}: {e}. Skipping.")
        continue

if not embeddings_list or not labels_list:
    print(f"FATAL ERROR: No valid data could be loaded from files in {embeddings_folder_path}.")
    sys.exit(1)

# Concatenate tensors
try:
    embeddings = torch.cat(embeddings_list, dim=0).float().numpy()
    labels = torch.cat(labels_list, dim=0).float().numpy()
except Exception as e:
    print(f"FATAL ERROR: Failed to concatenate loaded tensors: {e}")
    traceback.print_exc()
    sys.exit(1)

del embeddings_list, labels_list

# ---------------------------
# Reporting loaded data
# ---------------------------
print(f"Total embeddings loaded: {embeddings.shape[0]}")
print(f"Total labels loaded: {labels.shape[0]}")
if embeddings.shape[0] != labels.shape[0]:
    print(f"FATAL ERROR: Mismatch between embeddings ({embeddings.shape[0]}) and labels ({labels.shape[0]}).")
    sys.exit(1)
if labels.shape[1] != len(attributes):
    print(f"FATAL ERROR: Label columns ({labels.shape[1]}) do not match attributes ({len(attributes)}).")
    print("Ensure 'attributes' matches stage-1_prepare.py.")
    sys.exit(1)

# ---------------------------
# Split data
# ---------------------------
print("Splitting data into training and validation sets (random_state=42)...")
try:
    X_train, X_val, Y_train, Y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    X_full = np.concatenate([X_train, X_val], axis=0)
    Y_full = np.concatenate([Y_train, Y_val], axis=0)
    del embeddings, labels
except Exception as e:
    print(f"FATAL ERROR: Failed during train/validation split: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------
# Regularization strengths
# ---------------------------
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
print(f"Using alphas for Ridge regression: {alphas}")

# ---------------------------
# Ridge regression per attribute
# ---------------------------
print("Training Ridge regression models for each attribute...")
print(f"  {'Attribute':<45} {'Alpha':>8} {'Val MSE':>10} {'Pearson':>10} {'Spearman':>10} {'N_val':>8}")
print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
final_weights_dict = {}
final_weights_dict_80pct = {}
any_trained = False

for attr_idx in tqdm(range(Y_train.shape[1]), desc="Training Attributes"):
    attribute_name = attributes[attr_idx]
    y_train_attr = Y_train[:, attr_idx]
    y_val_attr = Y_val[:, attr_idx]

    valid_mask_train = ~np.isnan(y_train_attr)
    valid_mask_val = ~np.isnan(y_val_attr)

    y_train_filtered = y_train_attr[valid_mask_train]
    X_train_filtered = X_train[valid_mask_train]
    y_val_filtered = y_val_attr[valid_mask_val]
    X_val_filtered = X_val[valid_mask_val]

    n_val = len(y_val_filtered)

    if len(X_train_filtered) == 0 or len(X_val_filtered) == 0:
        print(f"  Warning: Skipping '{attribute_name}' (index {attr_idx}) due to insufficient data.")
        continue

    # Search over alphas, keep the best model directly.
    best_loss = np.inf
    best_clf = None
    best_alpha = np.nan

    for alpha in alphas:
        try:
            clf = Ridge(alpha=alpha, fit_intercept=False, solver='cholesky')
            clf.fit(X_train_filtered, y_train_filtered)
            pred = clf.predict(X_val_filtered)
            loss = mean_squared_error(y_val_filtered, pred)
            if loss < best_loss:
                best_loss = loss
                best_clf = clf
                best_alpha = alpha
        except Exception as e:
            print(f"  Warning: Error training attribute {attr_idx} with alpha {alpha}: {e}. Skipping alpha.")

    if best_clf is None:
        print(f"  Warning: No valid model for '{attribute_name}'. Skipping.")
        continue

    # Retrain with best alpha on full data (train + val) for final weights.
    y_full_attr = Y_full[:, attr_idx]
    valid_mask_full = ~np.isnan(y_full_attr)
    try:
        final_clf = Ridge(alpha=best_alpha, fit_intercept=False, solver='cholesky')
        final_clf.fit(X_full[valid_mask_full], y_full_attr[valid_mask_full])
        final_weights_dict[attr_idx] = final_clf.coef_
    except Exception as e:
        print(f"  Warning: Failed to retrain '{attribute_name}' on full data: {e}. Using 80% weights.")
        final_weights_dict[attr_idx] = best_clf.coef_
    final_weights_dict_80pct[attr_idx] = best_clf.coef_
    any_trained = True

    pearson_r = np.nan
    spearman_r = np.nan
    if n_val >= 2:
        pred_val = best_clf.predict(X_val_filtered)
        if np.std(y_val_filtered) > 0 and np.std(pred_val) > 0:
            pearson_r = pearsonr(y_val_filtered, pred_val)[0]
            spearman_r = spearmanr(y_val_filtered, pred_val)[0]

    print(f"  {attribute_name:<45} {best_alpha:>8.1f} {best_loss:>10.4f} {pearson_r:>10.4f} {spearman_r:>10.4f} {n_val:>8}")

if not any_trained:
    print("FATAL ERROR: No attributes were trained successfully.")
    sys.exit(1)

# Ensure each attribute has a weight vector (100% and 80% variants).
final_weights_list = []
final_weights_list_80pct = []
embedding_dim = X_train.shape[1]
for attr_idx in range(len(attributes)):
    zero = np.zeros(embedding_dim, dtype=np.float32)
    if attr_idx in final_weights_dict:
        final_weights_list.append(final_weights_dict[attr_idx])
    else:
        print(f"Warning: Attribute {attr_idx} ({attributes[attr_idx]}) had no valid model. Using zero vector as weights.")
        final_weights_list.append(zero)
    final_weights_list_80pct.append(final_weights_dict_80pct.get(attr_idx, zero))

# Stack weights
try:
    weights_array = np.stack(final_weights_list)
    weights_array_80pct = np.stack(final_weights_list_80pct)
    print(f"\nFinal regression weights matrix shape: {weights_array.shape}")
    if weights_array.shape[0] != len(attributes):
        print(f"FATAL ERROR: Weight rows ({weights_array.shape[0]}) != attributes ({len(attributes)}).")
        sys.exit(1)
    if weights_array.shape[1] != embedding_dim:
        print(f"FATAL ERROR: Weight cols ({weights_array.shape[1]}) != embedding dimension ({embedding_dim}).")
        sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: Failed to stack final weights: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------
# Save regression weights
# ---------------------------
if args.output_dir:
    save_dir = args.output_dir
else:
    save_dir = os.path.join(default_base_data_dir, "regression_weights")

try:
    os.makedirs(save_dir, exist_ok=True)
    print(f"Ensured output directory for weights exists: {save_dir}")
except OSError as e:
    print(f"FATAL ERROR: Could not create output directory {save_dir}: {e}")
    sys.exit(1)

save_path_weights = os.path.join(save_dir, f"{args.model_name}_{args.multi_objective_dataset_name}_100pct.pt")
save_path_weights_80pct = os.path.join(save_dir, f"{args.model_name}_{args.multi_objective_dataset_name}_80pct.pt")

print(f"Saving 100% regression weights to {save_path_weights}")
try:
    torch.save({"weight": torch.from_numpy(weights_array)}, save_path_weights)
    print("100% regression weights saved successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to save weights file to {save_path_weights}: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"Saving 80% regression weights to {save_path_weights_80pct}")
try:
    torch.save({"weight": torch.from_numpy(weights_array_80pct)}, save_path_weights_80pct)
    print("80% regression weights saved successfully.")
except Exception as e:
    print(f"Warning: Failed to save 80% weights file to {save_path_weights_80pct}: {e}")

# --- END ---
