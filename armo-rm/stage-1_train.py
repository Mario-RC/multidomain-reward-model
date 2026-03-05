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
from safetensors.torch import load_file
from argparse import ArgumentParser
import traceback  # For error logging
from datetime import datetime

print(f"Stage 1 Train started at {datetime.now().isoformat()}")

"""
Perform multi-objective linear regression on precomputed embeddings.
Loads shard embeddings and attribute labels, trains Ridge models per attribute,
selects the best regularization, and saves the resulting weights.
"""

# ---------------------------
# Argument Parsing
# ---------------------------
parser = ArgumentParser(description="Stage 1 Train: Linear probing on precomputed embeddings")
parser.add_argument(
    "--model_path",
    type=str,
    default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    help="Path or HF ID of the base reward model (used for naming output).",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Dataset name produced by stage-1_prepare (e.g., 'mdo').",
)
parser.add_argument(
    "--embeddings_dir",
    type=str,
    default=None,
    help="Optional override for the embeddings root. Defaults to ./data/ArmoRM/embeddings/.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional override for saving regression weights. Defaults to ./data/ArmoRM/regression_weights/.",
)
args = parser.parse_args()

# Derive a short model name for paths
args.model_name = args.model_path.split("/")[-1]

# ---------------------------
# Configuration and setup
# ---------------------------
# This list must match the labels saved by stage-1_prepare.py
attributes = [
    # coherence (co_)
    "co_discourse_structure", "co_logical_consistency", "co_mutual_grounding",
    "co_overall_coherence_score", "co_temporal_causal_coherence", "co_topic_coherence",
    # commonsense (cs_)
    "cs_causality", "cs_coherence", "cs_consistency", "cs_desire",
    "cs_empathy", "cs_reaction",
    # empathy (em_)
    "em_emotional_awareness", "em_emotional_validation", "em_helpful_response",
    "em_overall_empathy_score", "em_perspective_taking", "em_supportive_engagement",
    # multicultural (mu_)
    "mu_coherence", "mu_cultural_specificity", "mu_cultural_value",
    "mu_empathy", "mu_naturalness"
]
print(f"Using {len(attributes)} custom attributes for regression.")

# Base paths
script_dir = os.path.dirname(os.path.abspath(__file__))
default_base_data_dir = os.path.join(script_dir, "data", "ArmoRM")

# Determine embeddings directory
if args.embeddings_dir:
    embeddings_base_dir = args.embeddings_dir
else:
    embeddings_base_dir = os.path.join(default_base_data_dir, "embeddings", args.model_name)

# Specific dataset folder
embeddings_folder_path = os.path.join(embeddings_base_dir, args.dataset_name)
print(f"Looking for embedding files inside: {embeddings_folder_path}")

# Collect shard files
embedding_files = sorted(glob(os.path.join(embeddings_folder_path, "*.safetensors")))

if not embedding_files:
    print(f"FATAL ERROR: No embedding files (*.safetensors) found inside: {embeddings_folder_path}")
    print("Ensure stage-1_prepare.py created this folder, and --dataset_name matches that folder.")
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

results_list = []

# ---------------------------
# Ridge regression per attribute
# ---------------------------
print("Training Ridge regression models for each attribute and alpha...")
for attr_idx in tqdm(range(Y_train.shape[1]), desc="Training Attributes"):
    attribute_name = attributes[attr_idx]
    y_train_attr = Y_train[:, attr_idx]
    y_val_attr = Y_val[:, attr_idx]

    # Masks for valid labels
    valid_mask_train = ~np.isnan(y_train_attr)
    valid_mask_val = ~np.isnan(y_val_attr)

    y_train_filtered = y_train_attr[valid_mask_train]
    X_train_filtered = X_train[valid_mask_train]
    y_val_filtered = y_val_attr[valid_mask_val]
    X_val_filtered = X_val[valid_mask_val]

    # Skip if no data
    if len(X_train_filtered) == 0 or len(X_val_filtered) == 0:
        print(f"Warning: Skipping '{attribute_name}' (index {attr_idx}) due to insufficient data.")
        results_list.append({"attribute": attr_idx, "alpha": np.nan, "loss": np.nan})
        continue

    # Evaluate each alpha
    for alpha in alphas:
        try:
            clf = Ridge(alpha=alpha, fit_intercept=False, solver='cholesky')
            clf.fit(X_train_filtered, y_train_filtered)
            pred = clf.predict(X_val_filtered)
            loss = mean_squared_error(y_val_filtered, pred)
            results_list.append({"attribute": attr_idx, "alpha": alpha, "loss": loss})
        except Exception as e:
            print(f"Warning: Error training attribute {attr_idx} with alpha {alpha}: {e}. Skipping alpha.")
            results_list.append({"attribute": attr_idx, "alpha": alpha, "loss": np.inf})

df_results = pd.DataFrame.from_records(results_list)

if df_results.empty:
    print("FATAL ERROR: No Ridge regression models were trained successfully.")
    sys.exit(1)

# ---------------------------
# Select best alphas by validation loss
# ---------------------------
print("\nSelecting the best alpha for each attribute based on validation loss...")

# Drop skipped/failed rows
df_valid_results = df_results.dropna(subset=['loss'])
df_valid_results = df_valid_results[np.isfinite(df_valid_results['loss'])]

if df_valid_results.empty:
    print("FATAL ERROR: No attributes were trained successfully (all losses were NaN or Inf).")
    sys.exit(1)

best_indices = df_valid_results.groupby("attribute")["loss"].idxmin()
best_alphas = df_valid_results.loc[best_indices]

print("Best alphas selected:")
for _, row in best_alphas.iterrows():
    attr_idx = int(row['attribute'])
    print(f"  Attribute {attr_idx} ({attributes[attr_idx]}): Best alpha = {row['alpha']}, Min Val Loss = {row['loss']:.4f}")

# ---------------------------
# Fit final models with best alphas
# ---------------------------
print("\nFitting final Ridge regression models with the best alphas...")
final_weights_dict = {}
processed_attributes_indices = set()

for _, row in tqdm(best_alphas.iterrows(), total=len(best_alphas), desc="Fitting Final Models"):
    attr_idx = int(row["attribute"])
    best_alpha = row["alpha"]

    if np.isnan(best_alpha):
        continue

    y_train_attr = Y_train[:, attr_idx]
    valid_mask_train = ~np.isnan(y_train_attr)
    X_train_filtered = X_train[valid_mask_train]
    y_train_filtered = y_train_attr[valid_mask_train]

    if len(X_train_filtered) == 0:
        print(f"Warning: No valid training data for attribute {attr_idx} during final fit. Skipping.")
        continue

    try:
        clf_final = Ridge(alpha=best_alpha, fit_intercept=False)
        clf_final.fit(X_train_filtered, y_train_filtered)
        final_weights_dict[attr_idx] = clf_final.coef_
        processed_attributes_indices.add(attr_idx)
    except Exception as e:
        print(f"Error during final fit for attribute {attr_idx}: {e}. Skipping weight extraction.")

# Ensure each attribute has a weight vector
final_weights_list = []
embedding_dim = X_train.shape[1]
for attr_idx in range(len(attributes)):
    if attr_idx in final_weights_dict:
        final_weights_list.append(final_weights_dict[attr_idx])
    else:
        print(f"Warning: Attribute {attr_idx} ({attributes[attr_idx]}) had no valid model. Using zero vector as weights.")
        final_weights_list.append(np.zeros(embedding_dim, dtype=np.float32))

# Stack weights
try:
    weights_array = np.stack(final_weights_list)
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

save_path_weights = os.path.join(save_dir, f"{args.model_name}_{args.dataset_name}.pt")

print(f"Saving regression weights to {save_path_weights}")
try:
    torch.save({"weight": torch.from_numpy(weights_array)}, save_path_weights)
    print("Regression weights saved successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to save weights file to {save_path_weights}: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- END ---
print("\nStage 1 Train finished successfully.")
