# stage-1_prepare.py

import os
import sys
import math
import warnings
import torch
import datasets
# from datasets import Features, Value, Sequence # No longer needed for loading
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser
import traceback
import json
from datetime import datetime
from config_utils import load_yaml_config, apply_section_overrides
from utils import _build_save_paths, _resolve_local_dataset_file, _load_tokenizer_robust, _requires_remote_code

# Enable TF32 for faster matmul on supported GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"\n### Stage 1: Prepare started at {datetime.now().isoformat()} ###")


def _is_valid_score_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _has_at_least_one_attribute_score(record: dict) -> bool:
    scores = record.get("scores")
    if not isinstance(scores, dict):
        return False
    for attr in attributes:
        if _is_valid_score_value(scores.get(attr)):
            return True
    return False


def _keep_split(record: dict, target_split: str) -> bool:
    """Split filter aligned with Stage 2: keep all when target is 'all'."""
    if target_split == "all":
        return True
    split_value = record.get("split", "train")
    return str(split_value).lower() == target_split



from attributes import ATTRIBUTES as attributes
print(f"Using {len(attributes)} custom attributes for regression.")

# Parse CLI arguments
parser = ArgumentParser(description="Stage 1 Prepare: Extract embeddings and labels for multi-objective regression.")
parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
parser.add_argument("--model_key", type=str, default=None, help="Model key defined in config.yaml:model:registry.")
parser.add_argument("--model_path", type=str, default=None, help="Path or HF ID of the base Reward Model.")
parser.add_argument("--model_family", type=str, default="llama3", help="Model family (llama3, gemma2, qwen3, auto)")
parser.add_argument("--dataset_path", type=str, nargs='+', default=None, help="Path(s) to local JSON/JSONL files. Extension is optional (e.g. data/Multi-Domain-Data-Scoring).")
parser.add_argument("--output_dataset_name", type=str, default=None, help="Unique name for the output dataset folder/file prefix.")
parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split tag for filtering/naming (e.g., train, all).")
parser.add_argument("--n_shards", type=int, default=1, help="Total number of shards to divide the dataset into.")
parser.add_argument("--shard_idx", type=int, default=1, help="Index of the current shard to process (1-based).")
parser.add_argument("--max_seq_len", type=int, default=None, help="Max sequence length for truncation. If not set, uses model.config.max_position_embeddings.")
parser.add_argument("--device", type=int, default=0, help="CUDA device index for model inference (e.g., 0, 1).")
args = parser.parse_args()

config = load_yaml_config(args.config_path)
args = apply_section_overrides(args, config.get("stage_1_prepare", {}))

target_split = str(args.dataset_split).lower()

# Config values can provide a single string, while CLI with nargs='+' returns a list.
# Normalize here so iteration always treats dataset paths as full path entries.
if isinstance(args.dataset_path, str):
    args.dataset_path = [args.dataset_path]
elif isinstance(args.dataset_path, tuple):
    args.dataset_path = list(args.dataset_path)

if not args.dataset_path:
    print("FATAL ERROR: --dataset_path is required (or set stage_1_prepare.dataset_path in config.yaml).")
    sys.exit(1)
if not args.output_dataset_name:
    print("FATAL ERROR: --output_dataset_name is required (or set stage_1_prepare.output_dataset_name in config.yaml).")
    sys.exit(1)

# Manually load JSONL records into memory
all_data = []
print(f"Manually loading data from JSON Lines files: {args.dataset_path}")
for path in args.dataset_path:
    resolved_path = _resolve_local_dataset_file(path)
    if resolved_path is None:
        print(f"ERROR: Path '{path}' is not a valid local JSON/JSONL file. Skipping.")
        continue

    print(f"Reading file: {resolved_path}")
    loaded_count = 0
    skipped_malformed = 0
    skipped_non_train_split = 0
    skipped_no_attribute_score = 0
    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    if not _keep_split(record, target_split):
                        skipped_non_train_split += 1
                        continue
                    # Keep only records that contain a valid messages list
                    if 'messages' in record and isinstance(record['messages'], list):
                        if _has_at_least_one_attribute_score(record):
                            all_data.append(record)
                            loaded_count += 1
                        else:
                            skipped_no_attribute_score += 1
                    else:
                        skipped_malformed += 1
                except json.JSONDecodeError:
                    skipped_malformed += 1
        print(
            f"Successfully loaded {loaded_count} records from {resolved_path}. "
            f"Skipped {skipped_malformed} malformed lines and "
            f"{skipped_no_attribute_score} records without attribute scores and "
            f"{skipped_non_train_split} records from non-train split."
        )
    except Exception as e:
        print(f"FATAL ERROR: Failed to read or parse file '{path}'.")
        print(f"Details: {e}")
        traceback.print_exc()
        sys.exit(1)

if not all_data:
    print("FATAL ERROR: No valid data loaded from any specified file. Exiting.")
    sys.exit(1)

# Convert list of dicts to a datasets.Dataset for downstream processing
print("Converting loaded data into a Dataset object...")
try:
    ds = datasets.Dataset.from_list(all_data)
    print(f"Total examples in combined dataset: {len(ds)}")
    print("Inferred features by datasets from list:")
    print(ds.features)
except Exception as e:
    print(f"FATAL ERROR: Failed to create Dataset object from loaded data: {e}")
    print("This might indicate data inconsistencies that survived initial loading (e.g., mixed types within a column).")
    traceback.print_exc()
    sys.exit(1)

# Shuffle before sharding to ensure even distribution
print("Shuffling combined dataset...")
ds = ds.shuffle(seed=42)

# Split into shards if requested
if args.n_shards > 1:
    shard_index_0_based = args.shard_idx - 1
    if not (0 <= shard_index_0_based < args.n_shards):
        print(f"FATAL ERROR: Invalid shard_idx ({args.shard_idx}). Must be between 1 and {args.n_shards}.")
        sys.exit(1)
    print(f"Sharding dataset into {args.n_shards} shards. Processing shard {args.shard_idx} (index {shard_index_0_based}).")
    try:
        ds = ds.shard(num_shards=args.n_shards, index=shard_index_0_based)
        print(f"Size of current shard after sharding: {len(ds)}")
        if len(ds) == 0:
            print(f"Warning: Shard {args.shard_idx} is empty. No data will be processed.")
            print("Exiting successfully.")
            sys.exit(0)
    except Exception as e:
        print(f"FATAL ERROR: Failed to shard dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

# Load model and tokenizer on the selected device
device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
print(f"Loading model {args.model_path} onto device {device}...")
try:
    trust_remote_code = _requires_remote_code(args.model_path)
    if trust_remote_code:
        print("Using trust_remote_code=True for Qwen3 model loading compatibility.")

    model = AutoModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if device != 'cpu' else torch.float32,
        attn_implementation="flash_attention_2" if device != 'cpu' else None,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = _load_tokenizer_robust(args.model_path)
    if tokenizer.pad_token is None:
        # Fallback: use EOS token as PAD when PAD is undefined
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token.")
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to load model or tokenizer from {args.model_path}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Extract embeddings and label vectors from each example
embeddings = []
labels = []
skipped_formatting_tokenization = 0
skipped_inference = 0
skipped_label_extraction = 0
if args.max_seq_len is not None:
    max_seq_len = args.max_seq_len
elif hasattr(model.config, "max_position_embeddings"):
    max_seq_len = model.config.max_position_embeddings
else:
    raise ValueError(
        f"Model '{args.model_path}' does not have 'max_position_embeddings' in its config. "
        f"Please specify --max_seq_len explicitly."
    )

print(f"Processing {len(ds)} examples in current shard to extract embeddings and labels...")
for example in tqdm(ds, desc=f"Shard {args.shard_idx}/{args.n_shards} Processing"):
    if not isinstance(example, dict):
        skipped_formatting_tokenization += 1
        continue

    messages = example.get("messages")
    if not isinstance(messages, list):
        skipped_formatting_tokenization += 1
        continue

    # Format conversation using the model's chat template
    try:
        conv_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if tokenizer.bos_token is not None:
            conv_formatted = conv_formatted.replace(tokenizer.bos_token, "")
    except Exception:
        skipped_formatting_tokenization += 1
        continue

    # Tokenize the formatted conversation
    try:
        conv_tokenized = tokenizer(
            conv_formatted, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)
    except Exception:
        skipped_formatting_tokenization += 1
        continue

    # Run the model to obtain the embedding of the last token
    with torch.no_grad():
        try:
            output = model(**conv_tokenized)
            embedding = output.last_hidden_state[0, -1, :].cpu()
            embeddings.append(embedding)
        except Exception:
            skipped_inference += 1
            continue

    # Collect label values in the same order as the attributes list
    try:
        scores_dict = example.get("scores") or {}
        label_values = [scores_dict.get(attr, np.nan) for attr in attributes]
        label = [np.nan if x is None else float(x) for x in label_values]
        labels.append(label)
    except (TypeError, ValueError):
        if len(embeddings) > len(labels):
            embeddings.pop()
        skipped_label_extraction += 1
        continue

# Validate that we produced matched pairs of embeddings and labels
total_skipped = skipped_formatting_tokenization + skipped_inference + skipped_label_extraction
if not embeddings or not labels:
    print(f"ERROR: No valid embeddings or labels extracted. Processed {len(ds)}, skipped {total_skipped}.")
    sys.exit(1)
if len(embeddings) != len(labels):
    print(f"FATAL ERROR: Mismatch between final embeddings ({len(embeddings)}) and labels ({len(labels)}). Logic error likely.")
    sys.exit(1)

print(f"Successfully processed {len(embeddings)} examples.")
print(f"Skipped during formatting/tokenization: {skipped_formatting_tokenization}")
print(f"Skipped during inference: {skipped_inference}")
print(f"Skipped during label extraction: {skipped_label_extraction}")

# Convert lists to tensors for saving
try:
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    embeddings_tensor = torch.stack(embeddings, dim=0)
except Exception as e:
    print(f"FATAL ERROR: Failed to convert extracted data lists to tensors: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"Final Embeddings tensor shape: {embeddings_tensor.shape}")
print(f"Final Labels tensor shape: {labels_tensor.shape}")

# Build output directory structure: model/embeddings/<model>/<dataset_name>
script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, "model")
model_name = args.model_path.split("/")[-1]
output_dataset_folder_name = args.output_dataset_name
dataset_folder_with_split = f"{output_dataset_folder_name}-{args.dataset_split}"

base_file_stem = f"{output_dataset_folder_name}-{args.dataset_split}"
final_dir, save_path_full = _build_save_paths(
    base_data_dir=base_data_dir,
    model_name=model_name,
    dataset_folder=dataset_folder_with_split,
    base_file_stem=base_file_stem,
    n_shards=args.n_shards,
    shard_idx=args.shard_idx,
)
print(f"Ensured output directory exists: {final_dir}")

print(f"Saving embeddings and labels to: {save_path_full}")
try:
    save_file({"embeddings": embeddings_tensor, "labels": labels_tensor}, save_path_full)
    print(f"Successfully saved data for shard {args.shard_idx}/{args.n_shards}.")
except Exception as e:
    print(f"FATAL ERROR: Failed to save safetensors file to {save_path_full}: {e}")
    traceback.print_exc()
    sys.exit(1)

