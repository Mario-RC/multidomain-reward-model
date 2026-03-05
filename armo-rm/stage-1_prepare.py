# stage-1_prepare.py

import os
import sys
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

# Enable TF32 for faster matmul on supported GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Stage 1 Prepare started at {datetime.now().isoformat()}")

# Project-specific regression targets drawn from several evaluation dimensions
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

# Parse CLI arguments
parser = ArgumentParser(description="Stage 1 Prepare: Extract embeddings and labels for multi-objective regression.")
parser.add_argument("--model_path", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1", help="Path or HF ID of the base Reward Model.")
parser.add_argument("--dataset_path", type=str, nargs='+', required=True, help="Path(s) to local .jsonl file(s).")
parser.add_argument("--output_dataset_name", type=str, required=True, help="Unique name for the output dataset folder/file prefix.")
parser.add_argument("--n_shards", type=int, default=1, help="Total number of shards to divide the dataset into.")
parser.add_argument("--shard_idx", type=int, default=1, help="Index of the current shard to process (1-based).")
parser.add_argument("--device", type=int, default=0, help="CUDA device index for model inference (e.g., 0, 1).")
args = parser.parse_args()

# Manually load JSONL records into memory
all_data = []
print(f"Manually loading data from JSON Lines files: {args.dataset_path}")
for path in args.dataset_path:
    if not (os.path.isfile(path) and path.lower().endswith('.jsonl')):
        print(f"ERROR: Path '{path}' is not a valid .jsonl file. Skipping.")
        continue

    print(f"Reading file: {path}")
    loaded_count = 0
    skipped_malformed = 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    # Keep only records that contain a valid messages list
                    if 'messages' in record and isinstance(record['messages'], list):
                        all_data.append(record)
                        loaded_count += 1
                    else:
                        skipped_malformed += 1
                except json.JSONDecodeError:
                    skipped_malformed += 1
        print(f"Successfully loaded {loaded_count} records from {path}. Skipped {skipped_malformed} malformed lines.")
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
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device != 'cpu' else torch.float32,
        attn_implementation="flash_attention_2" if device != 'cpu' else None,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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
max_seq_len = model.config.max_position_embeddings

print(f"Processing {len(ds)} examples in current shard to extract embeddings and labels...")
for example in tqdm(ds, desc=f"Shard {args.shard_idx}/{args.n_shards} Processing"):
    # Format conversation using the model's chat template
    try:
        conv_formatted = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")
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

# Build output directory structure: data/ArmoRM/embeddings/<model>/<dataset_name>
script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, "data")
model_name = args.model_path.split("/")[-1]
output_dataset_folder_name = args.output_dataset_name

final_dir = os.path.join(base_data_dir, "ArmoRM", "embeddings", model_name, output_dataset_folder_name)
try:
    os.makedirs(final_dir, exist_ok=True)
    print(f"Ensured output directory exists: {final_dir}")
except OSError as e:
    print(f"FATAL ERROR: Could not create output directory {final_dir}: {e}")
    sys.exit(1)

# Filename encodes dataset name and shard indices for clarity
file_name = f"{output_dataset_folder_name}-{args.shard_idx:05d}-of-{args.n_shards:05d}.safetensors"
save_path_full = os.path.join(final_dir, file_name)

print(f"Saving embeddings and labels to: {save_path_full}")
try:
    save_file({"embeddings": embeddings_tensor, "labels": labels_tensor}, save_path_full)
    print(f"Successfully saved data for shard {args.shard_idx}/{args.n_shards}.")
except Exception as e:
    print(f"FATAL ERROR: Failed to save safetensors file to {save_path_full}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Stage 1 Prepare finished successfully.")
