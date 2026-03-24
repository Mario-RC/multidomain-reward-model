# stage-2_prepare.py

import os
import math
import sys
import torch
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser
from datetime import datetime
from config_utils import load_yaml_config, apply_section_overrides
from utils import (
    _build_save_paths, _resolve_local_dataset_file, _load_tokenizer_robust,
    _requires_remote_code, TOKEN_PATTERNS_BY_MODEL_TYPE as token_patterns,
    find_token_for_gating,
)

# Enable TF32 for faster matmul on supported GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"\n### Stage 2: Prepare started at {datetime.now().isoformat()} ###")


def _is_valid_score_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _extract_score_dicts(example: dict):
    score_dicts = []

    direct_scores = example.get("scores")
    if isinstance(direct_scores, dict):
        score_dicts.append(direct_scores)

    evaluation = example.get("evaluation")
    if isinstance(evaluation, dict):
        stage_1_scores = evaluation.get("stage_1_scores")
        if isinstance(stage_1_scores, dict):
            score_dicts.append(stage_1_scores)

        stage_2_scores = evaluation.get("stage_2_scores")
        if isinstance(stage_2_scores, dict):
            score_dicts.append(stage_2_scores)

    return score_dicts


def _has_at_least_one_attribute_score(example: dict) -> bool:
    score_dicts = _extract_score_dicts(example)
    if not score_dicts:
        # If no score metadata exists, keep the sample (e.g. generic preference datasets).
        return True

    for score_dict in score_dicts:
        for value in score_dict.values():
            if _is_valid_score_value(value):
                return True
    return False


def _is_train_split(example: dict) -> bool:
    """Keep only samples explicitly marked as train; default to keep when missing."""
    split_value = example.get("split")
    if split_value is None and isinstance(example.get("metadata"), dict):
        split_value = example["metadata"].get("split")
    if split_value is None:
        return True
    return str(split_value).lower() == "train"


def _render_chat_text(tokenizer, messages):
    """Render chat text with a safe fallback when chat_template is unavailable."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback for tokenizers without chat_template metadata.
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


# Parse command-line arguments.
parser = ArgumentParser()
parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
parser.add_argument("--model_key", type=str, default=None, help="Model key defined in config.yaml:model:registry.")
parser.add_argument("--model_path", type=str, default=None, help="Path to the pre-trained model (HuggingFace path or local folder).")
parser.add_argument("--model_family", type=str, default="llama3", help="Model family (llama3, gemma2, qwen3, auto)")
parser.add_argument("--output_dataset_name", type=str, default=None, help="Optional override for output dataset folder/file prefix.")
parser.add_argument("--dataset_path", type=str, default="RLHFlow/UltraFeedback-preference-standard", help="Path to the dataset (HuggingFace path or local folder)")
parser.add_argument("--source", default=None, type=str, help="Source filter for the dataset")
parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use. Use 'all' to aggregate all available splits.")
parser.add_argument("--n_shards", type=int, default=1, help="Total number of shards to divide the dataset into")
parser.add_argument("--shard_idx", type=int, default=1, help="Index of the current shard")
parser.add_argument("--device", type=int, default=0, help="CUDA device index to use for computation")
parser.add_argument("--seq_len", type=int, default=8192, help="Maximum sequence length for input")
args = parser.parse_args()  # Parse CLI inputs.

config = load_yaml_config(args.config_path)
args = apply_section_overrides(args, config.get("stage_2_prepare", {}))

if not args.model_path:
    print("FATAL ERROR: --model_path is required (set stage_2_prepare.model_path in config.yaml or pass --model_path).")
    sys.exit(1)

# Validate model family against loaded model config.
config = AutoConfig.from_pretrained(
    args.model_path,
    trust_remote_code=_requires_remote_code(args.model_path),
)
if args.model_family == "llama3":
    assert str(config.model_type).lower() in {"llama3", "llama"}, f"Expected llama/llama3 model_type, got {config.model_type}"
elif args.model_family == "gemma2":
    assert str(config.model_type).lower() in {"gemma2", "gemma"}, f"Expected gemma/gemma2 model_type, got {config.model_type}"
elif args.model_family in {"qwen3", "auto"}:
    pass
else:
    raise ValueError(f"Model family {args.model_family} is not supported")


# Resolve local output directory for generated embeddings.
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(script_dir, "model")

model_name = args.model_path.split("/")[-1]
dataset_base = args.output_dataset_name or args.dataset_path.split("/")[-1]
if args.source is not None:
    dataset_base += f"-{args.source}"
dataset_name = f"{dataset_base}-{args.dataset_split}"

# Final save directory: .../embeddings/<model_name>/<dataset_name>
final_dir = os.path.join(BASE_DATA_DIR, "embeddings", model_name, dataset_name)


# Load dataset and apply optional filtering/sharding.
# Detect if it's a local JSON/JSONL file or a HuggingFace dataset

all_data = []
local_dataset_file = _resolve_local_dataset_file(args.dataset_path)
if local_dataset_file is not None:
    print(f"Manually loading local JSONL file: {local_dataset_file}")
    import json
    kept = 0
    skipped_non_train_split = 0
    skipped_no_attribute_score = 0
    with open(local_dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if not _is_train_split(record):
                    skipped_non_train_split += 1
                    continue
                if _has_at_least_one_attribute_score(record):
                    all_data.append(record)
                    kept += 1
                else:
                    skipped_no_attribute_score += 1
            except Exception as e:
                continue
    print(
        f"Loaded {kept} records from local file and skipped "
        f"{skipped_no_attribute_score} records without attribute scores and "
        f"{skipped_non_train_split} records from non-train split."
    )
    if not all_data:
        print("FATAL ERROR: No valid records left after filtering by attribute scores.")
        sys.exit(1)
    # Create dataset from list (this handles inconsistent dictionaries much better)
    ds = datasets.Dataset.from_list(all_data)
else:
    # Standard loading for HuggingFace hub datasets.
    if args.dataset_split.lower() == "all":
        ds_dict = datasets.load_dataset(args.dataset_path)
        assert isinstance(ds_dict, datasets.DatasetDict)
        available_splits = list(ds_dict.keys())
        if not available_splits:
            print(f"FATAL ERROR: No splits available for dataset {args.dataset_path}.")
            sys.exit(1)
        print(f"Loading all splits from {args.dataset_path}: {available_splits}")
        ds = datasets.concatenate_datasets([ds_dict[split_name] for split_name in available_splits])
    else:
        ds = datasets.load_dataset(args.dataset_path, split=args.dataset_split)
        assert isinstance(ds, datasets.Dataset)

# Keep only train split rows when a split column exists.
if "split" in ds.column_names:
    original_len = len(ds)
    ds = ds.filter(lambda x: str(x.get("split", "train")).lower() == "train")
    print(f"Filtered dataset by split=train: kept {len(ds)} of {original_len} rows.")
if args.source is not None:
    ds = ds.filter(lambda x: x["source"] == args.source)
if args.n_shards > 1:
    ds = ds.shuffle(seed=0)
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)

# Load encoder model and tokenizer.
device = f"cuda:{args.device}"
model = AutoModel.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,  # bf16 reduces memory footprint on supported GPUs.
    device_map=device,
    attn_implementation="flash_attention_2",  # Use FlashAttention v2 when available.
    trust_remote_code=_requires_remote_code(args.model_path),
)
tokenizer = _load_tokenizer_robust(args.model_path)

# Accumulate pair embeddings and prompt embeddings.
embeddings = []
prompt_embeddings = []

# Process each preference pair.
for example in tqdm(ds, desc="Examples"):  # type: ignore[arg-type]
    example: dict
    # Always extract chosen and rejected responses
    chosen_response = example["chosen"]
    rejected_response = example["rejected"]

    # 1. Handle your custom dataset format (uses 'messages' list)
    if "messages" in example:
        full_chosen = example["messages"] + chosen_response
        full_rejected = example["messages"] + rejected_response

    # 2. Handle standard datasets like UltraFeedback/RewardBench (uses 'prompt' string)
    elif "prompt" in example:
        full_chosen = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": chosen_response},
        ]
        full_rejected = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": rejected_response},
        ]
        
    # 3. Fallback for other formats
    else:
        full_chosen = chosen_response
        full_rejected = rejected_response

    pair_embeddings = []
    pair_prompt_embeddings = []

    for iter_example in [full_chosen, full_rejected]:
        # Render conversation text via the tokenizer chat template.
        conv_formatted = _render_chat_text(tokenizer, iter_example)
        # Keep the old Llama 3 behavior without hardcoding a specific checkpoint name.
        if tokenizer.bos_token and conv_formatted.startswith(tokenizer.bos_token):
            conv_formatted = conv_formatted[len(tokenizer.bos_token):]

        # Tokenize and move tensors to the selected CUDA device.
        conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)

        input_ids = conv_tokenized["input_ids"]

        # Single sequence per forward pass (batch size = 1).
        if input_ids.shape[1] > args.seq_len:
            continue

        with torch.no_grad():
            output = model(**conv_tokenized)
            last_hidden_state = output.last_hidden_state[0]

            # Locate gating token and collect prompt/final token embeddings.
            gating_token_position = find_token_for_gating(
                input_ids[0].tolist(), args.model_family
            )
            prompt_embedding = last_hidden_state[gating_token_position].cpu()
            last_token_embedding = last_hidden_state[-1].cpu()

            pair_embeddings.append(last_token_embedding)
            pair_prompt_embeddings.append(prompt_embedding)

    # Keep only complete chosen/rejected pairs.
    if len(pair_embeddings) == 2:
        embeddings.append(torch.stack(pair_embeddings))
        prompt_embeddings.append(torch.stack(pair_prompt_embeddings))

# Stack collected outputs into tensors.
embeddings = torch.stack(embeddings)
prompt_embeddings = torch.stack(prompt_embeddings)

final_dir, save_path_full = _build_save_paths(
    base_data_dir=BASE_DATA_DIR,
    model_name=model_name,
    dataset_folder=dataset_name,
    base_file_stem=dataset_name,
    n_shards=args.n_shards,
    shard_idx=args.shard_idx,
)

# Save embeddings using `safetensors`.
save_file(
    {"embeddings": embeddings, "prompt_embeddings": prompt_embeddings},
    save_path_full,
)

# Log output path.
print(f"Saved embeddings to {save_path_full}")