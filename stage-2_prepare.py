# stage-2_prepare.py

import json
import os
import math
import sys
import torch
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm.auto import tqdm
from safetensors.torch import load_file, save_file
from argparse import ArgumentParser
from datetime import datetime
from config_utils import load_yaml_config, apply_model_registry, apply_section_overrides
from attributes import DOMAIN_TO_INDEX
from utils import (
    _attention_implementation, _build_save_paths, _resolve_local_dataset_file,
    _load_tokenizer_robust, _requires_remote_code, _stable_int64_id, _tokenize_chat,
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


def _keep_split(example: dict, target_split: str) -> bool:
    """Keep samples matching target_split; keep all when target is 'all'."""
    if target_split == "all":
        return True
    split_value = example.get("split")
    if split_value is None and isinstance(example.get("metadata"), dict):
        split_value = example["metadata"].get("split")
    if split_value is None:
        return True
    return str(split_value).lower() == target_split


def _render_chat_text(tokenizer, messages):
    """Render chat text with a safe fallback when chat_template is unavailable."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback for tokenizers without chat_template metadata.
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _as_messages(value, default_role="assistant"):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [{"role": default_role, "content": value}]
    raise TypeError(f"Expected response text or message list, got {type(value)}")


def _preference_messages(example):
    """Return the original prompt plus full chosen/rejected conversations."""
    chosen = _as_messages(example["chosen"])
    rejected = _as_messages(example["rejected"])
    if isinstance(example.get("messages"), list):
        prompt = list(example["messages"])
    elif isinstance(example.get("prompt"), list):
        prompt = list(example["prompt"])
    elif isinstance(example.get("prompt"), str):
        prompt = [{"role": "user", "content": example["prompt"]}]
    else:
        raise ValueError("Preference example has neither 'messages' nor 'prompt'.")
    return prompt, prompt + chosen, prompt + rejected


def _stable_prompt_group_id(prompt_messages):
    return _stable_int64_id(prompt_messages)


def _stable_pair_id(example, prompt, chosen, rejected):
    explicit_pair_id = example.get("pair_id")
    if explicit_pair_id is not None:
        return _stable_int64_id({"pair_id": str(explicit_pair_id)})
    return _stable_int64_id({
        "prompt": prompt, "chosen": chosen, "rejected": rejected,
    })


# Parse command-line arguments.
parser = ArgumentParser()
parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
parser.add_argument("--model_key", type=str, default=None, help="Model key defined in config.yaml:model_registry.")
parser.add_argument("--model_path", type=str, default=None, help="Path to the pre-trained model (HuggingFace path or local folder).")
parser.add_argument("--model_family", type=str, default="llama3", help="Model family (llama3, gemma2, qwen3, mistral, auto)")
parser.add_argument("--output_dataset_name", type=str, default=None, help="Optional override for output dataset folder/file prefix.")
parser.add_argument("--dataset_path", type=str, default="data/dataset/Multi-Domain-Data-Preference-Pairs", help="Path to the dataset (HuggingFace path or local folder)")
parser.add_argument("--source", default=None, type=str, help="Source filter for the dataset")
parser.add_argument("--prompt_batch_size", type=int, default=8, help="Number of prompt-only conversations encoded per forward pass.")
parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use. Use 'all' to aggregate all available splits.")
parser.add_argument("--n_shards", type=int, default=1, help="Total number of shards to divide the dataset into")
parser.add_argument("--shard_idx", type=int, default=1, help="Index of the current shard")
parser.add_argument("--device", type=int, default=0, help="CUDA device index to use for computation")
parser.add_argument("--seq_len", type=int, default=8192, help="Maximum sequence length for input")
parser.add_argument("--reuse_candidate_embeddings_path", type=str, default=None, help="Reuse aligned chosen/rejected final-token embeddings and only rebuild prompt embeddings.")
parser.add_argument("--selection_manifest", type=str, default=None, help="Optional JSONL manifest of pair_id values to retain without rewriting source data.")
args = parser.parse_args()  # Parse CLI inputs.

config = load_yaml_config(args.config_path)
args = apply_section_overrides(args, config.get("stage_2_prepare", {}))
try:
    args = apply_model_registry(args, config)
except ValueError as error:
    parser.error(str(error))

if args.n_shards < 1 or not 1 <= args.shard_idx <= args.n_shards:
    parser.error("--n_shards must be >= 1 and --shard_idx must be in 1..n_shards.")

if not args.model_path:
    print("FATAL ERROR: --model_path is required (set stage_2_prepare.model_path in config.yaml or pass --model_path).")
    sys.exit(1)

# Validate model family against loaded model config.
config = AutoConfig.from_pretrained(
    args.model_path,
    trust_remote_code=_requires_remote_code(args.model_path),
)
expected_model_types = {
    "llama3": {"llama3", "llama"},
    "gemma2": {"gemma2", "gemma"},
    "mistral": {"mistral"},
    "qwen3": {"qwen3"},
}
if args.model_family != "auto":
    expected = expected_model_types.get(args.model_family)
    if expected is None:
        parser.error(f"Unsupported --model_family: {args.model_family}")
    actual = str(config.model_type).lower()
    if actual not in expected:
        parser.error(
            f"--model_family {args.model_family!r} expects model_type in "
            f"{sorted(expected)}, but {args.model_path!r} reports {actual!r}."
        )


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
                if not _keep_split(record, args.dataset_split.lower()):
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

# Keep only matching split rows when a split column exists.
_target_split = args.dataset_split.lower()
if "split" in ds.column_names and _target_split != "all":
    original_len = len(ds)
    ds = ds.filter(lambda x: str(x.get("split", "train")).lower() == _target_split)
    print(f"Filtered dataset by split={_target_split}: kept {len(ds)} of {original_len} rows.")
if args.source is not None:
    ds = ds.filter(lambda x: x["source"] == args.source)
if args.n_shards > 1:
    ds = ds.shuffle(seed=0)
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)

candidate_source_size = len(ds)
candidate_row_indices = None
if args.selection_manifest:
    if args.n_shards != 1:
        raise ValueError("--selection_manifest requires --n_shards 1.")
    selected_pair_ids = set()
    with open(args.selection_manifest, "r", encoding="utf-8") as manifest_stream:
        for line in manifest_stream:
            manifest_record = json.loads(line)
            if manifest_record.get("pair_id") is not None:
                selected_pair_ids.add(str(manifest_record["pair_id"]))
    if not selected_pair_ids:
        raise ValueError(f"Selection manifest {args.selection_manifest} contains no pair_id values.")
    if "pair_id" not in ds.column_names:
        raise ValueError("--selection_manifest requires a pair_id column in the source dataset.")
    candidate_row_indices = [
        index for index, pair_id in enumerate(ds["pair_id"])
        if str(pair_id) in selected_pair_ids
    ]
    missing = selected_pair_ids - {str(ds[index]["pair_id"]) for index in candidate_row_indices}
    if missing:
        raise ValueError(f"Selection manifest contains {len(missing)} pair_id values absent from the selected split.")
    ds = ds.select(candidate_row_indices)
    print(f"Selection manifest retained {len(ds)} of {candidate_source_size} rows.")

# Load encoder model and tokenizer.
device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
attention_implementation = _attention_implementation(device)
print(
    "Attention implementation: "
    + (attention_implementation or "Transformers default (FlashAttention not required)")
)
model = AutoModel.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    device_map=device,
    attn_implementation=attention_implementation,
    trust_remote_code=_requires_remote_code(args.model_path),
)
tokenizer = _load_tokenizer_robust(args.model_path)

# Optionally reuse the expensive chosen/rejected final-token embeddings.
reused_candidate_embeddings = None
reused_pair_ids = None
if args.reuse_candidate_embeddings_path:
    if args.n_shards != 1:
        raise ValueError("--reuse_candidate_embeddings_path requires --n_shards 1 for row alignment.")
    reuse_payload = load_file(args.reuse_candidate_embeddings_path)
    reused_candidate_embeddings = reuse_payload.get("embeddings")
    if reused_candidate_embeddings is None:
        raise KeyError("Reuse file does not contain an 'embeddings' tensor.")
    reused_pair_ids = reuse_payload.get("pair_ids")
    if reused_pair_ids is None:
        raise KeyError(
            "Reuse file does not contain pair_ids; row alignment cannot be verified. "
            "Regenerate it with the current stage-2_prepare.py."
        )
    expected_reuse_rows = candidate_source_size if candidate_row_indices is not None else len(ds)
    if len(reused_candidate_embeddings) != expected_reuse_rows:
        raise ValueError(
            f"Reuse tensor has {len(reused_candidate_embeddings)} rows; expected {expected_reuse_rows} before manifest selection."
        )
    if len(reused_pair_ids) != expected_reuse_rows:
        raise ValueError(
            f"Reuse pair_ids has {len(reused_pair_ids)} rows; expected {expected_reuse_rows}."
        )
    if reused_candidate_embeddings.ndim != 3 or reused_candidate_embeddings.shape[1] != 2:
        raise ValueError(
            "Reuse embeddings must have shape [N, 2, H] for chosen/rejected candidates."
        )
    print(f"Reusing candidate embeddings from {args.reuse_candidate_embeddings_path}")

# Accumulate shared prompt embeddings and pair metadata.
embeddings = []
prompt_embeddings = []
difficulties = []
domains = []
group_ids = []
pair_ids = []
skipped_pairs = 0

if args.prompt_batch_size < 1:
    raise ValueError("--prompt_batch_size must be at least 1.")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def _embed_prompt_batch(prompt_batch):
    texts = tokenizer.apply_chat_template(
        prompt_batch, tokenize=False, add_generation_prompt=True,
    )
    if isinstance(texts, str):
        texts = [texts]
    encoding = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=args.seq_len,
    )
    encoding = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoding.items()
    }
    with torch.no_grad():
        output = model(**encoding)
        hidden = output.last_hidden_state
        mask = encoding.get("attention_mask")
        if mask is None:
            positions = torch.full(
                (hidden.shape[0],), hidden.shape[1] - 1,
                device=hidden.device, dtype=torch.long,
            )
        else:
            token_positions = torch.arange(
                hidden.shape[1], device=hidden.device
            ).unsqueeze(0)
            positions = (mask.long() * token_positions).argmax(-1)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, positions].cpu()

# Prompt-only passes are batched; candidates can be reused or computed once each.
batch_starts = range(0, len(ds), args.prompt_batch_size)
for batch_start in tqdm(batch_starts, desc="Prompt batches"):
    pending = []
    batch_stop = min(batch_start + args.prompt_batch_size, len(ds))
    for row_index in range(batch_start, batch_stop):
        example = ds[row_index]
        try:
            prompt, full_chosen, full_rejected = _preference_messages(example)
        except (KeyError, TypeError, ValueError):
            skipped_pairs += 1
            continue
        metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
        difficulty = metadata.get("difficulty", example.get("difficulty"))
        domain_name = metadata.get("domain", example.get("domain"))
        pending.append({
            "row_index": row_index,
            "prompt": prompt,
            "chosen": full_chosen,
            "rejected": full_rejected,
            "candidate_row_index": candidate_row_indices[row_index] if candidate_row_indices is not None else row_index,
            "difficulty": difficulty,
            "domain": DOMAIN_TO_INDEX.get(str(domain_name).lower(), -1),
            "group_id": _stable_prompt_group_id(prompt),
            "pair_id": _stable_pair_id(example, prompt, full_chosen, full_rejected),
        })
    if not pending:
        continue

    shared_batch = _embed_prompt_batch([item["prompt"] for item in pending])
    for local_index, item in enumerate(pending):
        if reused_candidate_embeddings is not None:
            reused_index = item["candidate_row_index"]
            if int(reused_pair_ids[reused_index].item()) != item["pair_id"]:
                raise ValueError(
                    f"Reuse pair_id mismatch at source row {reused_index}; "
                    "candidate embeddings are not aligned with the selected dataset."
                )
            pair_embeddings = reused_candidate_embeddings[item["candidate_row_index"]].cpu()
        else:
            candidate_embeddings = []
            try:
                for conversation in (item["chosen"], item["rejected"]):
                    candidate_encoding = _tokenize_chat(
                        tokenizer, conversation, device, args.seq_len,
                    )
                    with torch.no_grad():
                        output = model(**candidate_encoding)
                        hidden = output.last_hidden_state
                        mask = candidate_encoding.get("attention_mask")
                        position = (
                            int(torch.where(mask[0] != 0)[0][-1].item())
                            if mask is not None else hidden.shape[1] - 1
                        )
                        candidate_embeddings.append(hidden[0, position].cpu())
            except (RuntimeError, ValueError):
                skipped_pairs += 1
                continue
            if len(candidate_embeddings) != 2:
                skipped_pairs += 1
                continue
            pair_embeddings = torch.stack(candidate_embeddings)

        difficulty_map = {"easy": 0, "medium": 1, "hard": 2}
        embeddings.append(pair_embeddings)
        prompt_embeddings.append(shared_batch[local_index])
        difficulties.append(
            difficulty_map.get(
                str(item["difficulty"]).lower() if item["difficulty"] else "", 2
            )
        )
        domains.append(item["domain"])
        group_ids.append(item["group_id"])
        pair_ids.append(item["pair_id"])

if not embeddings:
    raise RuntimeError("No complete preference pairs were prepared.")

# Version 2 uses one prompt embedding per pair and carries split metadata.
embeddings = torch.stack(embeddings)
prompt_embeddings = torch.stack(prompt_embeddings)
difficulties_tensor = torch.tensor(difficulties, dtype=torch.int8)
domains_tensor = torch.tensor(domains, dtype=torch.int8)
group_ids_tensor = torch.tensor(group_ids, dtype=torch.int64)
pair_ids_tensor = torch.tensor(pair_ids, dtype=torch.int64)
format_version = torch.tensor([2], dtype=torch.int16)
print(
    f"Prepared {len(embeddings)} pairs with shared prompt embeddings; skipped {skipped_pairs}."
)

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
    {
        "embeddings": embeddings,
        "prompt_embeddings": prompt_embeddings,
        "difficulties": difficulties_tensor,
        "domains": domains_tensor,
        "group_ids": group_ids_tensor,
        "pair_ids": pair_ids_tensor,
        "format_version": format_version,
    },
    save_path_full,
)

# Log output path.
print(f"Saved embeddings to {save_path_full}")
