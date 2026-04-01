# utils.py — Shared utility functions for the multidomain_model pipeline.

import json
import os
from typing import Optional, Sequence

import torch
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Remote-code detection
# ---------------------------------------------------------------------------

def _requires_remote_code(model_path: str) -> bool:
    """Return True when the model needs trust_remote_code=True."""
    model_path_l = str(model_path).lower()
    return "qwen3" in model_path_l


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def _load_tokenizer_robust(model_path: str):
    """Load tokenizer with fallback to slow tokenizer when fast conversion deps are missing."""
    trust_remote_code = _requires_remote_code(model_path)
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except (ValueError, ImportError) as e:
        print(f"Warning: Fast tokenizer load failed ({e}). Retrying with use_fast=False...")
        return AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=trust_remote_code)


# ---------------------------------------------------------------------------
# Dataset / file resolution
# ---------------------------------------------------------------------------

def _resolve_local_dataset_file(dataset_path: str):
    """Resolve local JSON/JSONL path, accepting optional missing extension."""
    candidate_paths = [dataset_path]
    if not dataset_path.endswith(".jsonl") and not dataset_path.endswith(".json"):
        candidate_paths.extend([f"{dataset_path}.jsonl", f"{dataset_path}.json"])

    for candidate in candidate_paths:
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_jsonl_path(path: str) -> str:
    """Return *path* if it exists, otherwise try appending .jsonl."""
    if os.path.isfile(path):
        return path
    candidate = path + ".jsonl"
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Dataset not found: {path} (also tried {candidate})")


def load_cultural_test(data_dir: str) -> list[dict]:
    """Load all JSON cultural test files from *data_dir* and return a flat list of records."""
    records: list[dict] = []
    if not os.path.isdir(data_dir):
        return records
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            rows = json.load(f)
        records.extend(rows)
    return records


def parse_cultural_conversation(record: dict) -> list[dict]:
    """Parse a cultural test record's conversation field into chat messages.

    Maps the first speaker to 'user', the second to 'assistant', and merges
    consecutive turns from the same speaker.
    """
    conv = record.get("conversation", "")
    lines = conv.split("\\n")
    messages: list[dict] = []
    speakers: dict[str, str] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        idx = line.find(": ")
        if idx <= 0:
            continue
        speaker_id = line[:idx]
        text = line[idx + 2:]
        if speaker_id not in speakers:
            speakers[speaker_id] = "user" if len(speakers) == 0 else "assistant"
        role = speakers[speaker_id]
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": role, "content": text})
    return messages


def load_jsonl_test(path: str) -> list[dict]:
    """Load all records whose split == 'test' from a JSONL file."""
    path = _resolve_jsonl_path(path)
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            split = record.get("split") or record.get("metadata", {}).get("split")
            if split == "test":
                records.append(record)
    return records


# ---------------------------------------------------------------------------
# Save-path construction (stages 1 & 2)
# ---------------------------------------------------------------------------

def _build_save_paths(base_data_dir: str, model_name: str, dataset_folder: str, base_file_stem: str, n_shards: int, shard_idx: int):
    """Construct output dir and filename consistently across stages."""
    final_dir = os.path.join(base_data_dir, "embeddings", model_name, dataset_folder)
    os.makedirs(final_dir, exist_ok=True)
    if n_shards > 1:
        file_name = f"{base_file_stem}-{shard_idx:05d}-of-{n_shards:05d}.safetensors"
    else:
        file_name = f"{base_file_stem}.safetensors"
    return final_dir, os.path.join(final_dir, file_name)


# ---------------------------------------------------------------------------
# Inference model path resolution
# ---------------------------------------------------------------------------

def _resolve_inference_model_path(
    config: dict,
    cli_model_path: str | None,
    cli_model_parent_dir: str | None,
    cli_model_name: str | None,
) -> str:
    if cli_model_path:
        return cli_model_path

    inference_cfg = config.get("inference", {}) if isinstance(config, dict) else {}
    if not isinstance(inference_cfg, dict):
        inference_cfg = {}

    explicit_model_path = inference_cfg.get("model_path")
    if explicit_model_path:
        return str(explicit_model_path)

    if cli_model_parent_dir or cli_model_name:
        model_parent_dir = str(cli_model_parent_dir or inference_cfg.get("model_parent_dir", "model"))
        model_name = cli_model_name or inference_cfg.get("model_name")
        if not model_name:
            raise ValueError("model_name must be provided via --model_name or config.yaml inference.model_name")
        return os.path.join(model_parent_dir, str(model_name))

    model_name = inference_cfg.get("model_name")
    if not model_name:
        raise ValueError("model_name must be provided via --model_name or config.yaml inference.model_name")
    model_parent_dir = str(inference_cfg.get("model_parent_dir", "model"))
    return os.path.join(model_parent_dir, str(model_name))


# ---------------------------------------------------------------------------
# Token patterns and gating-position lookup
# ---------------------------------------------------------------------------

# Canonical mapping uses "llama3" (stage-2 convention); "llama" is an alias
# so that modeling_custom / stage-3 lookups also resolve correctly.
TOKEN_PATTERNS_BY_MODEL_TYPE = {
    # Llama3: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    "llama":  [128009, 128006, 78191, 128007, 271],
    # Gemma2: "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}


def find_token_for_gating(tokens: Sequence[int], model_type: Optional[str]) -> int:
    """Return the start index of the last model-specific token pattern.

    For Qwen3 (and any model_type without an explicit pattern), falls back to
    the last token position.
    """
    if model_type == "qwen3":
        return max(len(tokens) - 1, 0)

    token_pattern = TOKEN_PATTERNS_BY_MODEL_TYPE.get(model_type)
    if not token_pattern:
        return max(len(tokens) - 1, 0)

    token_pattern_len = len(token_pattern)
    search_end = len(tokens)
    for j in range(search_end - token_pattern_len, -1, -1):
        if list(tokens[j:j + token_pattern_len]) == token_pattern:
            return j
    # Fallback if exact marker pattern is not present in rendered prompt.
    return max(len(tokens) - 1, 0)


# ---------------------------------------------------------------------------
# Inference scoring helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_messages(model, tokenizer, messages, device, max_length):
    """Tokenize chat messages and run model forward pass."""
    encoding = tokenizer.apply_chat_template(
        messages, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    )
    if isinstance(encoding, torch.Tensor):
        input_ids = encoding.to(device)
        attention_mask = None
    else:
        # BatchEncoding or dict-like
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
    return model(input_ids=input_ids, attention_mask=attention_mask)
