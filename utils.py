# utils.py — Shared utility functions for the multidomain_model pipeline.

import json
import os
import importlib.util
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


def _attention_implementation(device: str) -> str | None:
    """Use FlashAttention on CUDA when installed; otherwise use Transformers defaults."""
    if str(device).startswith("cuda") and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return None


def _stable_int64_id(value) -> int:
    """Return a deterministic non-negative signed-int64 identifier."""
    import hashlib

    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) & ((1 << 63) - 1)


def debiasing_checkpoint_suffix(debiasing_dims, corr_threshold: float) -> str:
    """Encode reward-transform settings in a stable checkpoint suffix."""
    dims = sorted({int(dimension) for dimension in (debiasing_dims or ()) if int(dimension) >= 0})
    if not dims:
        return "_dbnone"
    threshold = format(float(corr_threshold), ".12g").replace("-", "m").replace(".", "p")
    dimension_text = "-".join(map(str, dims))
    return f"_db{dimension_text}_ct{threshold}"


def validate_shared_routing_config(routing_config) -> dict:
    """Validate the metadata contract required by packaged shared-gate checkpoints."""
    if not isinstance(routing_config, dict):
        raise ValueError("Stage 2 checkpoint is missing its training_config mapping.")
    if routing_config.get("format_version") != 2 or not routing_config.get("shared_prompt_gating", False):
        raise ValueError(
            "Stage 2 checkpoint must declare format_version=2 and "
            "shared_prompt_gating=true; legacy checkpoints are not packageable."
        )
    return routing_config


def shared_gate_checkpoint_filename(args, model_name: str, preference_name: str, reference_name: str) -> str:
    """Build the canonical Shared-Gate V2 checkpoint filename."""
    from attributes import attribute_selection_suffix

    defaults = {
        "learning_rate": 0.0005, "weight_decay": 0.0, "n_hidden": 1,
        "hidden_size": 64, "dropout": 0.1, "batch_size": 2048,
        "logit_scale": 2.0, "domain_loss_weight": 0.25,
        "entropy_weight": 0.02, "load_balance_weight": 0.05,
    }
    hyperparameters = "".join(
        f"_{key[:2]}{getattr(args, key, default)}"
        for key, default in defaults.items()
    )
    debiasing_dims = (
        [-1] if str(reference_name).lower() == "null"
        else getattr(args, "debiasing_dims", [-1])
    )
    suffix = debiasing_checkpoint_suffix(
        debiasing_dims, getattr(args, "corr_threshold", 0.04)
    )
    suffix += "_cv" if getattr(args, "curriculum", False) else ""
    suffix += "_bd" if getattr(args, "balance_difficulties", False) else ""
    suffix += "" if getattr(args, "balance_domains", True) else "_ubd"
    suffix += "_lgs" if getattr(args, "learnable_logit_scale", False) else ""
    entropy_floor = getattr(args, "entropy_floor_fraction", 0.35)
    suffix += "" if entropy_floor == 0.35 else f"_ef{entropy_floor}"
    suffix += attribute_selection_suffix(
        getattr(args, "attribute_subset", "full"),
        getattr(args, "exclude_attributes", []),
    )
    checkpoint_tag = getattr(args, "checkpoint_tag", None)
    suffix += f"_tag-{checkpoint_tag}" if checkpoint_tag else ""
    suffix += "_refit" if getattr(args, "train_on_all", False) else ""
    return (
        f"gating_network_sgv2_{model_name}_mo_{args.multi_objective_dataset_name}_"
        f"pref_{preference_name}_ref_{reference_name}"
        f"_t{getattr(args, 'temperature', 2.0):.1f}"
        f"_n{getattr(args, 'n_steps', 30000)}"
        f"_seed{getattr(args, 'seed', 0)}{hyperparameters}{suffix}.pt"
    )



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
    """Load all JSON/JSONL cultural test files from *data_dir* and return a flat list of records."""
    records: list[dict] = []
    if not os.path.isdir(data_dir):
        return records
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".jsonl"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list):
                records.extend(rows)
            else:
                records.append(rows)
    return records


def parse_cultural_conversation(record: dict) -> list[dict]:
    """Parse a cultural test record's conversation field into chat messages.

    Maps the first speaker to 'user', the second to 'assistant', and merges
    consecutive turns from the same speaker.
    """
    conv = record.get("conversation", "")
    lines = conv.split("\n")
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

    if cli_model_parent_dir or cli_model_name:
        model_parent_dir = str(cli_model_parent_dir or inference_cfg.get("model_parent_dir", "model"))
        model_name = cli_model_name or inference_cfg.get("model_name")
        if not model_name:
            raise ValueError("model_name must be provided via --model_name or config.yaml inference.model_name")
        return os.path.join(model_parent_dir, str(model_name))

    explicit_model_path = inference_cfg.get("model_path")
    if explicit_model_path:
        return str(explicit_model_path)

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
    # Mistral Instruct: "[/INST]" marks the start of the assistant response.
    "mistral": [733, 28748, 16289, 28793],
}


def find_token_for_gating(tokens: Sequence[int], model_type: Optional[str]) -> int:
    """Return the start index of the last model-specific token pattern.

    For Qwen3/auto (and any model_type without an explicit pattern), falls back
    to the last token position.
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

def _tokenize_chat(tokenizer, messages, device, max_length, *, add_generation_prompt=False):
    """Render then tokenize a chat consistently across preparation and inference."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )
    encoding = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoding.items()
    }


@torch.no_grad()
def _score_messages(model, tokenizer, messages, device, max_length, gating_output_override=None):
    """Tokenize chat messages and run one model forward pass."""
    if (
        gating_output_override is None
        and getattr(model.config, "shared_prompt_gating", False)
        and messages
        and messages[-1].get("role") == "assistant"
        and len(messages) > 1
    ):
        prompt_encoding = _tokenize_chat(
            tokenizer, messages[:-1], device, max_length,
            add_generation_prompt=True,
        )
        gating_output_override = model.compute_gating(
            input_ids=prompt_encoding["input_ids"],
            attention_mask=prompt_encoding.get("attention_mask"),
        )
    encoding = _tokenize_chat(tokenizer, messages, device, max_length)
    return model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding.get("attention_mask"),
        gating_output_override=gating_output_override,
    )


@torch.no_grad()
def _score_pair_shared_gate(
        model, tokenizer, prompt_messages, chosen_messages, rejected_messages,
        device, max_length,
):
    """Score a preference pair with one prompt-only gate shared by both candidates."""
    prompt_encoding = _tokenize_chat(
        tokenizer,
        prompt_messages,
        device,
        max_length,
        add_generation_prompt=True,
    )
    gating_output = model.compute_gating(
        input_ids=prompt_encoding["input_ids"],
        attention_mask=prompt_encoding.get("attention_mask"),
    )
    chosen = _score_messages(
        model, tokenizer, chosen_messages, device, max_length, gating_output,
    )
    rejected = _score_messages(
        model, tokenizer, rejected_messages, device, max_length, gating_output,
    )
    return chosen, rejected, gating_output
