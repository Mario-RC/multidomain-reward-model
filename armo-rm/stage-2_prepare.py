# stage-2_prepare.py

import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser
from datetime import datetime

# Enable TF32 for faster matmul on supported GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Token patterns used to locate the gating position by model family.
token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}

print(f"Stage 2 Prepare started at {datetime.now().isoformat()}")

def find_token_for_gating(lst, model_family):
    """Return the start index of the last model-specific token pattern."""
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


# Parse command-line arguments.
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1", help="Path to the pre-trained model (HuggingFace path or local folder)")
parser.add_argument("--model_family", type=str, default="llama3", help="Model family (llama3 or gemma2)")
parser.add_argument("--dataset_path", type=str, default="RLHFlow/UltraFeedback-preference-standard", help="Path to the dataset (HuggingFace path or local folder)")
parser.add_argument("--source", default=None, type=str, help="Source filter for the dataset")
parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
parser.add_argument("--n_shards", type=int, default=1, help="Total number of shards to divide the dataset into")
parser.add_argument("--shard_idx", type=int, default=1, help="Index of the current shard")
parser.add_argument("--device", type=int, default=0, help="CUDA device index to use for computation")
parser.add_argument("--seq_len", type=int, default=8192, help="Maximum sequence length for input")
args = parser.parse_args()  # Parse CLI inputs.

# Validate model family against loaded model config.
config = AutoConfig.from_pretrained(args.model_path)
if args.model_family == "llama3":
    assert config.model_type == "llama"
elif args.model_family == "gemma2":
    assert config.model_type == "gemma2"
else:
    raise ValueError(f"Model family {args.model_family} is not supported")


# Resolve local output directory for generated embeddings.
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(script_dir, "data", "ArmoRM")

model_name = args.model_path.split("/")[-1]
dataset_name = args.dataset_path.split("/")[-1]
if args.source is not None:
    dataset_name += f"-{args.source}"
dataset_name += f"-{args.dataset_split}"

# Final save directory: .../embeddings/<model_name>/<dataset_name>
final_dir = os.path.join(BASE_DATA_DIR, "embeddings", model_name, dataset_name)


# Load dataset and apply optional filtering/sharding.
# Detect if it's a local JSON/JSONL file or a HuggingFace dataset

all_data = []
if args.dataset_path.endswith(".jsonl") or args.dataset_path.endswith(".json"):
    print(f"Manually loading local JSONL file: {args.dataset_path}")
    import json
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_data.append(json.loads(line.strip()))
            except Exception as e:
                continue
    # Create dataset from list (this handles inconsistent dictionaries much better)
    ds = datasets.Dataset.from_list(all_data)
else:
    # Standard loading for HuggingFace hub datasets
    ds = datasets.load_dataset(args.dataset_path, split=args.dataset_split)
if args.source is not None:
    ds = ds.filter(lambda x: x["source"] == args.source)
if args.n_shards > 1:
    ds = ds.shuffle(seed=0)
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)

# Load encoder model and tokenizer.
device = f"cuda:{args.device}"
model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # bf16 reduces memory footprint on supported GPUs.
    device_map=device,
    attn_implementation="flash_attention_2",  # Use FlashAttention v2 when available.
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Accumulate pair embeddings and prompt embeddings.
embeddings = []
prompt_embeddings = []

# Process each preference pair.
for example in tqdm(ds, desc="Examples"):
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
        if args.model_path.endswith("FsfairX-LLaMA3-RM-v0.1"):
            # Match the model card formatting (remove BOS token for this checkpoint).
            conv_formatted = tokenizer.apply_chat_template(
                iter_example, tokenize=False, add_generation_prompt=False
            ).replace(tokenizer.bos_token, "")
        else:
            conv_formatted = tokenizer.apply_chat_template(iter_example, tokenize=False)

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

# Ensure output directory exists.
os.makedirs(final_dir, exist_ok=True)

# Build file name with optional shard suffix.
file_name = (
    f"{dataset_name}-{args.shard_idx:05d}-of-{args.n_shards:05d}.safetensors"
    if args.n_shards > 1
    else f"{dataset_name}.safetensors"
)

save_path_full = os.path.join(final_dir, file_name)

# Save embeddings using `safetensors`.
save_file(
    {"embeddings": embeddings, "prompt_embeddings": prompt_embeddings},
    save_path_full,
)

# Log output path.
print(f"Saved embeddings to {save_path_full}")