# stage-3_package_model.py

import os
import sys
import torch
from argparse import ArgumentParser
from datetime import datetime
from transformers import AutoConfig, AutoTokenizer
from modeling_custom import RewardModelWithGating
from config_utils import load_yaml_config
from attributes import ATTRIBUTES
from utils import _requires_remote_code

def _safe_torch_load(path: str):
    # Prefer safe weights-only loading when the installed PyTorch supports it.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_state_dict(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    raise TypeError(f"Unsupported checkpoint payload type: {type(obj)}")


def _extract_stage1_weight_tensor(obj) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if "weight" in obj and isinstance(obj["weight"], torch.Tensor):
            return obj["weight"]
        if "regression_layer.weight" in obj and isinstance(obj["regression_layer.weight"], torch.Tensor):
            return obj["regression_layer.weight"]
    raise TypeError("Stage 1 checkpoint must contain a tensor under 'weight' or 'regression_layer.weight'.")


def _build_defaults_from_config(config: dict, model_path: str, args=None):
    model_name = model_path.split("/")[-1]
    stage2_cfg = config.get("stage_2_train", {}) if isinstance(config, dict) else {}
    stage3_cfg = config.get("stage_3_package", {}) if isinstance(config, dict) else {}

    # Resolve dataset names: CLI args > stage_3_package config > stage_2_train config.
    multi_objective_dataset_name = (
        (getattr(args, "multi_objective_dataset_name", None) if args else None)
        or stage3_cfg.get("multi_objective_dataset_name")
        or str(stage2_cfg.get("multi_objective_dataset_name", "stage_1"))
    )
    preference_base = (
        (getattr(args, "preference_dataset_name", None) if args else None)
        or stage3_cfg.get("preference_dataset_name")
        or stage2_cfg.get("preference_dataset_name")
    )
    _ref_cli = (getattr(args, "reference_dataset_name", None) if args else None)
    if _ref_cli and _ref_cli.lower() == "null":
        # CLI explicitly said "null" → no reference dataset was used during training.
        reference_base = "null"
    elif _ref_cli:
        reference_base = _ref_cli
    else:
        _ref_cfg = stage3_cfg.get("reference_dataset_name")
        if _ref_cfg is None or str(_ref_cfg).lower() == "null":
            reference_base = "null"
        else:
            reference_base = _ref_cfg

    stage1_weights_path = os.path.join(
        "model", "regression_weights", f"{model_name}_{multi_objective_dataset_name}_100pct.pt"
    )
    stage2_weights_path = os.path.join(
        "model",
        "gating_network",
        (
            f"gating_network_{model_name}_mo_{multi_objective_dataset_name}_"
            f"pref_{preference_base}_ref_{reference_base}"
            f"_t{getattr(args, 'temperature', 10.0):.1f}_n{getattr(args, 'n_steps', 2000)}_seed{getattr(args, 'seed', 0)}"
            + "".join(
                f"_{k[:2]}{getattr(args, k, v)}" for k, v in
                {"learning_rate": 0.001, "weight_decay": 0.0, "n_hidden": 3, "hidden_size": 1024, "dropout": 0.2, "batch_size": 1024, "corr_threshold": 0.03, "logit_scale": 1.0}.items()
            )
            + ("_cv" if getattr(args, "curriculum", False) else "")
            + ".pt"
        ),
    )
    model_parent_dir = str(stage3_cfg.get("model_parent_dir", stage3_cfg.get("output_parent_dir", "model")))
    final_model_name = (
        stage3_cfg.get("output_model_name")
        or stage3_cfg.get("final_model_name")
        or f"multi-domain-rm-{model_name.lower()}"
    )
    output_dir = os.path.join(model_parent_dir, final_model_name)
    return stage1_weights_path, stage2_weights_path, output_dir


def main() -> None:
    parser = ArgumentParser(description="Stage 3: package final reward model.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_key", type=str, default=None, help="Model key defined in config.yaml:model:registry.")
    parser.add_argument("--model_path", type=str, default=None, help="Base model HF ID/path.")
    parser.add_argument("--stage_1_weights_path", type=str, default=None, help="Optional override for Stage 1 regression weights path.")
    parser.add_argument("--stage_2_weights_path", type=str, default=None, help="Optional override for Stage 2 gating network weights path.")
    parser.add_argument("--model_parent_dir", type=str, default="model", help="Optional output parent directory (e.g., model).")
    parser.add_argument("--multi_objective_dataset_name", type=str, default=None, help="Multi-objective dataset name from stage-1 (used to locate regression weights and gating network).")
    parser.add_argument("--preference_dataset_name", type=str, default=None, help="Preference dataset name (without split suffix, e.g., Multi-Domain-Data-Preference-Pairs).")
    parser.add_argument("--reference_dataset_name", type=str, default=None, help="Reference dataset name (without split suffix, e.g., UltraFeedback-preference-standard).")
    parser.add_argument("--output_model_name", type=str, default=None, help="Optional packaged model directory name.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for final packaged model output directory.")
    parser.add_argument("--model_family", type=str, default=None, help="Model family (llama3, gemma2, qwen3, auto).")
    parser.add_argument("--temperature", type=float, default=10.0, help="Temperature used in stage-2 training (for locating checkpoint).")
    parser.add_argument("--n_steps", type=int, default=2000, help="Number of steps used in stage-2 training (for locating checkpoint).")
    parser.add_argument("--seed", type=int, default=0, help="Seed used in stage-2 training (for locating checkpoint).")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate used in stage-2 (for locating checkpoint).")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay used in stage-2 (for locating checkpoint).")
    parser.add_argument("--n_hidden", type=int, default=3, help="Hidden layers used in stage-2 (for locating checkpoint).")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size used in stage-2 (for locating checkpoint).")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout used in stage-2 (for locating checkpoint).")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size used in stage-2 (for locating checkpoint).")
    parser.add_argument("--corr_threshold", type=float, default=0.03, help="Corr threshold used in stage-2 (for locating checkpoint).")
    parser.add_argument("--logit_scale", type=float, default=1.0, help="Logit scale used in stage-2 (for locating checkpoint).")
    parser.add_argument("--curriculum", action="store_true", default=False, help="Include _curriculum suffix when locating stage-2 checkpoint.")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    stage3_cfg = config.get("stage_3_package", {}) or {}
    if not args.model_path:
        args.model_path = stage3_cfg.get("model_path")

    inferred_stage1, inferred_stage2, inferred_output = _build_defaults_from_config(config, args.model_path, args)
    if args.stage_1_weights_path and os.path.sep not in args.stage_1_weights_path:
        stage_1_weights_path = os.path.join("model", "regression_weights", args.stage_1_weights_path)
    else:
        stage_1_weights_path = args.stage_1_weights_path or inferred_stage1
    stage_2_weights_path = args.stage_2_weights_path or inferred_stage2
    inferred_parent = os.path.dirname(inferred_output)
    inferred_name = os.path.basename(inferred_output)
    model_parent_dir = args.model_parent_dir or inferred_parent
    output_model_name = args.output_model_name or inferred_name
    if not output_model_name:
        print("FATAL ERROR: --output_model_name is required (set stage_3_package.output_model_name in config.yaml or pass --output_model_name).")
        sys.exit(1)
    output_dir = args.output_dir or os.path.join(model_parent_dir, output_model_name)

    print(f"\n### Stage 3: Package model started at {datetime.now().isoformat()} ###")
    print(f"  Base model:          {args.model_path}")
    print(f"  Stage 1 weights:     {stage_1_weights_path}")
    print(f"  Stage 2 weights:     {stage_2_weights_path}")
    print(f"  Output directory:    {output_dir}")

    print("Loading configuration and tokenizer...")
    trust_remote_code = _requires_remote_code(args.model_path)
    if trust_remote_code:
        print("Using trust_remote_code=True for Qwen3 model loading compatibility.")

    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=trust_remote_code)
    model_config.num_objectives = len(ATTRIBUTES)
    model_config.gating_hidden_dim = args.hidden_size
    model_config.gating_n_hidden = args.n_hidden
    model_config.gating_temperature = args.temperature
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=trust_remote_code)

    print("Instantiating custom architecture with base model weights...")
    model = RewardModelWithGating.from_pretrained(
        args.model_path,
        config=model_config,
        ignore_mismatched_sizes=True,
        trust_remote_code=trust_remote_code,
    )

    print(f"Loading Stage 1 regression weights from: {stage_1_weights_path}")
    stage1_payload = _safe_torch_load(stage_1_weights_path)
    stage1_weights = _extract_stage1_weight_tensor(stage1_payload)
    if tuple(stage1_weights.shape) != tuple(model.regression_layer.weight.shape):
        raise ValueError(
            f"Stage 1 tensor shape {tuple(stage1_weights.shape)} does not match "
            f"regression layer shape {tuple(model.regression_layer.weight.shape)}"
        )
    stage1_weights = stage1_weights.to(model.regression_layer.weight.dtype)
    model.regression_layer.weight.data.copy_(stage1_weights)

    print(f"Loading Stage 2 gating network weights from: {stage_2_weights_path}")
    stage2_payload = _safe_torch_load(stage_2_weights_path)
    stage2_state_dict = _resolve_state_dict(stage2_payload)
    if not isinstance(stage2_state_dict, dict):
        raise TypeError("Stage 2 checkpoint must resolve to a state_dict dictionary.")
    model.gating.load_state_dict(stage2_state_dict)

    # Load reward transform matrix from checkpoint if available; fall back to
    # identity for backward compatibility with older checkpoints.
    if isinstance(stage2_payload, dict) and "reward_transform_matrix" in stage2_payload:
        rtm = stage2_payload["reward_transform_matrix"].to(model.reward_transform_matrix.dtype)
        model.reward_transform_matrix.data.copy_(rtm)
        print(f"Loaded reward_transform_matrix from checkpoint.")
    else:
        with torch.no_grad():
            eye = torch.zeros_like(model.reward_transform_matrix)
            diag = torch.arange(model.num_objectives, device=eye.device)
            eye[diag, diag] = 1.0
            model.reward_transform_matrix.data.copy_(eye)
        print("No reward_transform_matrix in checkpoint; using identity (legacy checkpoint).")

    print(f"Saving finalized model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Multidomain reward model packaged at: {output_dir}")


if __name__ == "__main__":
    main()