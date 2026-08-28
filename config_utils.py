import os
import sys
from typing import Any, Dict

import yaml


def cli_has_flag(flag: str, argv=None) -> bool:
    args = argv if argv is not None else sys.argv[1:]
    negative_flag = f"--no-{flag[2:]}" if flag.startswith("--") else f"no-{flag}"
    return any(
        arg == flag or arg.startswith(f"{flag}=")
        or arg == negative_flag or arg.startswith(f"{negative_flag}=")
        for arg in args
    )


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def apply_section_overrides(args, section_cfg: Dict[str, Any], argv=None, skip_keys=None):
    """Apply YAML overrides for keys not explicitly passed on the CLI.

    Priority: CLI flag > YAML value > argparse default.
    """
    if not section_cfg:
        return args
    skip = set(skip_keys or [])
    for key, value in section_cfg.items():
        if key in skip:
            continue
        if not hasattr(args, key):
            continue
        if value is None:
            continue
        if not cli_has_flag(f"--{key}", argv=argv):
            setattr(args, key, value)
    return args


def apply_model_registry(args, config: Dict[str, Any], argv=None):
    """Resolve --model_key from config.yaml:model_registry.

    Explicit CLI values keep highest priority. A selected registry entry
    overrides stage-section defaults for model-specific fields.
    """
    model_key = getattr(args, "model_key", None)
    if not model_key:
        return args
    registry = config.get("model_registry", {}) if isinstance(config, dict) else {}
    if not isinstance(registry, dict) or model_key not in registry:
        available = sorted(registry) if isinstance(registry, dict) else []
        available_text = ", ".join(available) if available else "none"
        raise ValueError(
            f"Unknown model_key {model_key!r}; available keys: {available_text}."
        )
    entry = registry[model_key]
    if not isinstance(entry, dict):
        raise ValueError(f"model_registry.{model_key} must be a mapping.")
    for field in ("model_path", "model_family", "output_model_name"):
        if field in entry and hasattr(args, field) and not cli_has_flag(f"--{field}", argv=argv):
            setattr(args, field, entry[field])
    return args
