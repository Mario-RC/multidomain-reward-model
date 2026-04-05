import os
import sys
from typing import Any, Dict

import yaml


def cli_has_flag(flag: str, argv=None) -> bool:
    args = argv if argv is not None else sys.argv[1:]
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


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
