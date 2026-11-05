"""配置文件读取（JSON，默认 ``configs/default.json``）。"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.json"


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """加载默认配置，并按需用用户配置做深度覆盖。"""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            config = _merge(config, json.load(handle))
    return config
