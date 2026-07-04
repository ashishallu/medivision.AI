"""
config_loader.py
Loads config/config.yaml once and exposes it as a plain dict.
Every other python module imports `load_config()` from here instead of
re-parsing the YAML file itself.
"""

import os
import yaml

# Project root = two levels up from this file (python/utils/ -> project root)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")

_cached_config = None


def load_config() -> dict:
    """Load and cache config/config.yaml. Returns a dict."""
    global _cached_config
    if _cached_config is None:
        if not os.path.exists(_CONFIG_PATH):
            raise FileNotFoundError(
                f"Could not find config file at {_CONFIG_PATH}. "
                "Run this project from within the MediVision-AI folder."
            )
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _cached_config = yaml.safe_load(f)
    return _cached_config


def resolve_path(relative_path: str) -> str:
    """Resolve any project-relative path (as written in config.yaml) to an
    absolute path, regardless of the current working directory."""
    return os.path.join(_PROJECT_ROOT, relative_path)
