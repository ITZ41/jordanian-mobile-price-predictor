from pathlib import Path
import yaml

_CONFIG = None
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        with open(CONFIG_PATH) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get(key: str, default=None):
    """Get nested config value via dot notation, e.g. get('cleaning.min_price_jd')"""
    cfg = load_config()
    parts = key.split(".")
    for p in parts:
        if isinstance(cfg, dict) and p in cfg:
            cfg = cfg[p]
        else:
            return default
    return cfg
