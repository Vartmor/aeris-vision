from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded."""


def load_config(path: Path) -> dict[str, object]:
    """Read a YAML configuration with string keys at its root."""
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: {error}") from error

    if not isinstance(config, dict) or any(
        not isinstance(key, str) for key in config
    ):
        raise ConfigError(f"{path}: kökte metin anahtarları olan bir YAML eşlemesi bekleniyor.")

    return config
