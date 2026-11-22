from pathlib import Path

import pytest

from aeris.config import ConfigError, load_config


def test_load_config_preserves_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  name: xbd\n", encoding="utf-8")

    assert load_config(path) == {"dataset": {"name": "xbd"}}


@pytest.mark.parametrize("contents", ["", "- xbd", "1: xbd", "dataset: ["])
def test_load_config_rejects_invalid_documents(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match="config.yaml"):
        load_config(path)
