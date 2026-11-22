import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIGS = sorted((PROJECT_DIR / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_module_entrypoint_checks_project_configs(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aeris", "config", "check", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert path.name in result.stdout


def test_config_check_returns_error_for_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    result = subprocess.run(
        [sys.executable, "-m", "aeris", "config", "check", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert path.name in result.stderr
