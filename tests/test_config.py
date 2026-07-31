"""
tests/test_config.py

Sanity checks for config.py: defaults load without a .env file, and
directory bootstrapping works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import app_config, ensure_directories, llm_config, paths_config  # noqa: E402


def test_app_config_has_sensible_defaults():
    assert app_config.app_title
    assert app_config.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_llm_config_defaults_present():
    assert llm_config.model
    assert llm_config.base_url.startswith("http")
    assert llm_config.temperature >= 0
    assert llm_config.max_tokens > 0


def test_ensure_directories_creates_expected_folders():
    ensure_directories()
    for path in (
        paths_config.data_dir,
        paths_config.database_dir,
        paths_config.uploads_dir,
        paths_config.vector_store_dir,
        paths_config.assets_dir,
        paths_config.docs_dir,
    ):
        assert path.exists() and path.is_dir()
