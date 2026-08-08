"""
tests/test_multimodal.py

Unit tests for the Multimodal AI Assistant (Milestone 5). Uses small
real images generated with Pillow, and a temporary SQLite database
path per test. Vision LLM calls are mocked -- no network access
required.
"""

from __future__ import annotations

import dataclasses
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as root_config  # noqa: E402
from modules.multimodal.config import MultimodalConfig  # noqa: E402
from modules.multimodal.image_loader import InvalidImageError, validate_image  # noqa: E402
from modules.multimodal.manager import MultimodalManager  # noqa: E402
from utils import storage  # noqa: E402
from utils.llm_client import LLMConfigurationError, VisionNotSupportedError  # noqa: E402


def _make_image(fmt: str = "PNG", size=(64, 64), color=(255, 0, 0)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_config(tmp_path: Path, **overrides) -> MultimodalConfig:
    base = MultimodalConfig(uploads_dir=tmp_path / "uploads", max_file_size_mb=1)
    return dataclasses.replace(base, **overrides)


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Point the shared SQLite DB at a temp file for every test here."""
    test_db_path = tmp_path / "multimodal_test.db"
    patched_paths = dataclasses.replace(root_config.paths_config, sqlite_path=test_db_path)
    monkeypatch.setattr(root_config, "paths_config", patched_paths)
    monkeypatch.setattr(storage, "paths_config", patched_paths)
    storage.init_db()
    yield


# --- image_loader.py -----------------------------------------------------------


def test_validate_image_accepts_valid_png(tmp_path):
    cfg = _make_config(tmp_path)
    result = validate_image("photo.png", _make_image("PNG"), cfg)
    assert result.extension == ".png"
    assert result.mime_type == "image/png"
    assert result.width == 64
    assert result.height == 64


def test_validate_image_accepts_valid_jpg(tmp_path):
    cfg = _make_config(tmp_path)
    result = validate_image("photo.jpg", _make_image("JPEG"), cfg)
    assert result.extension == ".jpg"
    assert result.mime_type == "image/jpeg"


def test_validate_image_accepts_jpeg_extension(tmp_path):
    cfg = _make_config(tmp_path)
    result = validate_image("photo.jpeg", _make_image("JPEG"), cfg)
    assert result.mime_type == "image/jpeg"


def test_validate_image_rejects_unsupported_extension(tmp_path):
    cfg = _make_config(tmp_path)
    with pytest.raises(InvalidImageError):
        validate_image("document.pdf", b"fake bytes", cfg)


def test_validate_image_rejects_empty_file(tmp_path):
    cfg = _make_config(tmp_path)
    with pytest.raises(InvalidImageError):
        validate_image("photo.png", b"", cfg)


def test_validate_image_rejects_oversized_file(tmp_path):
    cfg = _make_config(tmp_path, max_file_size_mb=1)
    too_big = b"x" * (2 * 1024 * 1024)
    with pytest.raises(InvalidImageError):
        validate_image("photo.png", too_big, cfg)


def test_validate_image_rejects_fake_image_with_valid_extension(tmp_path):
    """A .png file that isn't actually a valid image (e.g. renamed
    text file) must be caught, not just trusted by extension."""
    cfg = _make_config(tmp_path)
    with pytest.raises(InvalidImageError):
        validate_image("fake.png", b"this is definitely not a real png file", cfg)


def test_validate_image_rejects_missing_extension(tmp_path):
    cfg = _make_config(tmp_path)
    with pytest.raises(InvalidImageError):
        validate_image("noextension", _make_image("PNG"), cfg)


# --- manager.py ----------------------------------------------------------------


def test_manager_validate_returns_image_metadata(tmp_path):
    cfg = _make_config(tmp_path)
    manager = MultimodalManager(cfg)
    result = manager.validate("photo.png", _make_image("PNG"))
    assert result.width == 64 and result.height == 64


def test_manager_analyze_rejects_empty_prompt(tmp_path):
    cfg = _make_config(tmp_path)
    manager = MultimodalManager(cfg)
    image_info = manager.validate("photo.png", _make_image("PNG"))
    result = manager.analyze("session-1", "photo.png", _make_image("PNG"), "   ", image_info)
    assert result.status == "error"
    assert "question" in result.message.lower()


def test_manager_analyze_success_saves_conversation(tmp_path):
    cfg = _make_config(tmp_path)
    manager = MultimodalManager(cfg)
    image_bytes = _make_image("PNG")
    image_info = manager.validate("photo.png", image_bytes)

    with patch("modules.multimodal.manager.analyze_image", return_value="This is a red square."):
        result = manager.analyze("session-1", "photo.png", image_bytes, "Describe this image.", image_info)

    assert result.status == "success"
    assert result.response == "This is a red square."

    history = manager.get_history("session-1")
    assert len(history) == 1
    assert history[0]["image_filename"] == "photo.png"
    assert history[0]["user_prompt"] == "Describe this image."
    assert history[0]["ai_response"] == "This is a red square."


def test_manager_analyze_handles_vision_not_supported(tmp_path):
    cfg = _make_config(tmp_path)
    manager = MultimodalManager(cfg)
    image_bytes = _make_image("PNG")
    image_info = manager.validate("photo.png", image_bytes)

    with patch(
        "modules.multimodal.manager.analyze_image",
        side_effect=VisionNotSupportedError("model does not support vision"),
    ):
        result = manager.analyze("session-1", "photo.png", image_bytes, "Describe this.", image_info)

    assert result.status == "error"
    assert "vision" in result.message.lower()
    assert manager.get_history("session-1") == []


def test_manager_analyze_handles_missing_api_key(tmp_path):
    cfg = _make_config(tmp_path)
    manager = MultimodalManager(cfg)
    image_bytes = _make_image("PNG")
    image_info = manager.validate("photo.png", image_bytes)

    with patch(
        "modules.multimodal.manager.analyze_image",
        side_effect=LLMConfigurationError("no key set"),
    ):
        result = manager.analyze("session-1", "photo.png", image_bytes, "Describe this.", image_info)

    assert result.status == "error"


def test_manager_analyze_never_crashes_on_unexpected_error(tmp_path):
    cfg = _make_config(tmp_path)
    manager = MultimodalManager(cfg)
    image_bytes = _make_image("PNG")
    image_info = manager.validate("photo.png", image_bytes)

    with patch(
        "modules.multimodal.manager.analyze_image",
        side_effect=RuntimeError("simulated unexpected failure"),
    ):
        result = manager.analyze("session-1", "photo.png", image_bytes, "Describe this.", image_info)

    assert result.status == "error"


def test_manager_get_history_empty_for_new_session(tmp_path):
    cfg = _make_config(tmp_path)
    manager = MultimodalManager(cfg)
    assert manager.get_history("brand-new-session") == []


def test_manager_history_isolated_per_session(tmp_path):
    cfg = _make_config(tmp_path)
    manager = MultimodalManager(cfg)
    image_bytes = _make_image("PNG")
    image_info = manager.validate("photo.png", image_bytes)

    with patch("modules.multimodal.manager.analyze_image", return_value="response A"):
        manager.analyze("session-A", "photo.png", image_bytes, "Question A?", image_info)
    with patch("modules.multimodal.manager.analyze_image", return_value="response B"):
        manager.analyze("session-B", "photo.png", image_bytes, "Question B?", image_info)

    assert len(manager.get_history("session-A")) == 1
    assert len(manager.get_history("session-B")) == 1
    assert manager.get_history("session-A")[0]["user_prompt"] == "Question A?"
