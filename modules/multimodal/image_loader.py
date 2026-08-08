"""
modules/multimodal/image_loader.py

Image file validation. Uses Pillow (already installed as a transitive
dependency of Streamlit itself -- no new package needed) to verify the
uploaded bytes are actually a valid, openable image, not just trusting
the file extension.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

from .config import SUPPORTED_EXTENSIONS, MultimodalConfig, mime_type_for_extension, multimodal_config

logger = get_logger(__name__)


class InvalidImageError(Exception):
    """Raised for an unsupported format, oversized file, empty upload,
    or a file that isn't actually a valid, readable image."""


@dataclass(frozen=True)
class ValidatedImage:
    extension: str
    mime_type: str
    width: int
    height: int
    format_name: str


def validate_image(
    filename: str, file_bytes: bytes, config: MultimodalConfig | None = None
) -> ValidatedImage:
    """Validate an uploaded image and return its metadata. Raises
    InvalidImageError with a friendly message on any problem --
    including a file with an image-like extension that isn't actually
    a valid image."""
    cfg = config or multimodal_config

    if not filename or "." not in filename:
        raise InvalidImageError("The uploaded file has no recognizable extension.")

    extension = "." + filename.rsplit(".", 1)[-1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file type '{extension}'. Supported formats: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}."
        )

    if not file_bytes:
        raise InvalidImageError(f"'{filename}' is empty (0 bytes).")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > cfg.max_file_size_mb:
        raise InvalidImageError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{cfg.max_file_size_mb} MB limit."
        )

    try:
        import io

        from PIL import Image, UnidentifiedImageError

        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()  # raises if the file isn't a genuine image

        # verify() invalidates the file handle for further reads, so
        # re-open to get dimensions/format.
        with Image.open(io.BytesIO(file_bytes)) as img:
            width, height = img.size
            format_name = img.format or extension.lstrip(".").upper()
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            f"'{filename}' does not appear to be a valid image file."
        ) from exc
    except ImportError as exc:
        raise InvalidImageError(
            "The 'Pillow' package is not installed. Run: pip install Pillow"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - any other Pillow failure -> friendly error
        logger.error("Failed to validate image '%s': %s", filename, exc)
        raise InvalidImageError(f"'{filename}' could not be read as an image.") from exc

    return ValidatedImage(
        extension=extension,
        mime_type=mime_type_for_extension(extension),
        width=width,
        height=height,
        format_name=format_name,
    )
