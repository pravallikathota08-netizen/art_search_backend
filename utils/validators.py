import os
from PIL import Image

# Same rules as main.py
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/heic"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_file_type(content_type: str) -> bool:
    """Return True if content type is an allowed image type."""
    return (content_type or "").lower() in ALLOWED_IMAGE_TYPES


def validate_file_size(upload_file) -> int:
    """Return the file size in bytes."""
    upload_file.seek(0, os.SEEK_END)
    size = upload_file.tell()
    upload_file.seek(0)
    return size


def is_real_image(path: str) -> bool:
    """Validate whether a file is a readable image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
