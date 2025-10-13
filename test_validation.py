"""
Unit tests for image validation and file size checks.
Covers:
- validate_file_type()
- validate_file_size()
- is_real_image()
"""

import os
import pytest
from PIL import Image
from utils.validators import validate_file_type, validate_file_size, is_real_image



MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit (same as in main.py)


# -----------------------------
# 1️⃣ Test: File type validation
# -----------------------------
@pytest.mark.parametrize("mime,expected", [
    ("image/jpeg", True),
    ("image/png", True),
    ("image/heic", True),
    ("text/plain", False),
    ("application/pdf", False),
])
def test_validate_file_type(mime, expected):
    """Ensure only valid MIME types are accepted."""
    assert validate_file_type(mime) == expected


# -----------------------------
# 2️⃣ Test: File size validation
# -----------------------------
def test_validate_file_size(tmp_path):
    """Verify small and large files are validated correctly."""
    # Small dummy file (1 KB)
    small_file = tmp_path / "small.jpg"
    small_file.write_bytes(os.urandom(1024))
    with open(small_file, "rb") as f:
        size = validate_file_size(f)
    assert size < MAX_FILE_SIZE

    # Large dummy file (>10 MB)
    large_file = tmp_path / "large.jpg"
    large_file.write_bytes(os.urandom(MAX_FILE_SIZE + 1024))
    with open(large_file, "rb") as f:
        size = validate_file_size(f)
    assert size > MAX_FILE_SIZE


# -----------------------------
# 3️⃣ Test: Image validity check
# -----------------------------
def test_is_real_image_valid(tmp_path):
    """Confirm a real image passes validation."""
    img_path = tmp_path / "valid_image.jpg"
    Image.new("RGB", (10, 10), color="blue").save(img_path)
    assert is_real_image(str(img_path)) is True


def test_is_real_image_invalid(tmp_path):
    """Reject a text file pretending to be an image."""
    fake_img = tmp_path / "fake.jpg"
    fake_img.write_text("this is not an image")
    assert is_real_image(str(fake_img)) is False


def test_is_real_image_corrupt(tmp_path):
    """Reject empty or corrupted files."""
    empty_img = tmp_path / "empty.jpg"
    empty_img.write_bytes(b"")
    assert is_real_image(str(empty_img)) is False
