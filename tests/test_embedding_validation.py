# tests/test_embedding_validation.py
import pytest
import numpy as np
from fastapi import HTTPException
from utils.embedding_validation import validate_embedding

def test_valid_embedding():
    """✅ Should pass for correct 512-d float32 vector"""
    vec = np.random.rand(512).astype(np.float32)
    out = validate_embedding(vec)
    assert out.shape == (512,)
    assert np.isfinite(out).all()

def test_wrong_dimension():
    """❌ Should raise HTTPException for wrong dimension"""
    vec = np.random.rand(128).astype(np.float32)
    with pytest.raises(HTTPException) as exc:
        validate_embedding(vec)
    assert "embedding dimension mismatch" in exc.value.detail

def test_nan_values():
    """❌ Should raise HTTPException if vector contains NaN"""
    vec = np.full((512,), np.nan, dtype=np.float32)
    with pytest.raises(HTTPException) as exc:
        validate_embedding(vec)
    assert "embedding contains NaN" in exc.value.detail


def test_non_array_input():
    """❌ Should raise if input is not convertible to numpy array"""
    with pytest.raises(HTTPException):
        validate_embedding("not_an_array")

def test_none_input():
    """❌ Should raise if embedding is None"""
    with pytest.raises(HTTPException):
        validate_embedding(None)
