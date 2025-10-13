
import numpy as np
from fastapi import HTTPException

def validate_embedding(vec, expected_dim: int = 512, name: str = "embedding") -> np.ndarray:
    """
    Validate embedding type, dimension, and numeric integrity.
    Raises HTTPException(400) for invalid vectors.
    """
    if vec is None:
        raise HTTPException(status_code=400, detail=f"{name} is None")

    if not isinstance(vec, np.ndarray):
        try:
            vec = np.array(vec, dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=400, detail=f"{name} could not be converted to np.ndarray")

    vec = vec.ravel()  # flatten safely

    if vec.shape[0] != expected_dim:
        raise HTTPException(status_code=400, detail=f"{name} dimension mismatch — expected {expected_dim}, got {vec.shape[0]}")

    if not np.isfinite(vec).all():
        raise HTTPException(status_code=400, detail=f"{name} contains NaN or infinite values")

    return vec.astype(np.float32)
