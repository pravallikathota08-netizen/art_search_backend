from typing import List, Optional
from pydantic import BaseModel
import pydantic  # <-- you forgot to import this earlier

# Detect if Pydantic v2 is being used
IS_V2 = hasattr(pydantic.BaseModel, "model_config")

# ───────────────────────────────
# Individual file result
# ───────────────────────────────
class FileResult(BaseModel):
    id: Optional[int] = None
    filename: str
    filepath: Optional[str] = None
    score: Optional[float] = None
    message: Optional[str] = None

    if IS_V2:
        model_config = {"from_attributes": True}
    else:
        class Config:
            orm_mode = True


# ───────────────────────────────
# Bulk upload response
# ───────────────────────────────
class UploadResponse(BaseModel):
    inserted: int
    skipped: int
    files: List[FileResult]


# ───────────────────────────────
# Search response
# ───────────────────────────────
class SearchResponse(BaseModel):
    query_filename: str
    results: List[FileResult]


# ───────────────────────────────
# Artwork metadata (for /artworks endpoint)
# ───────────────────────────────
class ArtworkMetadata(BaseModel):
    id: int
    filename: str
    filepath: str
    style: Optional[str] = None
    color: Optional[str] = None
    texture: Optional[str] = None
    emotion: Optional[str] = None

    if IS_V2:
        model_config = {"from_attributes": True}
    else:
        class Config:
            orm_mode = True


# ───────────────────────────────
# Auth token
# ───────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
