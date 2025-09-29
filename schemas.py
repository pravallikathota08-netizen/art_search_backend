from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# ───────────────────────────────
# Auth
# ───────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str

# ───────────────────────────────
# User
# ───────────────────────────────
class UserResponse(BaseModel):
    username: str
    
    class Config:
        from_attributes = True

# ───────────────────────────────
# Artwork
# ───────────────────────────────
class ArtworkResponse(BaseModel):
    id: int
    filename: str
    image_path: str
    
    class Config:
        from_attributes = True

# ───────────────────────────────
# Search
# ───────────────────────────────
class SearchResponse(BaseModel):
    imageUrl: str
    tags: List[str]
    matchReason: str
    similarity: Optional[float] = None

class UploadResponse(BaseModel):
    message: str


class SearchLogCreate(BaseModel):
    user_id: int
    query_image: str
    selected_filters: str
    results: str

class SearchLogResponse(SearchLogCreate):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True