from typing import List, Optional
from pydantic import BaseModel

# ───────────────────────────────
# Individual file result
# ───────────────────────────────
class FileResult(BaseModel):
    id: Optional[int] = None          
    filename: str
    filepath: Optional[str] = None    
    score: Optional[float] = None         
    message: Optional[str] = None     

    class Config:
        from_attributes = True

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
# Auth token
# ───────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
