from pydantic import BaseModel
from typing import List, Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    username: str
    
    class Config:
        from_attributes = True

class ArtworkResponse(BaseModel):
    id: int
    filename: str
    image_path: str
    
    class Config:
        from_attributes = True

class SearchResponse(BaseModel):
    imageUrl: str
    tags: List[str]
    matchReason: str
    similarity: Optional[float] = None

class UploadResponse(BaseModel):
    message: str
