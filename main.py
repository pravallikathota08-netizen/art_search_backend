from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker
import uvicorn
import os
from datetime import timedelta
from typing import List, Optional
import jwt
from passlib.context import CryptContext
import shutil
from pydantic import BaseModel
from colorthief import ColorThief
import sqlite3
import sqlite_vec
import json
import numpy as np

# local imports
from database import get_db, init_db
from models import User, Artwork
from ml_models import (
    generate_style_embedding,
    generate_texture_embedding,
    generate_palette_embedding,
    generate_emotion_embedding
)
from auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from schemas import Token, SearchResponse, UploadResponse
from config import CORS_ORIGINS, MAX_FILE_SIZE, ALLOWED_IMAGE_TYPES

# ───────────────────────────────
# App setup
# ───────────────────────────────
app = FastAPI(
    title="AI-Powered Art Search API",
    description="A multimodal similarity search engine for art and design styles",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# directories
os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")

# auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.on_event("startup")
async def startup_event():
    init_db()

# ───────────────────────────────
# Auth
# ───────────────────────────────
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},  # include role
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ───────────────────────────────
# Upload
# ───────────────────────────────
@app.post("/upload/bulk/", response_model=UploadResponse)
async def bulk_upload_images(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    # Only admins can bulk upload
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload artworks")

    processed_count = 0
    for file in files:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            continue

        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        if file_size > MAX_FILE_SIZE:
            continue

        file_path = f"images/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            style_embedding = generate_style_embedding(file_path)
            texture_embedding = generate_texture_embedding(file_path)
            palette_embedding = generate_palette_embedding(file_path)
            emotion_embedding = generate_emotion_embedding(file_path)

            artwork = Artwork(
                filename=file.filename,
                image_path=file_path,
                style_embedding=json.dumps(style_embedding),
                texture_embedding=json.dumps(texture_embedding),
                palette_embedding=json.dumps(palette_embedding),
                emotion_embedding=json.dumps(emotion_embedding),
            )
            db.add(artwork)
            db.flush()

            insert_vectors_to_db(
                artwork.id,
                style_embedding,
                texture_embedding,
                palette_embedding,
                emotion_embedding
            )

            processed_count += 1
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue

    db.commit()
    return UploadResponse(message=f"Uploaded and processed {processed_count} images.")

# ───────────────────────────────
# Utils
# ───────────────────────────────
def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "on")

WEIGHTS = {
    "style": 0.2,
    "texture": 0.15,
    "palette": 0.6,
    "emotion": 0.05,
}

def insert_vectors_to_db(artwork_id, style_embedding, texture_embedding, palette_embedding, emotion_embedding):
    try:
        conn = sqlite3.connect("artworks.db")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("""
            INSERT OR REPLACE INTO artworks_vectors 
            (rowid, style_embedding, texture_embedding, palette_embedding, emotion_embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (artwork_id, style_embedding, texture_embedding, palette_embedding, emotion_embedding))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error inserting vectors: {e}")
        return False

def calculate_cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    dot = np.dot(v1, v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# ───────────────────────────────
# Search (respects filters)
# ───────────────────────────────
@app.post("/search/", response_model=List[SearchResponse])
async def search_similar_artworks(
    file: UploadFile = File(...),
    style: bool = Form(True),
    texture: bool = Form(True),
    colorPalette: bool = Form(True),
    emotion: bool = Form(True),
    selected_color: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    style = str_to_bool(style)
    texture = str_to_bool(texture)
    colorPalette = str_to_bool(colorPalette)
    emotion = str_to_bool(emotion)

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image type")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Only generate embeddings for enabled filters
        query_embeddings = {}
        if style:
            query_embeddings["style"] = generate_style_embedding(temp_path)
        if texture:
            query_embeddings["texture"] = generate_texture_embedding(temp_path)
        if colorPalette:
            query_embeddings["palette"] = generate_palette_embedding(temp_path)
        if emotion:
            query_embeddings["emotion"] = generate_emotion_embedding(temp_path)

        artworks = db.query(Artwork).all()
        similarities = []

        for artwork in artworks:
            total_score = 0.0
            total_weight = 0.0
            reasons = []

            if style and artwork.style_embedding:
                stored = json.loads(artwork.style_embedding)
                score = calculate_cosine_similarity(query_embeddings["style"], stored)
                total_score += score * WEIGHTS["style"]
                total_weight += WEIGHTS["style"]
                reasons.append(f"Style {score:.2f}")

            if texture and artwork.texture_embedding:
                stored = json.loads(artwork.texture_embedding)
                score = calculate_cosine_similarity(query_embeddings["texture"], stored)
                total_score += score * WEIGHTS["texture"]
                total_weight += WEIGHTS["texture"]
                reasons.append(f"Texture {score:.2f}")

            if colorPalette and artwork.palette_embedding:
                stored = json.loads(artwork.palette_embedding)
                score = calculate_cosine_similarity(query_embeddings["palette"], stored)

                if selected_color and selected_color.lower() not in json.dumps(stored).lower():
                    score *= 0.7

                total_score += score * WEIGHTS["palette"]
                total_weight += WEIGHTS["palette"]
                reasons.append(f"Palette {score:.2f}")

            if emotion and artwork.emotion_embedding:
                stored = json.loads(artwork.emotion_embedding)
                score = calculate_cosine_similarity(query_embeddings["emotion"], stored)
                total_score += score * WEIGHTS["emotion"]
                total_weight += WEIGHTS["emotion"]
                reasons.append(f"Emotion {score:.2f}")

            final_score = total_score / total_weight if total_weight > 0 else 0
            similarities.append((artwork, final_score, reasons))

        # Sort by score
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 🔥 Apply a minimum similarity threshold
        threshold = 0.5
        filtered_results = [(art, score, reasons) for art, score, reasons in similarities if score >= threshold]

        if not filtered_results:
            return []  # return nothing if no good matches

        top_results = filtered_results[:10]

        return [
            SearchResponse(
                imageUrl=f"http://127.0.0.1:8000/images/{art.filename}",
                tags=["style", "texture", "palette", "emotion"],
                matchReason=", ".join(reasons) if reasons else "No filters applied",
                similarity=score,
            )
            for art, score, reasons in top_results
        ]

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ───────────────────────────────
# Palette
# ───────────────────────────────
PALETTE = ["#FF5733", "#33FF57", "#3357FF", "#FFD700", "#FF69B4"]

class ColorChoice(BaseModel):
    color: str

@app.get("/palette")
async def get_palette(current_user: User = Depends(get_current_user)):
    return {"colors": PALETTE}

@app.post("/palette/select")
async def select_color(choice: ColorChoice, current_user: User = Depends(get_current_user)):
    if choice.color not in PALETTE:
        raise HTTPException(status_code=400, detail="Invalid color")
    return {"message": f"Color {choice.color} selected", "color": choice.color}

@app.post("/palette/extract")
async def extract_palette(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    temp_path = f"temp_palette_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        ct = ColorThief(temp_path)
        colors = ct.get_palette(color_count=5)
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        return {"colors": hex_colors}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ───────────────────────────────
# Health
# ───────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Art Search API running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
