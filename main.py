# main.py
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import os
from datetime import timedelta
from typing import List, Optional, Dict, Tuple
from schemas import Token, SearchResponse, UploadResponse, FileResult

import shutil
import json
import logging
import uuid
from PIL import Image
import numpy as np
from pydantic import BaseModel
from colorthief import ColorThief

# local imports
from database import get_db, init_db
from models import User, Artwork, Embedding
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
from config import CORS_ORIGINS, MAX_FILE_SIZE, ALLOWED_IMAGE_TYPES

# ───────────────────────────────
# App setup
# ───────────────────────────────
app = FastAPI(
    title="AI-Powered Art Search API",
    description="A multimodal similarity search engine for art and design styles",
    version="2.0.1"
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
os.makedirs("logs", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()]
)
logger = logging.getLogger("api")

# auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ───────────────────────────────
# Helpers
# ───────────────────────────────
def secure_filename(original_name: str) -> str:
    name, ext = os.path.splitext(original_name or "")
    ext = (ext or ".jpg").lower()
    return f"{uuid.uuid4().hex}{ext}"

def validate_file_type(content_type: str) -> bool:
    return (content_type or "").lower() in ALLOWED_IMAGE_TYPES

def validate_file_size(upload_file: UploadFile) -> int:
    upload_file.file.seek(0, 2)
    size = upload_file.file.tell()
    upload_file.file.seek(0)
    return size

def is_real_image(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-10)

def normalize_vector_flexible(vec: np.ndarray, target_dim: int = 512) -> np.ndarray:
    """Pad/trim to target_dim then L2 normalize."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.shape[0] < target_dim:
        v = np.pad(v, (0, target_dim - v.shape[0]))
    elif v.shape[0] > target_dim:
        v = v[:target_dim]
    return l2_normalize(v)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # expects normalized inputs
    return float(np.dot(a, b))

def rgb_from_hex(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b

def rgb_distance(c1: Tuple[int,int,int], c2: Tuple[int,int,int]) -> float:
    return float(np.linalg.norm(np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32)))

def extract_dominant_palette(path: str, count: int = 5) -> List[Tuple[int, int, int]]:
    try:
        ct = ColorThief(path)
        return ct.get_palette(color_count=count)
    except Exception:
        return []

# Simple auto-tag rules you can replace later with classifiers
def auto_tags_from_embeddings(embs: Dict[str, np.ndarray]) -> Dict[str, str]:
    style_label   = "Abstract" if float(np.mean(embs["style"]))   >= 0 else "Realistic"
    color_label   = "Vibrant"  if float(np.mean(embs["palette"])) >= 0 else "Muted"
    texture_label = "Smooth"   if float(np.mean(embs["texture"])) >= 0 else "Rough"
    emotion_label = "Happy"    if float(np.mean(embs["emotion"])) >= 0 else "Calm"
    return {
        "style": style_label,
        "color": color_label,
        "texture": texture_label,
        "emotion": emotion_label,
    }

# ───────────────────────────────
# Embeddings
# ───────────────────────────────
EMBED_DIM = 512  # expected dimension for storage

def generate_all_embeddings(image_path: str) -> dict:
    embs = {
        "style":   normalize_vector_flexible(generate_style_embedding(image_path),   EMBED_DIM),
        "texture": normalize_vector_flexible(generate_texture_embedding(image_path), EMBED_DIM),
        "palette": normalize_vector_flexible(generate_palette_embedding(image_path), EMBED_DIM),
        "emotion": normalize_vector_flexible(generate_emotion_embedding(image_path), EMBED_DIM),
    }
    return embs

def calculate_cosine_similarity(vec1, vec2):
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.dot(v1, v2))

# ───────────────────────────────
# Upload (supports two routes)
# ───────────────────────────────
async def _bulk_upload_impl(
    files: List[UploadFile],
    current_user: User,
    db: Session
) -> UploadResponse:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload artworks")

    results = []
    processed = failed = 0

    for uf in files:
        try:
            # ──────────────── validate file ────────────────
            if not validate_file_type(uf.content_type):
                failed += 1
                results.append(FileResult(filename=uf.filename, status="error", message="Invalid file type"))
                continue

            size = validate_file_size(uf)
            if size > MAX_FILE_SIZE:
                failed += 1
                results.append(FileResult(filename=uf.filename, status="error", message=f"File too large ({size} bytes)"))
                continue

            # ──────────────── save locally ────────────────
            safe_name = secure_filename(uf.filename)
            file_path = os.path.join("images", safe_name)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(uf.file, buffer)

            if not is_real_image(file_path):
                os.remove(file_path)
                failed += 1
                results.append(FileResult(filename=uf.filename, status="error", message="File is not a valid image"))
                continue

            # ──────────────── embeddings ────────────────
            embs = generate_all_embeddings(file_path)
            tags = auto_tags_from_embeddings(embs)

            # ──────────────── save artwork ────────────────
            artwork = Artwork(
                filename=safe_name,
                filepath=file_path,
                style=tags.get("style"),
                color=tags.get("color"),
                texture=tags.get("texture"),
                emotion=tags.get("emotion"),
            )

            db.add(artwork)
            db.flush()  # ensures artwork.id available

            # ──────────────── save all embeddings in ONE row ────────────────
            db.add(Embedding(
                artwork_id=artwork.id,
                vector=json.dumps({
                    "style": embs["style"].tolist(),
                    "texture": embs["texture"].tolist(),
                    "palette": embs["palette"].tolist(),
                    "emotion": embs["emotion"].tolist()
                }),
                style_vector=json.dumps(embs["style"].tolist()),
                color_vector=json.dumps(embs["palette"].tolist()),
                texture_vector=json.dumps(embs["texture"].tolist()),
                emotion_vector=json.dumps(embs["emotion"].tolist())
            ))

            db.commit()
            processed += 1

            results.append(FileResult(
                id=artwork.id,
                filename=uf.filename,
                filepath=file_path,
                status="success",
                message="Uploaded & embedded (style, texture, palette, emotion)"
            ))

        except Exception as e:
            logger.exception(f"Error processing {uf.filename}: {e}")
            db.rollback()
            try:
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            failed += 1
            results.append(FileResult(filename=uf.filename, status="error", message=str(e)))

    return UploadResponse(
        inserted=processed,
        skipped=failed,
        files=results
    )

# Original route your frontend hit earlier
@app.post("/upload/bulk/", response_model=UploadResponse)
async def bulk_upload_images(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await _bulk_upload_impl(files, current_user, db)

# Alias route used by your newer frontend
@app.post("/upload_bulk", response_model=UploadResponse)
async def bulk_upload_images_alias(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await _bulk_upload_impl(files, current_user, db)

# ───────────────────────────────
# Search (supports two routes)
# ───────────────────────────────
PALETTE = ["#FF5733", "#33FF57", "#3357FF", "#FFD700", "#FF69B4"]  # sample palette for UI

def _normalized_weights(style, texture, palette, emotion,
                        style_weight, texture_weight, palette_weight, emotion_weight):
    weights = {}
    total = 0.0

    if style:
        weights["style"] = max(0.0, style_weight)
        total += weights["style"]
    if texture:
        weights["texture"] = max(0.0, texture_weight)
        total += weights["texture"]
    if palette:
        weights["palette"] = max(0.0, palette_weight)
        total += weights["palette"]
    if emotion:
        weights["emotion"] = max(0.0, emotion_weight)
        total += weights["emotion"]

    # Normalize
    if total > 0:
        for k in weights:
            weights[k] /= total
    return weights

async def _search_impl(
    file: UploadFile,
    style: bool,
    texture: bool,
    colorPalette: bool,
    emotion: bool,
    style_weight: float,
    texture_weight: float,
    palette_weight: float,
    emotion_weight: float,
    selected_color: Optional[str],
    current_user: User,
    db: Session
) -> SearchResponse:
    import json, numpy as np, os, shutil, uuid

    # ✅ Validate file type & size
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image type")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename or 'query.jpg'}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # ✅ Step 1: Normalize and log active weights
        weights = _normalized_weights(
            style, texture, colorPalette, emotion,
            style_weight, texture_weight, palette_weight, emotion_weight
        )
        logger.info(f"✅ Final normalized weights: {weights}")

        # ✅ Step 2: Generate query embeddings dynamically
        query_embs = {}
        if style:
            query_embs["style"] = normalize_vector_flexible(generate_style_embedding(temp_path), 512)
        if texture:
            query_embs["texture"] = normalize_vector_flexible(generate_texture_embedding(temp_path), 512)
        if colorPalette:
            query_embs["palette"] = normalize_vector_flexible(generate_palette_embedding(temp_path), 512)
        if emotion:
            query_embs["emotion"] = normalize_vector_flexible(generate_emotion_embedding(temp_path), 512)

        if not query_embs:
            raise HTTPException(status_code=400, detail="No filters selected")

        logger.info(f"🧩 Generated query embeddings for: {list(query_embs.keys())}")

        # ✅ Step 3: Iterate artworks & calculate weighted similarity
        results = []
        artworks = db.query(Artwork).all()

        for art in artworks:
            if not art.embedding:
                continue

            emb = art.embedding
            individual_scores = {}
            reasons = []

            # Map DB columns to embedding types
            column_map = {
                "style": emb.style_vector,
                "texture": emb.texture_vector,
                "palette": emb.color_vector,
                "emotion": emb.emotion_vector,
            }

            for model_name, qvec in query_embs.items():
                vec_json = column_map.get(model_name)
                if not vec_json:
                    continue

                try:
                    stored_vec = np.array(json.loads(vec_json), dtype=np.float32)
                    if stored_vec.size == 0:
                        continue
                except Exception as ve:
                    logger.warning(f"⚠️ Error loading vector for {model_name} of {art.id}: {ve}")
                    continue

                # Calculate similarity
                score = calculate_cosine_similarity(qvec, stored_vec)
                individual_scores[model_name] = score
                reasons.append(f"{model_name.capitalize()} {score:.2f} (×{int(weights.get(model_name, 0)*100)}%)")

            if not individual_scores:
                continue

            # Weighted final score
            weighted_sum = sum(individual_scores[m] * weights.get(m, 0.0) for m in individual_scores)
            total_weight = sum(weights.get(m, 0.0) for m in individual_scores)
            final_score = round((weighted_sum / total_weight) * 100.0, 2) if total_weight > 0 else 0.0

            results.append((art.id, final_score, "; ".join(reasons)))

        logger.info(f"🧠 Computed {len(results)} raw results before sorting")

        # ✅ Step 4: Sort top 10 results
        top = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        base_url = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

        # ✅ Step 5: Convert to FileResult-compatible dicts (matching schema)
        final_results = []
        for aid, score, reason in top:
            artwork = db.query(Artwork).get(aid)
            if not artwork:
                continue

            final_results.append(FileResult(
                id=aid,
                filename=artwork.filename,
                filepath=f"{base_url}/images/{artwork.filename}",
                score=score,
                message=reason
            ))

        return SearchResponse(
            query_filename=file.filename,
            results=final_results
        )


    except Exception as e:
        logger.exception(f"💥 Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Original route (with trailing slash)
@app.post("/search/", response_model=SearchResponse)
async def search_similar_artworks(
    file: UploadFile = File(...),
    style: bool = Form(True),
    texture: bool = Form(True),
    colorPalette: bool = Form(True),
    emotion: bool = Form(True),
    style_weight: float = Form(25.0),
    texture_weight: float = Form(25.0),
    palette_weight: float = Form(25.0),
    emotion_weight: float = Form(25.0),
    selected_color: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await _search_impl(
        file, style, texture, colorPalette, emotion,
        style_weight, texture_weight, palette_weight, emotion_weight,
        selected_color, current_user, db
    )

# Alias route (no trailing slash)
@app.post("/search", response_model=SearchResponse)
async def search_similar_artworks_alias(
    file: UploadFile = File(...),
    style: bool = Form(True),
    texture: bool = Form(True),
    colorPalette: bool = Form(True),
    emotion: bool = Form(True),
    style_weight: float = Form(25.0),
    texture_weight: float = Form(25.0),
    palette_weight: float = Form(25.0),
    emotion_weight: float = Form(25.0),
    selected_color: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await _search_impl(
        file, style, texture, colorPalette, emotion,
        style_weight, texture_weight, palette_weight, emotion_weight,
        selected_color, current_user, db
    )

# ───────────────────────────────
# Palette helpers
# ───────────────────────────────
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
    temp_path = f"temp_palette_{uuid.uuid4().hex}_{file.filename}"
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

