# main.py
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import os
import shutil
import json
import logging
import uuid
from datetime import timedelta
from typing import List, Optional
from PIL import Image
import numpy as np
from pydantic import BaseModel
from colorthief import ColorThief
from datetime import datetime
from vector_db import init_weaviate_schema, insert_embedding_to_weaviate
from schemas import Token, SearchResponse, UploadResponse, FileResult, ArtworkMetadata

from weaviate import Client
import numpy as np
from utils.embedding_validation import validate_embedding


# local imports
from database import get_db, init_db
from models import User, Artwork, Embedding
from ml_models import (
    generate_style_embedding,
    generate_texture_embedding,
    generate_palette_embedding,
    generate_emotion_embedding,
)
from auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from config import CORS_ORIGINS, MAX_FILE_SIZE, ALLOWED_IMAGE_TYPES
from schemas import Token, SearchResponse, UploadResponse, FileResult

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
weaviate_client = Client(WEAVIATE_URL)

# ───────────────────────────────
# App setup
# ───────────────────────────────
app = FastAPI(
    title="AI-Powered Art Search API",
    description="A multimodal similarity search engine for art and design styles",
    version="2.0.3",
)

app.mount("/images", StaticFiles(directory="images"), name="images")   # for uploads
app.mount("/data", StaticFiles(directory="data"), name="data")         # for datasets
app.mount("/wikiart", StaticFiles(directory="data/wikiart"), name="wikiart")  # for direct wikiart paths

# CORS setup
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
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger("api")

# auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.on_event("startup")
async def startup_event():
    if os.getenv("SKIP_DB_INIT", "false").lower() != "true":
        init_db()
    init_weaviate_schema()

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
    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(data={"sub": user.username, "role": user.role}, expires_delta=expires)
    return {"access_token": token, "token_type": "bearer"}

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

def calculate_cosine_similarity(v1, v2) -> float:
    v1 = l2_normalize(v1)
    v2 = l2_normalize(v2)
    return float(np.dot(v1, v2))

def extract_dominant_palette(path: str, count: int = 5):
    try:
        ct = ColorThief(path)
        return ct.get_palette(color_count=count)
    except Exception:
        return []



STYLE_LABELS = ["Impressionism", "Cubism", "Expressionism", "Renaissance", "Realism", "Modern"]
TEXTURE_LABELS = ["Smooth", "Rough", "Detailed", "Abstract"]
COLOR_LABELS = ["Warm", "Cool", "Neutral"]
EMOTION_LABELS = ["Calm", "Energetic", "Melancholic", "Joyful", "Mysterious"]

def auto_tags_from_embeddings(embs: dict):
    """
    Simple heuristic metadata generator based on vector averages.
    Replace later with ML model.
    """
    try:
        # ✅ Compatibility fix: allow either "palette" or "color"
        if "palette" not in embs and "color" in embs:
            embs["palette"] = embs["color"]

        # pick random tags based on embedding patterns (temporary)
        style = STYLE_LABELS[int(abs(np.mean(embs["style"])) * 10) % len(STYLE_LABELS)]
        texture = TEXTURE_LABELS[int(abs(np.mean(embs["texture"])) * 10) % len(TEXTURE_LABELS)]
        color = COLOR_LABELS[int(abs(np.mean(embs["palette"])) * 10) % len(COLOR_LABELS)]
        emotion = EMOTION_LABELS[int(abs(np.mean(embs["emotion"])) * 10) % len(EMOTION_LABELS)]

        return {
            "style": style,
            "texture": texture,
            "color": color,
            "emotion": emotion,
        }
    except Exception as e:
        print(f"⚠️ auto_tags_from_embeddings failed: {e}")
        return None



# ───────────────────────────────
# Embeddings
# ───────────────────────────────
EMBED_DIM = 512

def generate_all_embeddings(image_path: str):
    return {
        "style": normalize_vector_flexible(generate_style_embedding(image_path), EMBED_DIM),
        "texture": normalize_vector_flexible(generate_texture_embedding(image_path), EMBED_DIM),
        "palette": normalize_vector_flexible(generate_palette_embedding(image_path), EMBED_DIM),
        "emotion": normalize_vector_flexible(generate_emotion_embedding(image_path), EMBED_DIM),
    }

# ───────────────────────────────
# Upload
# ───────────────────────────────
async def _bulk_upload_impl(files: List[UploadFile], current_user: User, db: Session) -> UploadResponse:
    if not current_user or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload artworks")

    results, processed, failed = [], 0, 0

    for uf in files:
        try:
            
            # 1️⃣ Validate file
            if not validate_file_type(uf.content_type):
                failed += 1
                results.append(FileResult(filename=uf.filename, status="error", message="Invalid file type"))
                continue

            size = validate_file_size(uf)
            if size > MAX_FILE_SIZE:
                failed += 1
                results.append(FileResult(filename=uf.filename, status="error", message="File too large"))
                continue

            # 2️⃣ Save to /images
            safe_name = secure_filename(uf.filename)
            file_path = os.path.join("images", safe_name)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(uf.file, buffer)

            if not is_real_image(file_path):
                os.remove(file_path)
                failed += 1
                results.append(FileResult(filename=uf.filename, status="error", message="Invalid image"))
                continue

            # 3️⃣ Generate embeddings
            embs = generate_all_embeddings(file_path)
            tags = auto_tags_from_embeddings(embs)

            artwork = Artwork(
                filename=safe_name,
                filepath=file_path,
                style=tags.get("style"),
                color=tags.get("color"),
                texture=tags.get("texture"),
                emotion=tags.get("emotion"),
                metadata_json=tags,
                is_permanent=True
            )
            
            db.add(artwork)
            db.commit()         # ensure ID is generated
            db.refresh(artwork) # refresh to get primary key
            
            #Push embeddings into Weaviate
            insert_embedding_to_weaviate(artwork, embs, permanent=True)

            # Create Embedding
            emb_row = Embedding(
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
                emotion_vector=json.dumps(embs["emotion"].tolist()),
            )
            db.add(emb_row)
            db.commit()

            processed += 1
            results.append(
                FileResult(
                    id=artwork.id,
                    filename=uf.filename,
                    filepath=file_path,
                    status="success",
                    message="Uploaded & embedded successfully",
                )
            )

        except Exception as e:
            logger.exception(f"Error processing {uf.filename}: {e}")
            db.rollback()
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            failed += 1
            results.append(FileResult(filename=uf.filename, status="error", message=str(e)))

    return UploadResponse(inserted=processed, skipped=failed, files=results)


@app.post("/upload/bulk", response_model=UploadResponse)
async def bulk_upload_images(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await _bulk_upload_impl(files, current_user, db)

# ───────────────────────────────
# Search
# ───────────────────────────────
def _normalized_weights(style, texture, color, emotion, sw, tw, cw, ew):
    weights = {}
    total = 0.0
    for key, val, use in [
        ("style", sw, style),
        ("texture", tw, texture),
        ("color", cw, color),
        ("emotion", ew, emotion),
    ]:
        if use:
            weights[key] = max(0.0, val)
            total += weights[key]
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
    sw: float,
    tw: float,
    pw: float,
    ew: float,
    selected_color: Optional[str],
    current_user: User,
    db: Session,
) -> SearchResponse:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Validate size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    # Save temporary query image
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename or 'query.jpg'}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Normalize selected weights
        weights = _normalized_weights(style, texture, colorPalette, emotion, sw, tw, pw, ew)
        logger.info(f"Normalized weights: {weights}")

        # Generate embeddings for query image
        query_embs = {}
        if style:
            query_embs["style"] = normalize_vector_flexible(generate_style_embedding(temp_path))
        if texture:
            query_embs["texture"] = normalize_vector_flexible(generate_texture_embedding(temp_path))
        if colorPalette:
            query_embs["palette"] = normalize_vector_flexible(generate_palette_embedding(temp_path))
        if emotion:
            query_embs["emotion"] = normalize_vector_flexible(generate_emotion_embedding(temp_path))

        if not query_embs:
            raise HTTPException(status_code=400, detail="No filters selected")
        
        # Combine embeddings exactly like stored in Weaviate (concat order matters)
        ordered_keys = ["style", "texture", "color", "emotion"]
        parts = []
        for k in ordered_keys:
            if k in query_embs:
                parts.append(query_embs[k] * weights.get(k, 0))
            else:
                parts.append(np.zeros(512, dtype=np.float32))  # maintain dimension

        combined = np.concatenate(parts).astype(np.float32)
        logger.info(f"✅ Combined query vector shape: {combined.shape}")

        # 🧠 Query Weaviate directly
        WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
        client = Client(WEAVIATE_URL)

        res = (
            client.query
            .get(
                "ArtEmbedding",
                [
                    "filename",
                    "filepath",
                    "style",
                    "texture",
                    "color",
                    "emotion",
                    "_additional {certainty}"
                ],
            )
            .with_near_vector({"vector": combined.tolist()})
            .with_limit(20)
            .do()
        )

        # Handle possible null structures safely
        hits = (
            res.get("data", {})
               .get("Get", {})
               .get("ArtEmbedding", [])
        )

        if not hits:
            logger.warning(f"⚠️ No results returned from Weaviate: {json.dumps(res, indent=2)}")
            return SearchResponse(query_filename=file.filename, results=[])

        results = []  # ✅ initialize before loop
        base_url = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

        for h in hits:
            image_path = h.get("filepath") or h.get("filename")
            if not image_path.startswith("http"):
                if not image_path.startswith(("images/", "data/")):
                    image_path = f"data/wikiart/{os.path.basename(image_path)}"
                image_path = f"{base_url}/{image_path}"

            certainty = h.get("_additional", {}).get("certainty", 0)
            score = round(certainty * 100, 2)

            results.append(
                FileResult(
                    id=None,
                    filename=h.get("filename"),
                    filepath=image_path,
                    score=score,
                    message=f"Weaviate match — similarity {score:.2f}%",
                )
            )

        return SearchResponse(query_filename=file.filename, results=results)

    except Exception as e:
        logger.exception(f"Weaviate search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/search", response_model=SearchResponse)
async def search_similar_artworks(
    file: UploadFile = File(...),
    style: bool = Form(True),
    texture: bool = Form(True),
    color: bool = Form(True),
    emotion: bool = Form(True),
    style_weight: float = Form(25.0),
    texture_weight: float = Form(25.0),
    color_weight: float = Form(25.0),
    emotion_weight: float = Form(25.0),
    selected_color: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await _search_impl(
        file, style, texture, color, emotion,
        style_weight, texture_weight, color_weight, emotion_weight,
        selected_color, current_user, db
    )

@app.get("/artworks", response_model=List[ArtworkMetadata])
async def list_all_artworks(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns all uploaded artworks with their metadata.
    Accessible to authenticated users.
    """
    try:
        artworks = db.query(Artwork).order_by(Artwork.id.desc()).all()
        base_url = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

        for art in artworks:
            # Ensure the image path is fully resolvable
            if art.filepath and not art.filepath.startswith("http"):
                if not art.filepath.startswith(("images/", "data/")):
                    art.filepath = f"data/wikiart/{os.path.basename(art.filepath)}"
                art.filepath = f"{base_url}/{art.filepath}"

        return artworks

    except Exception as e:
        logger.exception(f"Error fetching artworks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ───────────────────────────────
# Palette helpers
# ───────────────────────────────
PALETTE = ["#FF5733", "#33FF57", "#3357FF", "#FFD700", "#FF69B4"]

class ColorChoice(BaseModel):
    color: str

@app.get("/palette")
async def get_palette(current_user: User = Depends(get_current_user)):
    return {"colors": PALETTE}

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
