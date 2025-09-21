from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import uvicorn
import os
from datetime import datetime, timedelta
from typing import List, Optional
import jwt
from passlib.context import CryptContext
import shutil
from pathlib import Path

from database import get_db, init_db
from models import User, Artwork
from ml_models import generate_style_embedding, generate_texture_embedding, generate_palette_embedding, generate_emotion_embedding
from auth import authenticate_user, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
from schemas import Token, UserResponse, ArtworkResponse, SearchResponse, UploadResponse
import json
from config import CORS_ORIGINS, MAX_FILE_SIZE, ALLOWED_IMAGE_TYPES
import sqlite3
import sqlite_vec

# Create FastAPI app
app = FastAPI(
    title="AI-Powered Art Search API",
    description="A multimodal similarity search engine for art and design styles",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create images directory
os.makedirs("images", exist_ok=True)

# Mount static files
app.mount("/images", StaticFiles(directory="images"), name="images")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# Authentication endpoints
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
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Bulk upload endpoint
@app.post("/upload/bulk/", response_model=UploadResponse)
async def bulk_upload_images(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    processed_count = 0
    
    for file in files:
        # Validate file type and size
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            continue
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            continue
            
        # Save file
        file_path = f"images/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate embeddings
        try:
            style_embedding = generate_style_embedding(file_path)
            texture_embedding = generate_texture_embedding(file_path)
            palette_embedding = generate_palette_embedding(file_path)
            emotion_embedding = generate_emotion_embedding(file_path)
            
            # Store in database (convert lists to JSON strings)
            artwork = Artwork(
                filename=file.filename,
                image_path=file_path,
                style_embedding=json.dumps(style_embedding),
                texture_embedding=json.dumps(texture_embedding),
                palette_embedding=json.dumps(palette_embedding),
                emotion_embedding=json.dumps(emotion_embedding)
            )
            db.add(artwork)
            db.flush()  # Get the ID without committing
            
            # Insert vectors into vector table for efficient search
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
    return UploadResponse(message=f"Successfully uploaded and processed {processed_count} images.")

# Similarity search endpoint
@app.post("/search/", response_model=List[SearchResponse])
async def search_similar_artworks(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    # Validate file type and size
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="File must be a valid image type")
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size too large")
    
    # Save query image temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Generate embeddings for query image
        query_style_embedding = generate_style_embedding(temp_path)
        
        # Try to use sqlite-vec for efficient similarity search
        try:
            conn = sqlite3.connect("artworks.db")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            
            # Use sqlite-vec for similarity search
            cursor = conn.execute("""
                SELECT 
                    a.id, a.filename, a.image_path,
                    vec_distance_cosine(style_embedding, ?) as distance
                FROM artworks_vectors v
                JOIN artworks a ON a.id = v.rowid
                ORDER BY distance
                LIMIT 10
            """, (query_style_embedding,))
            
            results = cursor.fetchall()
            conn.close()
            
            # Format results
            top_results = []
            for row in results:
                artwork_id, filename, image_path, distance = row
                similarity = 1 - distance  # Convert distance to similarity
                artwork = Artwork(id=artwork_id, filename=filename, image_path=image_path)
                top_results.append((artwork, similarity))
                
        except Exception as e:
            print(f"sqlite-vec search failed, using fallback: {e}")
            # Fallback to manual similarity calculation
            artworks = db.query(Artwork).all()
            similarities = []
            for artwork in artworks:
                stored_embedding = json.loads(artwork.style_embedding)
                similarity = calculate_cosine_similarity(query_style_embedding, stored_embedding)
                similarities.append((artwork, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:10]
        
        # Format results
        results = []
        for artwork, similarity in top_results:
            results.append(SearchResponse(
                imageUrl=f"/images/{artwork.filename}",
                tags=["art", "style", "design"],  # Placeholder tags
                matchReason=f"Similar style with {similarity:.2f} similarity score",
                similarity=similarity
            ))
        
        return results
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Helper function to insert vectors into vector table
def insert_vectors_to_db(artwork_id, style_embedding, texture_embedding, palette_embedding, emotion_embedding):
    """Insert vectors into the sqlite-vec vector table for efficient similarity search"""
    try:
        conn = sqlite3.connect("artworks.db")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        # Insert vectors into the vector table
        conn.execute("""
            INSERT OR REPLACE INTO artworks_vectors 
            (rowid, style_embedding, texture_embedding, palette_embedding, emotion_embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (
            artwork_id,
            style_embedding,
            texture_embedding,
            palette_embedding,
            emotion_embedding
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error inserting vectors: {e}")
        return False

# Helper function for cosine similarity (fallback)
def calculate_cosine_similarity(vec1, vec2):
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Art Search API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
