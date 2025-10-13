# generate_and_sync_embeddings.py
import os
import sys
import json
import logging
import sqlite3
from uuid import uuid4
from typing import Optional
import numpy as np
from tqdm import tqdm

from weaviate import Client

# Import your existing embedding functions from main.py
from main import (
    generate_style_embedding,
    generate_texture_embedding,
    generate_palette_embedding as generate_color_embedding,
    generate_emotion_embedding,
)

# Logging setup
logger = logging.getLogger("sync")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

CLASS_NAME = "ArtEmbedding"

def get_db_conn(db_path: str):
    """Connect to the SQLite artworks database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_schema(client: Client):
    """Ensure the ArtEmbedding schema exists in Weaviate."""
    if not client.schema.exists(CLASS_NAME):
        schema = {
            "class": CLASS_NAME,
            "description": "Stores artwork embeddings and metadata for search",
            "vectorizer": "none",
            "properties": [
                {"name": "filename", "dataType": ["text"]},
                {"name": "filepath", "dataType": ["text"]},
                {"name": "style", "dataType": ["text"]},
                {"name": "texture", "dataType": ["text"]},
                {"name": "palette", "dataType": ["text"]},
                {"name": "emotion", "dataType": ["text"]},
            ],
        }
        client.schema.create_class(schema)
        logger.info("✅ Created Weaviate class ArtEmbedding")
    else:
        logger.info("ℹ️ Weaviate class ArtEmbedding already exists")

def build_vector_from_image(image_path: str) -> Optional[np.ndarray]:
    """Generate and concatenate all embeddings for one image."""
    try:
        s = generate_style_embedding(image_path)
        t = generate_texture_embedding(image_path)
        c = generate_color_embedding(image_path)     # “color” → stored as “palette”
        e = generate_emotion_embedding(image_path)

        if not all([s is not None, t is not None, c is not None, e is not None]):
            return None

        parts = [np.array(v, dtype=np.float32).flatten() for v in (s, t, c, e)]
        return np.concatenate(parts)
    except Exception as ex:
        logger.error(f"Embedding failed for {image_path}: {ex}")
        return None

def sync_embeddings(db_path: str, images_root: str):
    """Generate and push all embeddings from SQLite into Weaviate."""
    client = Client(os.getenv("WEAVIATE_URL", "http://weaviate:8080"))

    if not client.is_ready():
        raise RuntimeError("❌ Weaviate is not ready.")

    ensure_schema(client)

    conn = get_db_conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT filename, filepath, style, color, texture, emotion FROM artworks")
    rows = cur.fetchall()

    logger.info(f"📦 Found {len(rows)} artworks in {db_path}")
    batch = []
    pbar = tqdm(rows, desc="Syncing to Weaviate", unit="img")

    for r in pbar:
        filename = r["filename"]
        filepath = r["filepath"]
        style = r["style"]
        texture = r["texture"]
        color = r["color"]
        emotion = r["emotion"]

        # Build full image path correctly based on DB structure
        image_path = os.path.join("/app", filepath)

        if not os.path.exists(image_path):
            logger.warning(f"⚠️ Missing image file: {image_path}")
            continue

        vec = build_vector_from_image(image_path)
        if vec is None:
            continue

        obj = {
            "filename": filename,
            "filepath": filepath,
            "style": style,
            "texture": texture,
            "palette": color,
            "emotion": emotion,
        }

        batch.append({"object": obj, "vector": vec})
        if len(batch) >= 100:
            with client.batch(batch_size=100) as b:
                for item in batch:
                    b.add_data_object(
                        data_object=item["object"],
                        class_name=CLASS_NAME,
                        uuid=str(uuid4()),
                        vector=item["vector"].tolist(),
                    )
            batch = []

    if batch:
        with client.batch(batch_size=100) as b:
            for item in batch:
                b.add_data_object(
                    data_object=item["object"],
                    class_name=CLASS_NAME,
                    uuid=str(uuid4()),
                    vector=item["vector"].tolist(),
                )

    logger.info("✅ Sync complete. All embeddings pushed to Weaviate.")

if __name__ == "__main__":
    DB_PATH = os.getenv("DB_PATH", "/app/data/artworks.db")
    IMAGES_ROOT = os.getenv("IMAGES_ROOT", "/app/data/wikiart")  # ✅ keep this since images are in /wikiart
    sync_embeddings(DB_PATH, IMAGES_ROOT)