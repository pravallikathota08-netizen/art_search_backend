# vector_db.py
import os
import logging
import numpy as np
from weaviate import Client
from uuid import uuid4
import time

logger = logging.getLogger("weaviate")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")

def get_weaviate_client(retries: int = 15, delay: int = 5) -> Client:
    """
    Initialize and return a Weaviate client with retry logic.
    Waits until Weaviate is ready before returning.
    """
    for attempt in range(1, retries + 1):
        try:
            client = Client(WEAVIATE_URL)
            if client.is_ready():
                logger.info(f"✅ Connected to Weaviate at {WEAVIATE_URL}")
                return client
            else:
                logger.warning(f"⚠️ Weaviate not ready (attempt {attempt}/{retries})")
        except Exception as e:
            logger.warning(f"⏳ Waiting for Weaviate (attempt {attempt}/{retries}): {e}")
        time.sleep(delay)

    # if we exit loop → failed to connect
    logger.error(f"❌ Could not connect to Weaviate after {retries} attempts.")
    raise RuntimeError("Weaviate is not reachable.")

# Global client instance
client = get_weaviate_client()

# ───────────────────────────────────────────────
# Create schema class (called once at startup)
# ───────────────────────────────────────────────
def init_weaviate_schema():
    """Define schema for artwork embeddings if it doesn't already exist."""
    class_name = "ArtEmbedding"

    if not client.schema.exists(class_name):
        schema = {
            "class": class_name,
            "description": "Stores artwork embeddings and metadata for search",
            "vectorizer": "none",
# We'll send custom vectors
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
        logger.info("✅ Weaviate schema 'ArtEmbedding' created successfully.")
    else:
        logger.info("ℹ️ Weaviate schema 'ArtEmbedding' already exists.")

# ───────────────────────────────────────────────
# Insert embeddings (used during upload)
# ───────────────────────────────────────────────
def insert_embedding_to_weaviate(artwork, embs: dict):
    """
    Push embeddings and metadata for an artwork into Weaviate.
    artwork: SQLAlchemy Artwork object
    embs: dict of numpy arrays (style, texture, palette, emotion)
    """
    try:
        # Combine all embeddings into one long vector
        vector = np.concatenate([
            embs["style"], embs["texture"], embs["palette"], embs["emotion"]
        ])

        obj = {
            "filename": artwork.filename,
            "filepath": artwork.filepath,
            "style": artwork.style,
            "texture": artwork.texture,
            "palette": artwork.color,  
            "emotion": artwork.emotion,
        }

        client.data_object.create(
            data_object=obj,
            class_name="ArtEmbedding",
            vector=vector.tolist(),
            uuid=str(uuid4())
        )

        logger.info(f"Inserted '{artwork.filename}' into Weaviate successfully.")
    except Exception as e:
        logger.error(f"Error inserting '{artwork.filename}' into Weaviate: {e}")
