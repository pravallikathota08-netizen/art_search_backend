import os
import time
import uuid
import logging
import numpy as np
import json
from weaviate import Client
import traceback
from weaviate.exceptions import UnexpectedStatusCodeException, ObjectAlreadyExistsError

# ───────────────────────────────
# Setup logging and connection
# ───────────────────────────────
logger = logging.getLogger("weaviate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")


def get_weaviate_client(retries: int = 15, delay: int = 5) -> Client:
    """Try to connect to Weaviate with retries."""
    for attempt in range(1, retries + 1):
        try:
            client = Client(WEAVIATE_URL)
            if hasattr(client, "is_ready") and client.is_ready():
                logger.info(f"✅ Connected to Weaviate at {WEAVIATE_URL}")
                return client
            else:
                client.schema.get()  # fallback check
                logger.info(f"✅ Connected to Weaviate at {WEAVIATE_URL}")
                return client
        except Exception as e:
            logger.warning(f"⏳ Waiting for Weaviate (attempt {attempt}/{retries}): {e}")
            time.sleep(delay)

    logger.error(f"❌ Could not connect to Weaviate after {retries} attempts.")
    raise RuntimeError("Weaviate is not reachable.")


# Global client instance
client = get_weaviate_client()


# ───────────────────────────────
# Schema definition
# ───────────────────────────────
def init_weaviate_schema():
    """Define schema for artwork embeddings if it doesn't already exist."""
    class_name = "ArtEmbedding"

    if not client.schema.exists(class_name):
        schema = {
            "class": class_name,
            "description": "Stores artwork embeddings and metadata for image search",
            "vectorizer": "none",
            "properties": [
                {"name": "filename", "dataType": ["text"]},
                {"name": "filepath", "dataType": ["text"]},
                {"name": "style", "dataType": ["text"]},
                {"name": "color", "dataType": ["text"]},
                {"name": "texture", "dataType": ["text"]},
                {"name": "emotion", "dataType": ["text"]},
                {"name": "metadata_json", "dataType": ["text"]},
                {"name": "is_permanent", "dataType": ["boolean"]},
            ],
        }
        client.schema.create_class(schema)
        logger.info("✅ Weaviate schema 'ArtEmbedding' created successfully.")
    else:
        logger.info("ℹ️ Weaviate schema 'ArtEmbedding' already exists.")


# ───────────────────────────────
# Helper functions
# ───────────────────────────────
def _combine_vector(embs: dict) -> np.ndarray:
    """Combine the 4 embeddings into a single vector."""
    # Compatibility fix — map palette → color automatically
    if "palette" in embs and "color" not in embs:
        embs["color"] = embs["palette"]

    return np.concatenate([
        embs["style"],
        embs["texture"],
        embs["color"],
        embs["emotion"],
    ]).astype(np.float32)


# ───────────────────────────────
# Main insert/update logic
# ───────────────────────────────
def insert_embedding_to_weaviate(artwork, embs: dict, object_id: str):
    """
    Atomically replace or create an artwork embedding + metadata in Weaviate.
    Ensures all fields (including color) are fully updated each run.
    """
    from weaviate.exceptions import WeaviateBaseError
    import numpy as np
    import json

    try:
        # Compatibility fix for palette vs color
        if "palette" in embs and "color" not in embs:
            embs["color"] = embs["palette"]

        # Combine all sub-embeddings into one vector
        full_vector = np.concatenate([
            embs["style"],
            embs["texture"],
            embs["color"],
            embs["emotion"]
        ]).astype(np.float32).tolist()

        # Prepare metadata payload
        data = {
            "filename": artwork.filename,
            "filepath": artwork.filepath,
            "style": artwork.style or "Unknown",
            "texture": artwork.texture or "Unknown",
            "color": artwork.color or "Unknown",
            "emotion": artwork.emotion or "Unknown",
            "metadata_json": json.dumps({
                "style": artwork.style or "Unknown",
                "color": artwork.color or "Unknown",
                "texture": artwork.texture or "Unknown",
                "emotion": artwork.emotion or "Unknown"
            }),
            "is_permanent": True
        }

        try:
            # Try replacing (update if exists)
            client.data_object.replace(
                uuid=object_id,
                class_name="ArtEmbedding",
                data_object=data,
                vector=full_vector
            )
            logger.info(f"♻️ Replaced existing record in Weaviate: {artwork.filename}")

        except WeaviateBaseError as e:
            # If object not found, create instead
            if "no object with id" in str(e):
                client.data_object.create(
                    class_name="ArtEmbedding",
                    uuid=object_id,
                    data_object=data,
                    vector=full_vector
                )
                logger.info(f"✨ Created new record in Weaviate: {artwork.filename}")
            else:
                logger.error(f"❌ Weaviate API error for {artwork.filename}: {e}")

    except Exception as e:
        logger.error(f"❌ Unexpected error for {artwork.filename}: {e}")
