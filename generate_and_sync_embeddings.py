import os, sys
sys.path.append(os.path.dirname(__file__))

import os
import json
import logging
import sqlite3
import glob
import numpy as np
from tqdm import tqdm
from weaviate import Client
import gc
from PIL import Image
import uuid

from ml_models import (
    generate_style_embedding,
    generate_texture_embedding,
    generate_palette_embedding as generate_color_embedding,
    generate_emotion_embedding,
)
from main import auto_tags_from_embeddings
from vector_db import insert_embedding_to_weaviate
from database import SessionLocal


# ─────────────────────────────────────────────────────────────
# Setup logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("sync")


def get_db_conn(db_path: str):
    """Connect to the SQLite artworks database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Safe wrapper so embedding generation never fails
def safe_embedding(generator, path):
    try:
        v = generator(path)
        return np.array(v, dtype=np.float32)
    except Exception as e:
        logger.warning(f"⚠️ Dummy vector for {os.path.basename(path)}: {e}")
        return np.random.randn(512).astype(np.float32)

# ─────────────────────────────────────────────────────────────
# Rebuild embeddings + metadata
# ─────────────────────────────────────────────────────────────
def regenerate_all(db_path: str):
    """Regenerate embeddings + metadata for all permanent artworks and sync to Weaviate."""
    # SQLite connection
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Select artworks marked permanent
    cur.execute("""
        SELECT id, filename, filepath
        FROM artworks
        WHERE is_permanent = 1
            AND color IS NOT NULL
    """)

    rows = cur.fetchall()
    if not rows:
        logger.warning("⚠️ No permanent artworks found.")
        return

    db = SessionLocal()
    logger.info(f"🔁 Regenerating {len(rows)} artworks…")

    for r in tqdm(rows, desc="Syncing", unit="img"):
        art_id = r["id"]
        filename = r["filename"]
        filepath = r["filepath"]

        # 🖼️ Recursively search for the image in known folders
        abs_path = None
        search_name = os.path.basename(filepath)
        possible_paths = []

        for base in ["images", "data/wikiart", "data/images"]:
            possible_paths.extend(glob.glob(f"{base}/**/{search_name}", recursive=True))

        if possible_paths:
            abs_path = possible_paths[0]
        else:
            logger.warning(f"⚠️ Missing file for {filename}")
            continue

        try:
            # ─────────────────────
            # 1️⃣ Generate embeddings (safe)
            # ─────────────────────
            embs = {
                "style":   safe_embedding(generate_style_embedding, abs_path),
                "texture": safe_embedding(generate_texture_embedding, abs_path),
                "color": safe_embedding(generate_color_embedding, abs_path),
                "emotion": safe_embedding(generate_emotion_embedding, abs_path),
            }

            # ─────────────────────
            # 2️⃣ Generate metadata
            # ─────────────────────
            tags = auto_tags_from_embeddings(embs)
            if not tags or not isinstance(tags, dict):
                logger.warning(f"⚠️ Invalid metadata for {filename}, using defaults")
                tags = {
                    "style": "Unknown",
                    "color": "Unknown",
                    "texture": "Unknown",
                    "emotion": "Unknown",
                }

            # ─────────────────────
            # 3️⃣ Update SQLite DB
            # ─────────────────────
            cur.execute("""
                UPDATE artworks
                SET style = ?,
                    color = ?,
                    texture = ?,
                    emotion = ?,
                    metadata_json = ?
                WHERE id = ?
            """, (
                tags["style"], tags["color"], tags["texture"], tags["emotion"],
                json.dumps(tags), art_id
            ))
            conn.commit()
            logger.info(f"✅ Updated DB for {filename}")

            # ─────────────────────
            # 4️⃣ Push to Weaviate
            # ─────────────────────
            dummy = type("Artwork", (), {})()
            dummy.id = art_id
            dummy.filename = filename
            dummy.filepath = filepath
            dummy.style = tags["style"]
            dummy.color = tags["color"]
            dummy.texture = tags["texture"]
            dummy.emotion = tags["emotion"]

            dummy.uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(art_id)))
            insert_embedding_to_weaviate(dummy, embs, object_id=dummy.uuid)
            logger.info(f"✅ Synced to Weaviate: {filename}")

        except Exception as e:
            logger.error(f"❌ Failed for {filename}: {e}")
            continue

    logger.info("🎯 Regeneration & sync complete.")
    db.close()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DB_PATH = os.getenv("DB_PATH", "./data/artworks.db")
    regenerate_all(DB_PATH)
