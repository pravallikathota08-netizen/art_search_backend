# load_dataset.py
import os
import numpy as np
from typing import Iterable
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
from models import Artwork, Embedding

# Import embedding generators directly
from ml_models import (
    generate_style_embedding,
    generate_texture_embedding,
    generate_palette_embedding,
    generate_emotion_embedding,
)

# ───────────────────────────────
# Helper 1: Iterate over all image paths
# ───────────────────────────────
def iter_image_paths(root: str) -> Iterable[str]:
    """Yield full paths of all valid image files in dataset_dir."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                yield os.path.join(dirpath, name)

# ───────────────────────────────
# Helper 2: Build all embeddings for an image
# ───────────────────────────────
def build_combined_embedding(image_path: str):
    """Generate and combine all embeddings (style, texture, color, emotion)."""
    try:
        s = np.array(generate_style_embedding(image_path), dtype=np.float32)
        t = np.array(generate_texture_embedding(image_path), dtype=np.float32)
        c = np.array(generate_palette_embedding(image_path), dtype=np.float32)
        e = np.array(generate_emotion_embedding(image_path), dtype=np.float32)

        combined = np.concatenate([s, t, c, e])
        return combined, {
            "style_vector": s,
            "texture_vector": t,
            "color_vector": c,
            "emotion_vector": e,
        }
    except Exception as ex:
        print(f"⚠️ Error generating embeddings for {os.path.basename(image_path)}: {ex}")
        return None, None

# ───────────────────────────────
# Main: Load dataset and store in DB
# ───────────────────────────────
def load_dataset_once(dataset_dir: str = "data/wikiart", batch_size: int = 50):
    """Walk through images, generate embeddings, and insert into DB."""
    init_db()
    db: Session = SessionLocal()

    inserted, skipped, batch = 0, 0, 0

    for path in iter_image_paths(dataset_dir):
        filename = os.path.basename(path)

        # Skip already existing images
        exists = db.query(Artwork).filter(Artwork.filename == filename).one_or_none()
        if exists:
            skipped += 1
            continue

        try:
            art = Artwork(filename=filename, filepath=path)
            db.add(art)
            db.flush()  # get generated ID

            combined, per_axis = build_combined_embedding(path)
            if combined is None:
                db.rollback()
                continue

            emb = Embedding(
                artwork_id=art.id,
                vector=combined.tolist(),
                style_vector=per_axis["style_vector"].tolist(),
                color_vector=per_axis["color_vector"].tolist(),
                texture_vector=per_axis["texture_vector"].tolist(),
                emotion_vector=per_axis["emotion_vector"].tolist(),
            )
            db.add(emb)

            batch += 1
            inserted += 1

            # Commit in batches for performance
            if batch >= batch_size:
                db.commit()
                print(f"✅ Committed {inserted} total so far... (skipped {skipped})")
                batch = 0

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            db.rollback()

    db.commit()
    db.close()
    print(f"🎯 Dataset load complete. Inserted: {inserted}, Skipped: {skipped}")

# ───────────────────────────────
# Entry point
# ───────────────────────────────
if __name__ == "__main__":
    load_dataset_once()