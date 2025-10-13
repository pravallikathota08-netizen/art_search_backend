# load_dataset.py
import os
from typing import Iterable
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
from models import Artwork, Embedding
from main import build_combined_embedding

def iter_image_paths(root: str) -> Iterable[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                yield os.path.join(dirpath, name)

def load_dataset_once(dataset_dir: str = "data/wikiart", batch_size: int = 50):
    init_db()
    db: Session = SessionLocal()

    inserted, skipped, batch = 0, 0, 0

    for path in iter_image_paths(dataset_dir):
        filename = os.path.basename(path)
        exists = db.query(Artwork).filter(Artwork.filename == filename).one_or_none()
        if exists:
            skipped += 1
            continue

        try:
            art = Artwork(filename=filename, filepath=path)
            db.add(art)
            db.flush()

            combined, per_axis = build_combined_embedding(path)
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

if __name__ == "__main__":
    load_dataset_once()
