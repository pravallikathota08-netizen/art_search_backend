import os, sys

# 🔧 Add project root (where main.py, database.py, models.py live) to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

print("🧭 Project root added to path:", PROJECT_ROOT)

import pandas as pd
from database import SessionLocal
from models import Artwork

db = SessionLocal()
rows = db.query(Artwork).all()

data = [
    {
        "id": a.id,
        "filename": a.filename,
        "filepath": a.filepath,
        "style": a.style,
        "color": a.color,
        "texture": a.texture,
        "emotion": a.emotion,
    }
    for a in rows
]

df = pd.DataFrame(data)
df.to_csv("data/artworks_metadata.csv", index=False)
print(f"✅ Exported {len(df)} records to data/artworks_metadata.csv")
