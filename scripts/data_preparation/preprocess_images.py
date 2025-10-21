import os
from PIL import Image

def preprocess_folder(folder, size=(224, 224)):
    count = 0
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(size)
            img.save(path, quality=90)
            count += 1
        except Exception as e:
            print(f"⚠️  Skipped {filename}: {e}")
    print(f"✅ Processed {count} images in {folder}")

for split in ["train", "val", "test"]:
    preprocess_folder(f"data/{split}")