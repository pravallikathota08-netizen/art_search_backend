import os, shutil, pandas as pd

splits = {
    "train": "data/train_split.csv",
    "val":   "data/val_split.csv",
    "test":  "data/test_split.csv",
}

for split, csv_path in splits.items():
    df = pd.read_csv(csv_path)
    out_dir = f"data/{split}"
    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():
        src = row["filepath"]
        dst = os.path.join(out_dir, os.path.basename(src))
        if os.path.exists(src):
            shutil.copy(src, dst)

    print(f"✅ Copied {len(df)} images → {out_dir}")