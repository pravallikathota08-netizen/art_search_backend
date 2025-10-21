# split_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split

# ───────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────
DATA_FILE = "data/artworks_metadata.csv"  # path to your dataset metadata
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

# ───────────────────────────────────────────────
# LOAD & RANDOMIZE
# ───────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_FILE)

print("🔀 Randomizing dataset...")
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ───────────────────────────────────────────────
# SPLIT DATASET
# ───────────────────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=(1 - TRAIN_RATIO), random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), random_state=SEED)

print(f"✅ Split complete:")
print(f"  Train: {len(train_df)} rows")
print(f"  Val:   {len(val_df)} rows")
print(f"  Test:  {len(test_df)} rows")

# ───────────────────────────────────────────────
# SAVE SPLITS
# ───────────────────────────────────────────────
train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)
test_df.to_csv("data/test_split.csv", index=False)

print("💾 Saved split files:")
print("  • data/train_split.csv")
print("  • data/val_split.csv")
print("  • data/test_split.csv")
