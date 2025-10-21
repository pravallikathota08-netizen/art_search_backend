import pandas as pd

splits = {
    "train": "data/train_split.csv",
    "val":   "data/val_split.csv",
    "test":  "data/test_split.csv",
}

for name, path in splits.items():
    df = pd.read_csv(path)
    dist = df["style"].value_counts(normalize=True) * 100
    print(f"\n📊 {name.upper()} set distribution (%):\n{dist.round(2)}")
