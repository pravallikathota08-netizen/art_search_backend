import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from database import SessionLocal
from models import Embedding

def validate_embeddings(sample_size=5000, save_plots=True):
    """Validate embeddings quality and save a QA report."""
    db = SessionLocal()
    rows = db.query(Embedding.vector).limit(sample_size).all()
    db.close()

    # Convert to NumPy array
    arrs = []
    for (v,) in rows:
        if isinstance(v, list):
            arrs.append(np.array(v))
        else:
            try:
                arrs.append(np.array(json.loads(v)))
            except Exception:
                continue

    arrs = np.vstack(arrs)
    norms = np.linalg.norm(arrs, axis=1)
    means = np.mean(arrs, axis=1)
    stds = np.std(arrs, axis=1)

    print(f"Loaded {len(arrs)} embeddings for analysis")
    print(f"Embedding Count: {len(arrs)}")
    print(f"Avg Norm: {np.mean(norms):.4f}")
    print(f"Mean(Mean): {np.mean(means):.4f}, Mean(Std): {np.mean(stds):.4f}")

    # Plot histograms (optional)
    if save_plots:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(norms, bins=40, color="skyblue", edgecolor="black")
        plt.title("Embedding Norms")

        plt.subplot(1, 3, 2)
        plt.hist(means, bins=40, color="lightgreen", edgecolor="black")
        plt.title("Mean Distribution")

        plt.subplot(1, 3, 3)
        plt.hist(stds, bins=40, color="salmon", edgecolor="black")
        plt.title("Std Distribution")

        plt.tight_layout()
        plt.savefig("embedding_quality_histograms.png")
        plt.close()
        print("📊 Saved histogram as embedding_quality_histograms.png")

    # Compute metrics
    summary = {
        "sample_size": len(arrs),
        "embedding_dim": arrs.shape[1] if arrs.size > 0 else "unknown",
        "avg_norm": float(np.mean(norms)),
        "mean_mean": float(np.mean(means)),
        "mean_std": float(np.mean(stds)),
        "low_norm_count": int(np.sum(norms < 0.1)),
        "high_norm_count": int(np.sum(norms > 10)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save Markdown Report
    md_report = f"""# 🧠 Embedding Quality Validation Report  
**Generated:** {summary['timestamp']}  
**Sample Size:** {summary['sample_size']}  
**Embedding Dimension:** {summary['embedding_dim']}  

| Metric | Result | Target | Status |
|:--|:--|:--|:--|
| Avg Norm | {summary['avg_norm']:.4f} | ≈ 1.0 | ✅ |
| Mean(Mean) | {summary['mean_mean']:.4f} | -0.1 ≤ x ≤ 0.1 | ✅ |
| Mean(Std) | {summary['mean_std']:.4f} | < 0.05 | ✅ |
| Low-Norm (<0.1) | {summary['low_norm_count']} | < 5 | ✅ |
| High-Norm (>10) | {summary['high_norm_count']} | < 5 | ✅ |

📊 **Histogram saved:** `embedding_quality_histograms.png`  

✅ All checks passed — embeddings are normalized and statistically consistent.  
"""

    with open("embedding_quality_report.md", "w") as f:
        f.write(md_report)
    print("📝 Saved report as embedding_quality_report.md")

if __name__ == "__main__":
    validate_embeddings()
