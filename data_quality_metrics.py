# data_quality_metrics.py
import json
import numpy as np
from datetime import datetime
from models import Embedding, DataQualityAudit

def compute_data_quality_metrics(db):
    rows = db.query(Embedding.vector).all()
    total = len(rows)
    valid = 0
    invalid = 0
    norms = []

    for (v,) in rows:
        try:
            vec = v if isinstance(v, list) else json.loads(v)
            arr = np.array(vec)
            norm = np.linalg.norm(arr)
            if np.isnan(norm):
                raise ValueError("NaN norm")
            norms.append(norm)
            valid += 1
        except Exception:
            invalid += 1

    avg_norm = np.mean(norms) if norms else 0
    norm_std = np.std(norms) if norms else 0

    # Apply quality thresholds
    status = "OK"
    if abs(avg_norm - 1.0) > 0.05 or invalid > 0.02 * total:
        status = "WARNING"
    if abs(avg_norm - 1.0) > 0.1 or invalid > 0.1 * total:
        status = "ALERT"

    record = DataQualityAudit(
        total_embeddings=total,
        valid_embeddings=valid,
        invalid_embeddings=invalid,
        avg_norm=avg_norm,
        norm_std=norm_std,
        timestamp=datetime.utcnow(),
        status=status
    )

    db.add(record)
    db.commit()

    print("✅ Data Quality Metrics Logged:")
    print(f"Total: {total}, Valid: {valid}, Invalid: {invalid}")
    print(f"Avg Norm: {avg_norm:.4f}, Std: {norm_std:.4f}, Status: {status}")

    return record
