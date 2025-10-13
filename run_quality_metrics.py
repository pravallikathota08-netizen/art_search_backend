# run_quality_metrics.py
"""
Utility script to compute and log data quality metrics for embeddings.
Run it anytime from the terminal:
    python3 run_quality_metrics.py
"""

from database import SessionLocal
from data_quality_metrics import compute_data_quality_metrics

def main():
    print("🚀 Running Data Quality Metrics Audit...")
    db = SessionLocal()
    try:
        record = compute_data_quality_metrics(db)
        print("\n✅ Successfully logged metrics to database:")
        print(f"  - Total embeddings: {record.total_embeddings}")
        print(f"  - Valid embeddings: {record.valid_embeddings}")
        print(f"  - Invalid embeddings: {record.invalid_embeddings}")
        print(f"  - Avg norm: {record.avg_norm:.4f}")
        print(f"  - Std deviation: {record.norm_std:.4f}")
        print(f"  - Status: {record.status}")
    except Exception as e:
        print(f"❌ Error running quality metrics: {e}")
    finally:
        db.close()
        print("📦 Database connection closed.")

if __name__ == "__main__":
    main()
