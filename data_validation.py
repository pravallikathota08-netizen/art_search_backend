# data_validation.py
from sqlalchemy.orm import Session
from models import Artwork, MetadataValidation
from datetime import datetime

def validate_metadata_integrity(db: Session):
    results = {"checked": 0, "passed": 0, "failed": 0, "errors": []}

    artworks = db.query(Artwork).all()
    results["checked"] = len(artworks)

    for art in artworks:
        issues = []
        if not art.title or art.title.strip() == "":
            issues.append("Missing title")
        if not art.artist or art.artist.strip() == "":
            issues.append("Missing artist")
        if not art.filepath or art.filepath.strip() == "":
            issues.append("Missing filepath")
        if hasattr(art, "tags") and (not art.tags or len(art.tags) == 0):
            issues.append("No tags")

        if issues:
            results["failed"] += 1
            db.add(
                MetadataValidation(
                    artwork_id=art.id,
                    status="FAILED",
                    issues=", ".join(issues),
                    timestamp=datetime.utcnow(),
                )
            )
        else:
            results["passed"] += 1
            db.add(
                MetadataValidation(
                    artwork_id=art.id,
                    status="PASSED",
                    issues=None,
                    timestamp=datetime.utcnow(),
                )
            )

    db.commit()
    return results
