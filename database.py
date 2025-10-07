# database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ────────────────────────────────────────────────
# Database URL: defaults to SQLite if not provided
# ────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/artworks.db")

# ────────────────────────────────────────────────
# Connection args: only for SQLite
# ────────────────────────────────────────────────
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

# ────────────────────────────────────────────────
# Engine setup (echo=False avoids noisy logs)
# ────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)

# ────────────────────────────────────────────────
# Session and Base
# ────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def get_db():
    """FastAPI dependency to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create tables and seed default admin user if missing."""
    from models import User, Artwork, Embedding
    from auth import get_password_hash

    print(f"🗄️ Connecting to database: {DATABASE_URL}")

    # Create all tables
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

        admin_user = db.query(User).filter(User.username == admin_username).first()
        if not admin_user:
            admin_user = User(
                username=admin_username,
                hashed_password=get_password_hash(admin_password),
                role="admin",
            )
            db.add(admin_user)
            db.commit()
            print(f"👤 Default admin created (username={admin_username}, password={admin_password})")
        else:
            print("ℹ️ Admin user already exists")
    finally:
        db.close()

    print("✅ Database initialization complete.")
