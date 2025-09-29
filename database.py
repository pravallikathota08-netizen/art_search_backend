from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlite3
import sqlite_vec
import os

# Database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./artworks.db"

# Create engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database
def init_db():
    # Import models so they're registered
    import models
    from models import User
    from auth import get_password_hash

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Enable vector search extension
    conn = sqlite3.connect("artworks.db")
    conn.enable_load_extension(True)

    try:
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        vec_version, = conn.execute("SELECT vec_version()").fetchone()
        print(f"✅ sqlite-vec extension loaded successfully, version: {vec_version}")

        # Create virtual table for embeddings
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS artworks_vectors USING vec0(
                style_embedding float[512],
                texture_embedding float[512],
                palette_embedding float[512],
                emotion_embedding float[512]
            )
        """)
    except Exception as e:
        print(f"⚠️ Warning: Could not load sqlite-vec extension: {e}")
        print("➡️ Falling back to Python similarity calculation")
        conn.enable_load_extension(False)

    # Create default admin user
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                hashed_password=get_password_hash("admin123"),  # default password
                role="admin"  # make sure role column exists in models.User
            )
            db.add(admin_user)
            db.commit()
            print("👤 Default admin created (username=admin, password=admin123)")
        else:
            print("ℹ️ Admin user already exists")
    finally:
        db.close()

    conn.close()
