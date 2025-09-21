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

# Models are now imported from models.py to avoid duplication

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
def init_db():
    # Import models to ensure they're registered with Base
    import models
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Enable vector search extension
    conn = sqlite3.connect("artworks.db")
    conn.enable_load_extension(True)
    
    try:
        # Load the sqlite-vec extension
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        # Test the extension
        vec_version, = conn.execute("SELECT vec_version()").fetchone()
        print(f"sqlite-vec extension loaded successfully, version: {vec_version}")
        
        # Create vector tables for efficient similarity search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS artworks_vectors USING vec0(
                style_embedding float[512],
                texture_embedding float[512],
                palette_embedding float[512],
                emotion_embedding float[512]
            )
        """)
        
    except Exception as e:
        print(f"Warning: Could not load sqlite-vec extension: {e}")
        print("Vector search will use fallback similarity calculation")
        conn.enable_load_extension(False)
    
    # Create default admin user
    from auth import get_password_hash
    from models import User
    
    db = SessionLocal()
    try:
        # Check if admin user exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                hashed_password=get_password_hash("admin")
            )
            db.add(admin_user)
            db.commit()
            print("Default admin user created (username: admin, password: admin)")
    finally:
        db.close()
    
    conn.close()
