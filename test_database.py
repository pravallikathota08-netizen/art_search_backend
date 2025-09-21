#!/usr/bin/env python3
"""
Test script to verify that the database models are working correctly
"""

from database import init_db, get_db
from models import User, Artwork

def test_database_init():
    """Test that database initialization works without errors"""
    try:
        print("Testing database initialization...")
        init_db()
        print("✅ Database initialization successful!")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def test_model_creation():
    """Test that models can be created without conflicts"""
    try:
        print("Testing model creation...")
        db = next(get_db())
        
        # Test User model
        user = User(username="test_user", hashed_password="test_hash")
        print("✅ User model created successfully!")
        
        # Test Artwork model
        artwork = Artwork(
            filename="test.jpg",
            image_path="/test/path",
            style_embedding="[]",
            texture_embedding="[]",
            palette_embedding="[]",
            emotion_embedding="[]"
        )
        print("✅ Artwork model created successfully!")
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Running database tests...")
    print("-" * 50)
    
    success = True
    success &= test_database_init()
    success &= test_model_creation()
    
    print("-" * 50)
    if success:
        print("🎉 All tests passed! The SQLAlchemy error has been fixed.")
    else:
        print("💥 Some tests failed. Please check the errors above.")
