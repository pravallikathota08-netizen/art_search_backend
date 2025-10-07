# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import JSON
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")

class Artwork(Base):
    __tablename__ = "artworks"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String)

    title = Column(String, nullable=True)
    artist = Column(String, nullable=True)
    style = Column(String, nullable=True)
    color = Column(String, nullable=True)
    texture = Column(String, nullable=True)
    emotion = Column(String, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    # ✅ single relationship, not a list
    embedding = relationship("Embedding", back_populates="artwork", uselist=False, cascade="all, delete-orphan")

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    artwork_id = Column(Integer, ForeignKey("artworks.id", ondelete="CASCADE"), unique=True, nullable=False)

    vector = Column(JSON, nullable=False)
    style_vector = Column(JSON, nullable=True)
    color_vector = Column(JSON, nullable=True)
    texture_vector = Column(JSON, nullable=True)
    emotion_vector = Column(JSON, nullable=True)

    artwork = relationship("Artwork", back_populates="embedding")
