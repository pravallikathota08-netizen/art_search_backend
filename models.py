from sqlalchemy import Column, Integer, String, Text, Float,DateTime,ForeignKey 
from sqlalchemy.orm import relationship
from datetime import datetime
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
    filename = Column(String, index=True)
    image_path = Column(String)
    style_embedding = Column(Text)  # JSON string of vector
    texture_embedding = Column(Text)  # JSON string of vector
    palette_embedding = Column(Text)  # JSON string of vector
    emotion_embedding = Column(Text)  # JSON string of vector
