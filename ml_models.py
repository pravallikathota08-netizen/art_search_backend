import numpy as np
import json
from PIL import Image
import os

# Embedding dimensions
EMBEDDING_DIM = 512

def generate_style_embedding(image_path):
    """
    Generate style embedding for an image.
    In a real application, this would use a model like CLIP or DINOv2.
    For now, we return a mock vector based on image properties.
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image properties that could influence style
        width, height = image.size
        aspect_ratio = width / height
        
        # Create a deterministic but varied embedding based on image properties
        np.random.seed(hash(image_path) % 2**32)
        base_vector = np.random.rand(EMBEDDING_DIM)
        
        # Modify vector based on image characteristics
        base_vector[0] = aspect_ratio
        base_vector[1] = width / 1000.0  # Normalized width
        base_vector[2] = height / 1000.0  # Normalized height
        
        # Add some variation based on file size
        file_size = os.path.getsize(image_path)
        base_vector[3] = (file_size % 1000) / 1000.0
        
        return base_vector.tolist()
        
    except Exception as e:
        print(f"Error generating style embedding for {image_path}: {e}")
        # Return random vector as fallback
        np.random.seed(42)
        return np.random.rand(EMBEDDING_DIM).tolist()

def generate_texture_embedding(image_path):
    """
    Generate texture embedding for an image.
    In a real application, this would analyze texture patterns.
    """
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate texture-related features
        gray = np.mean(img_array, axis=2)
        
        # Calculate local standard deviation as texture measure
        texture_measure = np.std(gray)
        
        # Create embedding based on texture properties
        np.random.seed(hash(image_path) % 2**32 + 1)
        base_vector = np.random.rand(EMBEDDING_DIM)
        
        # Modify based on texture characteristics
        base_vector[0] = texture_measure / 100.0  # Normalized texture measure
        base_vector[1] = np.mean(gray) / 255.0  # Average brightness
        
        return base_vector.tolist()
        
    except Exception as e:
        print(f"Error generating texture embedding for {image_path}: {e}")
        np.random.seed(43)
        return np.random.rand(EMBEDDING_DIM).tolist()

def generate_palette_embedding(image_path):
    """
    Generate color palette embedding for an image.
    In a real application, this would extract dominant colors.
    """
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image for faster processing
        image = image.resize((100, 100))
        img_array = np.array(image)
        
        # Calculate color statistics
        mean_r = np.mean(img_array[:, :, 0])
        mean_g = np.mean(img_array[:, :, 1])
        mean_b = np.mean(img_array[:, :, 2])
        
        std_r = np.std(img_array[:, :, 0])
        std_g = np.std(img_array[:, :, 1])
        std_b = np.std(img_array[:, :, 2])
        
        # Create embedding based on color properties
        np.random.seed(hash(image_path) % 2**32 + 2)
        base_vector = np.random.rand(EMBEDDING_DIM)
        
        # Modify based on color characteristics
        base_vector[0] = mean_r / 255.0
        base_vector[1] = mean_g / 255.0
        base_vector[2] = mean_b / 255.0
        base_vector[3] = std_r / 255.0
        base_vector[4] = std_g / 255.0
        base_vector[5] = std_b / 255.0
        
        return base_vector.tolist()
        
    except Exception as e:
        print(f"Error generating palette embedding for {image_path}: {e}")
        np.random.seed(44)
        return np.random.rand(EMBEDDING_DIM).tolist()

def generate_emotion_embedding(image_path):
    """
    Generate emotion embedding for an image.
    In a real application, this would use emotion recognition models.
    """
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate brightness and contrast as emotion indicators
        gray = np.mean(img_array, axis=2)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Calculate color saturation
        hsv = image.convert('HSV')
        hsv_array = np.array(hsv)
        saturation = np.mean(hsv_array[:, :, 1])
        
        # Create embedding based on emotional characteristics
        np.random.seed(hash(image_path) % 2**32 + 3)
        base_vector = np.random.rand(EMBEDDING_DIM)
        
        # Modify based on emotional characteristics
        base_vector[0] = brightness / 255.0
        base_vector[1] = contrast / 255.0
        base_vector[2] = saturation / 255.0
        
        return base_vector.tolist()
        
    except Exception as e:
        print(f"Error generating emotion embedding for {image_path}: {e}")
        np.random.seed(45)
        return np.random.rand(EMBEDDING_DIM).tolist()
