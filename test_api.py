#!/usr/bin/env python3
"""
Simple test script for the AI Art Search API
Run this after starting the FastAPI server to test the endpoints
"""

import requests
import json
import os
from PIL import Image
import numpy as np
import sqlite3
import sqlite_vec

# API base URL
BASE_URL = "http://localhost:8000"

def create_test_image(filename, size=(100, 100), color=(255, 0, 0)):
    """Create a simple test image"""
    image = Image.new('RGB', size, color)
    image.save(filename)
    return filename

def test_sqlite_vec():
    """Test sqlite-vec extension"""
    print("Testing sqlite-vec extension...")
    try:
        conn = sqlite3.connect("artworks.db")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        vec_version, = conn.execute("SELECT vec_version()").fetchone()
        print(f"sqlite-vec version: {vec_version}")
        
        # Test vector operations
        test_vector = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 507  # 512-dim vector
        result = conn.execute("SELECT vec_distance_cosine(?, ?)", (test_vector, test_vector)).fetchone()
        print(f"Self-similarity test: {result[0]} (should be 0.0)")
        
        conn.close()
        print("sqlite-vec test passed!")
    except Exception as e:
        print(f"sqlite-vec test failed: {e}")
    print()

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_authentication():
    """Test authentication"""
    print("Testing authentication...")
    data = {
        'username': 'admin',
        'password': 'admin'
    }
    response = requests.post(f"{BASE_URL}/token", data=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        token_data = response.json()
        print(f"Token received: {token_data['access_token'][:50]}...")
        return token_data['access_token']
    else:
        print(f"Error: {response.text}")
        return None

def test_bulk_upload(token):
    """Test bulk upload endpoint"""
    print("Testing bulk upload...")
    
    # Create test images
    test_images = []
    for i in range(3):
        filename = f"test_image_{i}.jpg"
        create_test_image(filename, color=(i*80, 100, 200-i*50))
        test_images.append(filename)
    
    # Prepare files for upload
    files = []
    for filename in test_images:
        files.append(('files', (filename, open(filename, 'rb'), 'image/jpeg')))
    
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.post(f"{BASE_URL}/upload/bulk/", files=files, headers=headers)
    
    # Close files
    for _, (_, file_obj, _) in files:
        file_obj.close()
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Clean up test images
    for filename in test_images:
        if os.path.exists(filename):
            os.remove(filename)

def test_search(token):
    """Test search endpoint"""
    print("Testing search...")
    
    # Create a query image
    query_image = "query_image.jpg"
    create_test_image(query_image, color=(150, 75, 100))
    
    # Search for similar images
    headers = {'Authorization': f'Bearer {token}'}
    files = {'file': (query_image, open(query_image, 'rb'), 'image/jpeg')}
    
    response = requests.post(f"{BASE_URL}/search/", files=files, headers=headers)
    files['file'][1].close()
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} similar images:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['imageUrl']} (similarity: {result.get('similarity', 'N/A')})")
    else:
        print(f"Error: {response.text}")
    print()
    
    # Clean up query image
    if os.path.exists(query_image):
        os.remove(query_image)

def main():
    """Run all tests"""
    print("AI Art Search API Test Suite")
    print("=" * 40)
    
    # Test sqlite-vec extension
    test_sqlite_vec()
    
    # Test health
    test_health()
    
    # Test authentication
    token = test_authentication()
    if not token:
        print("Authentication failed. Cannot continue with other tests.")
        return
    
    # Test bulk upload
    test_bulk_upload(token)
    
    # Test search
    test_search(token)
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
