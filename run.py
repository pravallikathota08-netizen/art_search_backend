#!/usr/bin/env python3
"""
Startup script for the AI Art Search API
"""

import uvicorn
import os
from pathlib import Path

def main():
    """Start the FastAPI application"""
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    print("Starting AI Art Search API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("Default credentials: admin/admin")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
