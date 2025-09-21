# Installation Guide for AI Art Search API

This guide provides detailed installation instructions for the AI Art Search API with sqlite-vec support.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd art_search_backend
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install sqlite-vec Extension

The `sqlite-vec` extension needs to be installed separately. Here are the installation methods:

#### Option A: Using pip (Recommended)

```bash
pip install sqlite-vec
```

#### Option B: Using conda

```bash
conda install -c conda-forge sqlite-vec
```

#### Option C: From Source

```bash
# Clone sqlite-vec repository
git clone https://github.com/asg017/sqlite-vec.git
cd sqlite-vec

# Build and install
python -m pip install -e .
```

### 5. Verify Installation

Run the test script to verify everything is working:

```bash
python test_api.py
```

This will test:
- sqlite-vec extension loading
- API health check
- Authentication
- Image upload and processing
- Vector similarity search

### 6. Start the Application

```bash
python run.py
```

Or alternatively:

```bash
uvicorn main:app --reload
```

## Troubleshooting

### sqlite-vec Installation Issues

If you encounter issues with sqlite-vec installation:

1. **Windows Users**: Make sure you have Visual Studio Build Tools installed
2. **macOS Users**: Install Xcode command line tools: `xcode-select --install`
3. **Linux Users**: Install build essentials: `sudo apt-get install build-essential`

### Database Issues

If the database initialization fails:

1. Delete the existing database file: `rm artworks.db`
2. Restart the application
3. Check the console output for error messages

### Vector Search Issues

If vector search is not working:

1. Check if sqlite-vec is properly installed: `python -c "import sqlite_vec; print('OK')"`
2. Verify the extension loads: Run `python test_api.py`
3. Check the application logs for fallback messages

## Platform-Specific Notes

### Windows

- Ensure you have the Microsoft Visual C++ Redistributable installed
- Use PowerShell or Command Prompt as Administrator if needed

### macOS

- You may need to install Xcode command line tools
- Use Homebrew for easier dependency management: `brew install python`

### Linux

- Install Python development headers: `sudo apt-get install python3-dev`
- For Ubuntu/Debian: `sudo apt-get install build-essential`
- For CentOS/RHEL: `sudo yum groupinstall "Development Tools"`

## Development Setup

For development, you may want to install additional tools:

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Production Deployment

For production deployment:

1. Use a production WSGI server like Gunicorn
2. Set up proper environment variables
3. Use a reverse proxy like Nginx
4. Consider using PostgreSQL with pgvector for better performance

Example production command:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Support

If you encounter issues:

1. Check the [sqlite-vec documentation](https://github.com/asg017/sqlite-vec)
2. Review the application logs
3. Run the test script to identify specific issues
4. Check the FastAPI documentation for API-related issues
