# AI-Powered Art and Design Style Search Application

A robust FastAPI backend that powers a multimodal similarity search engine for art and design styles. This application provides API endpoints for user authentication, bulk image processing, and similarity search using a vector-enabled SQLite database.

## Features

- **Authentication**: OAuth2 with Password Flow for secure API access
- **Bulk Image Processing**: Upload and process multiple images with automatic embedding generation
- **Similarity Search**: Find similar artworks using vector similarity search
- **Vector Database**: SQLite with vector search capabilities for efficient similarity queries
- **RESTful API**: Well-documented endpoints with automatic OpenAPI/Swagger documentation
- **CORS Support**: Configured for frontend integration

## Technology Stack

- **Framework**: FastAPI
- **Database**: SQLite with sqlite-vec extension
- **Image Processing**: Pillow (PIL)
- **Authentication**: JWT tokens with bcrypt password hashing
- **Vector Embeddings**: Placeholder functions (ready for integration with CLIP, DINOv2, etc.)

## Project Structure

```
art_search_backend/
├── main.py              # FastAPI application and endpoints
├── database.py          # Database configuration and models
├── models.py            # SQLAlchemy models
├── schemas.py           # Pydantic schemas for request/response
├── auth.py              # Authentication and authorization
├── ml_models.py         # ML model placeholders and embedding functions
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── images/             # Directory for uploaded images (created automatically)
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
uvicorn main:app --reload
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## API Endpoints

### Authentication

#### POST /token
Get access token for API authentication.

**Request Body** (form data):
```
username: admin
password: admin
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

### Image Processing

#### POST /upload/bulk/
Upload and process multiple images (requires authentication).

**Headers**:
```
Authorization: Bearer <access_token>
```

**Request Body** (multipart/form-data):
- `files`: List of image files

**Response**:
```json
{
  "message": "Successfully uploaded and processed 15 images."
}
```

### Search

#### POST /search/
Search for similar artworks (requires authentication).

**Headers**:
```
Authorization: Bearer <access_token>
```

**Request Body** (multipart/form-data):
- `file`: Single image file

**Response**:
```json
[
  {
    "imageUrl": "/images/result1.jpg",
    "tags": ["art", "style", "design"],
    "matchReason": "Similar style with 0.85 similarity score",
    "similarity": 0.85
  }
]
```

### Static Files

#### GET /images/{filename}
Serve uploaded images for frontend display.

### Health Check

#### GET /health
Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "message": "AI Art Search API is running"
}
```

## Database Schema

### Users Table
- `id`: Primary key
- `username`: Unique username
- `hashed_password`: Bcrypt hashed password

### Artworks Table
- `id`: Primary key
- `filename`: Original filename
- `image_path`: Path to stored image
- `style_embedding`: JSON string of style vector (512 dimensions)
- `texture_embedding`: JSON string of texture vector (512 dimensions)
- `palette_embedding`: JSON string of color palette vector (512 dimensions)
- `emotion_embedding`: JSON string of emotion vector (512 dimensions)

## Vector Search Extension

The application uses the `sqlite-vec` extension for efficient vector similarity search. This extension provides:

- **Vector Storage**: Efficient storage of high-dimensional vectors using virtual tables
- **Similarity Search**: Fast cosine similarity calculations with built-in distance functions
- **Indexing**: Optimized search performance with vector indices
- **Native Integration**: Seamless integration with SQLite for vector operations

### Embedding Dimensions

All embeddings are **512-dimensional** vectors, which is a common size for many pre-trained models like CLIP, DINOv2, and other vision transformers.

### sqlite-vec Features

The application leverages sqlite-vec's advanced features:

- **Virtual Tables**: Uses `vec0` virtual table for efficient vector storage
- **Distance Functions**: Utilizes `vec_distance_cosine()` for similarity calculations
- **Automatic Indexing**: Built-in vector indexing for fast similarity searches
- **Fallback Support**: Graceful degradation to manual similarity calculation if extension fails

Example sqlite-vec usage:
```python
import sqlite3
import sqlite_vec

db = sqlite3.connect("artworks.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Create vector table
db.execute("""
    CREATE VIRTUAL TABLE artworks_vectors USING vec0(
        style_embedding float[512],
        texture_embedding float[512],
        palette_embedding float[512],
        emotion_embedding float[512]
    )
""")

# Perform similarity search
results = db.execute("""
    SELECT *, vec_distance_cosine(style_embedding, ?) as distance
    FROM artworks_vectors
    ORDER BY distance
    LIMIT 10
""", (query_vector,)).fetchall()
```

## CORS Configuration

The API is configured to allow requests from:
- `http://localhost:3000` (React development server)
- `http://127.0.0.1:3000` (Alternative localhost)

To modify CORS settings, update the `allow_origins` list in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Default Credentials

The application creates a default admin user on startup:
- **Username**: `admin`
- **Password**: `admin`

**Important**: Change these credentials in production!

## ML Model Integration

The current implementation uses placeholder functions that generate deterministic random vectors based on image properties. To integrate real ML models:

1. Replace the functions in `ml_models.py` with actual model calls
2. Install the required ML libraries (torch, transformers, etc.)
3. Update the `requirements.txt` with new dependencies
4. Modify the embedding dimensions if needed

Example integration with CLIP:
```python
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_style_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.squeeze().tolist()
```

## Error Handling

The API includes comprehensive error handling:
- **401 Unauthorized**: Invalid or missing authentication
- **400 Bad Request**: Invalid file types or malformed requests
- **500 Internal Server Error**: Server-side processing errors

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Database Reset

To reset the database, simply delete the `artworks.db` file and restart the application.

### Adding New Users

Users can be added programmatically through the database or by modifying the `init_db()` function in `database.py`.

## Production Considerations

1. **Security**: Change default credentials and secret keys
2. **Database**: Consider using PostgreSQL with pgvector for production
3. **File Storage**: Use cloud storage (AWS S3, Google Cloud Storage) for images
4. **Caching**: Implement Redis for caching frequent searches
5. **Monitoring**: Add logging and monitoring for production use
6. **Scaling**: Consider using multiple workers with Gunicorn

## License

This project is open source and available under the MIT License.
