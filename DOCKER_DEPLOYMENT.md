# Docker Deployment Guide

This guide explains how to deploy the AI Art Search Backend using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier deployment)

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and navigate to the project directory:**
   ```bash
   cd art_search_backend
   ```

2. **Create environment file (optional):**
   ```bash
   cp env.example .env
   # Edit .env file with your preferred settings
   ```

3. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t art-search-backend .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 \
     -e SECRET_KEY="your-secret-key" \
     -e ADMIN_USERNAME="admin" \
     -e ADMIN_PASSWORD="admin" \
     -v $(pwd)/images:/app/images \
     -v $(pwd)/data:/app/data \
     art-search-backend
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `your-secret-key-change-this-in-production` | Secret key for JWT tokens |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token expiration time in minutes |
| `DATABASE_URL` | `sqlite:///./artworks.db` | Database connection string |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed CORS origins |
| `ADMIN_USERNAME` | `admin` | Default admin username |
| `ADMIN_PASSWORD` | `admin` | Default admin password |
| `EMBEDDING_DIM` | `512` | Embedding dimensions for ML models |

## Volumes

The following directories are mounted as volumes for data persistence:

- `./images:/app/images` - Stores uploaded images
- `./data:/app/data` - Stores database and other application data

## API Endpoints

- `POST /token` - Authentication (username/password: admin/admin)
- `POST /upload/bulk/` - Bulk upload images
- `POST /search/` - Search similar artworks
- `GET /health` - Health check
- `GET /docs` - API documentation (Swagger UI)

## Production Deployment

For production deployment, consider the following:

1. **Change default credentials:**
   ```bash
   export ADMIN_USERNAME="your-admin-username"
   export ADMIN_PASSWORD="your-secure-password"
   ```

2. **Use a strong secret key:**
   ```bash
   export SECRET_KEY="your-very-secure-secret-key"
   ```

3. **Configure proper CORS origins:**
   ```bash
   export CORS_ORIGINS="https://yourdomain.com,https://www.yourdomain.com"
   ```

4. **Use a production database:**
   ```bash
   export DATABASE_URL="postgresql://user:password@host:port/dbname"
   ```

## Troubleshooting

### Container won't start
- Check if port 8000 is already in use
- Verify Docker is running
- Check container logs: `docker-compose logs`

### Permission issues
- Ensure the application has write permissions to mounted volumes
- Check file ownership in the container

### Database issues
- The SQLite database will be created automatically
- For data persistence, ensure the `./data` directory is mounted

## Development

For development with hot reload:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Create a `docker-compose.dev.yml` file with:

```yaml
version: '3.8'
services:
  art-search-backend:
    volumes:
      - .:/app
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

## Monitoring

The application includes a health check endpoint at `/health` that returns the application status.

Monitor the application logs:
```bash
docker-compose logs -f art-search-backend
```
