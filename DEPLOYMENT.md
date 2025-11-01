# Deployment Guide

## Overview

This document provides instructions for deploying the Housing Price Prediction API.

## Prerequisites

- Docker and Docker Compose installed
- OR Python 3.9+ and Conda installed

## Quick Start with Docker (Recommended)

1. **Build and start the API**:
   ```bash
   docker-compose up --build
   ```

2. **The API will be available at**: `http://localhost:5000`

3. **Test the API**:
   ```bash
   # In a new terminal
   python test_api.py
   ```

4. **Stop the API**:
   ```bash
   docker-compose down
   ```

## Development Setup (Without Docker)

1. **Create Conda environment**:
   ```bash
   conda env create -f conda_environment.yml
   conda activate housing
   ```

2. **Install Flask** (not in conda_environment.yml):
   ```bash
   pip install Flask gunicorn requests
   ```

3. **Train the model** (if not already done):
   ```bash
   python create_model.py
   ```

4. **Run the API**:
   ```bash
   python app.py
   ```

5. **Test the API** (in a new terminal):
   ```bash
   conda activate housing
   python test_api.py
   ```

## API Endpoints

### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "service": "Housing Price Prediction API",
  "version": "1.0.0"
}
```

### GET /api/v1/model/info
Get model information.

**Response**:
```json
{
  "model_name": "Housing Price Predictor",
  "model_type": "K-Nearest Neighbors Regression",
  "version": "1.0",
  "required_features": ["bedrooms", "bathrooms", ...]
}
```

### POST /api/v1/predict
Make a prediction with all available features.

**Request**:
```json
{
  "bedrooms": 4,
  "bathrooms": 1.0,
  "sqft_living": 1680,
  "sqft_lot": 5043,
  "floors": 1.5,
  "waterfront": 0,
  "view": 0,
  "condition": 4,
  "grade": 6,
  "sqft_above": 1680,
  "sqft_basement": 0,
  "yr_built": 1911,
  "yr_renovated": 0,
  "zipcode": "98118",
  "lat": 47.5354,
  "long": -122.273,
  "sqft_living15": 1560,
  "sqft_lot15": 5765
}
```

**Response**:
```json
{
  "predicted_price": 324500.50,
  "currency": "USD",
  "model_version": "1.0",
  "demographic_data_included": true,
  "message": "Prediction successful"
}
```

### POST /api/v1/predict/minimal
Make a prediction with only required features (Bonus endpoint).

**Request**:
```json
{
  "bedrooms": 4,
  "bathrooms": 1.0,
  "sqft_living": 1680,
  "sqft_lot": 5043,
  "floors": 1.5,
  "sqft_above": 1680,
  "sqft_basement": 0,
  "zipcode": "98118"
}
```

**Response**: Same as `/api/v1/predict`

## Scaling Considerations

### Horizontal Scaling
The API is stateless and can be scaled horizontally:

```yaml
# docker-compose.yml
services:
  api:
    build: .
    deploy:
      replicas: 3  # Run 3 instances
```

Use a load balancer (nginx, HAProxy) to distribute traffic:
```yaml
services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

  api:
    build: .
    deploy:
      replicas: 3
```

### Vertical Scaling
Increase worker processes in Dockerfile:
```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "8", "app:app"]
```

### Model Versioning
To deploy a new model version without downtime:

1. Train new model and save to `model_v2/`
2. Update `app.py` to load from new path
3. Build new Docker image: `docker build -t housing-api:v2 .`
4. Use blue-green deployment or rolling update

## Health Checks

The `/health` endpoint is used by Docker Compose for health checks:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

## Monitoring

Consider adding:
- Prometheus metrics
- Application logging (structured JSON logs)
- Request tracing
- Performance monitoring

## Security

For production deployment:
- Add authentication (API keys, OAuth)
- Enable HTTPS/TLS
- Implement rate limiting
- Set up firewall rules
- Use secrets management for sensitive data
