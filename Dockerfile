# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "rmbg-ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
