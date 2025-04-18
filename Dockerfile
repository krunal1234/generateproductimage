# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CACHE_DIR="/app/cache" \
    OUTPUT_DIR="/app/output"

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Create necessary directories (CACHE_DIR, OUTPUT_DIR)
RUN mkdir -p /app/cache && mkdir -p /app/output

# Copy the application files to the container
COPY . /app/

# Expose the port the app runs on
EXPOSE 8000

# Ensure the model files are downloaded when starting the container
RUN python -c "from briarmbg import BriaRMBG; BriaRMBG()"

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "rmbg-ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
