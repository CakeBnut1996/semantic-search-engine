# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency files
COPY pyproject.toml .
# If we have a lockfile, copy it too
# COPY uv.lock .

# Install dependencies (only the core API dependencies)
RUN pip install --no-cache-dir .

# Copy the source code
COPY seed/ ./seed/
COPY config.yaml .

# Make port 8000 available
EXPOSE 8000

# Run the FastAPI server using the entry point defined in pyproject.toml
CMD ["uvicorn", "seed.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
