# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in pyproject.toml
# Note: Using pip directly for simplicity, or we could use uv
RUN pip install . fastapi uvicorn

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run server.py when the container launches
CMD ["uvicorn", "server.py:app", "--host", "0.0.0.0", "--port", "8000"]
