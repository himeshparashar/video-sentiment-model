FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model.pth .
COPY models.py .
COPY inference.py .

# Set environment variables
ENV PORT=8080

# Run the application
CMD ["python", "inference.py"] 