import os
from google.cloud import storage
from google.cloud import aiplatform
import subprocess
import tempfile
import shutil

def deploy_to_gcp():
    # Initialize GCP clients
    storage_client = storage.Client()
    aiplatform.init(project="your-project-id", location="us-central1")
    
    # Create a temporary directory for model artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy model files to temp directory
        model_files = [
            "inference.py",
            "requirements.txt",
            "model.pth"
        ]
        
        for file in model_files:
            shutil.copy2(file, os.path.join(temp_dir, file))
        
        # Create a Dockerfile
        dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "inference.py"]
"""
        with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)
        
        # Build and push Docker image to Google Container Registry
        image_name = "gcr.io/your-project-id/video-sentiment-model"
        subprocess.run([
            "docker", "build", "-t", image_name, temp_dir
        ])
        subprocess.run([
            "docker", "push", image_name
        ])
        
        # Deploy to Cloud Run
        service_name = "video-sentiment-service"
        subprocess.run([
            "gcloud", "run", "deploy", service_name,
            "--image", image_name,
            "--platform", "managed",
            "--region", "us-central1",
            "--allow-unauthenticated",
            "--memory", "4Gi",
            "--cpu", "2",
            "--min-instances", "0",
            "--max-instances", "10"
        ])

if __name__ == "__main__":
    deploy_to_gcp() 