import os
import google.cloud.storage as storage
from google.cloud import aiplatform
import tempfile
import shutil
import subprocess

def create_dockerfile():
    """Create Dockerfile for the model container"""
    dockerfile_content = """
FROM python:3.11-slim

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
COPY inference.py .

# Set environment variables
ENV PORT=8080

# Run the application
CMD ["python", "inference.py"]
"""
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)

def build_and_push_docker_image(project_id):
    """Build and push Docker image to Google Container Registry"""
    # Create a temporary directory for the build context
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy necessary files to temp directory
        files_to_copy = ['Dockerfile', 'requirements.txt', 'model.pth', 'inference.py']
        for file in files_to_copy:
            if os.path.exists(file):
                shutil.copy2(file, os.path.join(temp_dir, file))
        
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Build and push the Docker image
            image_name = f"gcr.io/{project_id}/sentiment-analysis-model"
            subprocess.run([
                "gcloud", "builds", "submit",
                "--tag", image_name,
                "."
            ], check=True)
            return image_name
        finally:
            # Change back to original directory
            os.chdir(original_dir)

def upload_model_to_gcs(model_path, bucket_name):
    """Upload model files to Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Create a temporary directory to store the model files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy model files to temp directory
        model_files = ['model.pth', 'inference.py']
        for file in model_files:
            shutil.copy2(os.path.join(model_path, file), temp_dir)
        
        # Upload files to GCS
        for file in model_files:
            blob = bucket.blob(f'model/{file}')
            blob.upload_from_filename(os.path.join(temp_dir, file))

def deploy_model(project_id, region, bucket_name):
    """Deploy the model to Vertex AI"""
    aiplatform.init(project=project_id, location=region)
    
    # Build and push Docker image
    container_image = build_and_push_docker_image(project_id)
    
    # Create model
    model = aiplatform.Model.upload(
        display_name="sentiment-analysis-model",
        artifact_uri=f"gs://{bucket_name}/model",
        serving_container_image_uri=container_image,
    )
    
    # Deploy model to endpoint
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=1,
    )
    
    return endpoint

def main():
    # GCP Configuration
    project_id = "sentiment-analysis-456808"  # Replace with your GCP project ID
    region = "us-central1"  # Replace with your preferred region
    bucket_name = "sentiment-analysis-saas"  # Replace with your GCS bucket name
    
    # Create Dockerfile
    create_dockerfile()
    
    # Upload model to GCS
    upload_model_to_gcs("model", bucket_name)
    
    # Deploy model
    endpoint = deploy_model(project_id, region, bucket_name)
    print(f"Model deployed to endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    main() 