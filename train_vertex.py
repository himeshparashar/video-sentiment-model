from google.cloud import aiplatform

def start_training():
    aiplatform.init(project="sentiment-analysis-456808", location="us-central1")

    job = aiplatform.CustomJob(
        display_name="pytorch-training-job",
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_P100",
                    "accelerator_count": 1
                },
                "replica_count": 1,
                "python_package_spec": {
                    "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
                    "package_uris": ["gs://sentiment-analysis-saas/trainer-0.1.tar.gz"],  # Optional: GCS .tar.gz if using a custom package
                    "python_module": "train_gcp",  # The `train.py` script
                    "args": [
                        "--batch-size", "32",
                        "--epochs", "25",
                        "--train-path", "gs://sentiment-analysis-saas/dataset/train",
                        "--val-path", "gs://sentiment-analysis-saas/dataset/dev",
                        "--test-path", "gs://sentiment-analysis-saas/dataset/test"
                    ]
                }
            }
        ],
        base_output_dir="gs://sentiment-analysis-saas/tensorboard/",
        staging_bucket="gs://sentiment-analysis-saas/staging",
        
    )

    job.run(sync=True)

if __name__ == "__main__":
    start_training()
