import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import whisper
from transformers import AutoTokenizer
import sys
import json
from google.cloud import storage
import tempfile
from flask import Flask, request, jsonify

app = Flask(__name__)

EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Initialize model and components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalSentimentModel().to(device)
model_path = os.path.join(os.getcwd(), 'model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
transcriber = whisper.load_model(
    "base",
    device="cpu" if device.type == "cpu" else device,
)

# Initialize GCP clients
storage_client = storage.Client()

def download_from_gcs(gcs_uri):
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(temp_file.name)
        return temp_file.name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'video_path' not in request.json:
            return jsonify({'error': 'No video_path provided'}), 400
            
        gcs_uri = request.json['video_path']
        local_path = download_from_gcs(gcs_uri)
        
        result = transcriber.transcribe(local_path, word_timestamps=True)
        utterance_processor = VideoUtteranceProcessor()
        predictions = []

        for segment in result["segments"]:
            try:
                segment_path = utterance_processor.extract_segment(
                    local_path,
                    segment["start"],
                    segment["end"]
                )

                video_frames = utterance_processor.video_processor.process_video(
                    segment_path)
                audio_features = utterance_processor.audio_processor.extract_features(
                    segment_path)
                text_inputs = tokenizer(
                    segment["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                # Move to device
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                video_frames = video_frames.unsqueeze(0).to(device)
                audio_features = audio_features.unsqueeze(0).to(device)

                # Get predictions
                with torch.inference_mode():
                    outputs = model(text_inputs, video_frames, audio_features)
                    emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                    sentiment_probs = torch.softmax(
                        outputs["sentiments"], dim=1)[0]

                    emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                    sentiment_values, sentiment_indices = torch.topk(
                        sentiment_probs, 3)

                predictions.append({
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"],
                    "emotions": [
                        {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()} for idx, conf in zip(emotion_indices, emotion_values)
                    ],
                    "sentiments": [
                        {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()} for idx, conf in zip(sentiment_indices, sentiment_values)
                    ]
                })

            except Exception as e:
                print("Segment failed inference: " + str(e))

            finally:
                # Cleanup
                if os.path.exists(segment_path):
                    os.remove(segment_path)
        
        # Cleanup
        if os.path.exists(local_path):
            os.remove(local_path)
            
        return jsonify({"utterances": predictions})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))