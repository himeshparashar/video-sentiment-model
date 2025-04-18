from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from inference import model_fn, predict_fn
import json
from typing import List, Dict, Any
from contextlib import asynccontextmanager

model_dict = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_dict
    model_dict = model_fn(".")
    yield
    # Cleanup if needed

app = FastAPI(title="Video Sentiment Analysis API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Prepare input data
        input_data = {"video_path": temp_file_path}
        
        # Run prediction
        predictions = predict_fn(input_data, model_dict)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 