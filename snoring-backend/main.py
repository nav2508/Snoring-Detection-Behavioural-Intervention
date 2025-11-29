import os
import io
import tempfile
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib to use Agg backend for non-GUI environments
import matplotlib
matplotlib.use('Agg')

app = FastAPI(title="Snoring Detection API", version="1.0.0")

# CORS middleware to allow requests from your Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
MODEL_PATHS = [
    'models/best_snoring_model.keras',
    'models/final_snoring_detection_model.h5',
    'best_snoring_model.keras',
    'final_snoring_detection_model.h5'
]

def create_model_architecture():
    """Create the exact model architecture used in training"""
    def create_improved_model(input_shape=(61, 257, 1)):
        """Create the improved CNN model matching your architecture"""
        model = tf.keras.Sequential([
            # First conv block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                                  input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second conv block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third conv block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth conv block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global pooling and dense layers
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    return create_improved_model()

def load_model_with_architecture(model_path: str):
    """Load model by first creating architecture then loading weights"""
    try:
        # Method 1: Try direct load first
        logger.info(f"Attempting direct model load from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Model loaded successfully with direct method")
        return model
    except Exception as e:
        logger.warning(f"Direct load failed: {e}")
        
        try:
            # Method 2: Create architecture and load weights
            logger.info("Creating model architecture and loading weights...")
            model = create_model_architecture()
            
            # Load weights
            if model_path.endswith('.keras') or model_path.endswith('.h5'):
                model.load_weights(model_path)
                logger.info("✅ Model weights loaded successfully")
                
                # Compile the model (not necessary for inference but good practice)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                return model
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return None
                
        except Exception as e2:
            logger.error(f"Failed to load weights: {e2}")
            return None

@app.on_event("startup")
async def load_model():
    """Load the TensorFlow model on startup"""
    global model
    try:
        for model_path in MODEL_PATHS:
            if os.path.exists(model_path):
                logger.info(f"Found model at: {model_path}")
                model = load_model_with_architecture(model_path)
                if model is not None:
                    logger.info(f"✅ Model successfully loaded from: {model_path}")
                    
                    # Test the model with a dummy input to verify it works
                    try:
                        dummy_input = np.random.random((1, 61, 257, 1)).astype(np.float32)
                        prediction = model.predict(dummy_input, verbose=0)
                        logger.info(f"✅ Model test prediction successful: {prediction[0][0]}")
                    except Exception as test_error:
                        logger.warning(f"Model test failed: {test_error}")
                    
                    break
                else:
                    logger.warning(f"Failed to load model from: {model_path}")
        else:
            logger.error("❌ No model file found or all loading attempts failed.")
            logger.info("Please ensure model files are in one of these locations:")
            for path in MODEL_PATHS:
                logger.info(f"  - {path}")
            
    except Exception as e:
        logger.error(f"❌ Error during model loading process: {e}")
        model = None

@app.get("/")
async def root():
    return {"message": "Snoring Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for connection testing"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_status": model_status
    }

def convert_mp3_to_wav(mp3_file_content: bytes) -> io.BytesIO:
    """Convert MP3 file content to WAV format in memory"""
    try:
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
            temp_mp3.write(mp3_file_content)
            temp_mp3_path = temp_mp3.name
        
        # Load and convert MP3
        audio = AudioSegment.from_file(temp_mp3_path, format="mp3")
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Export to bytes buffer
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Clean up temporary file
        os.unlink(temp_mp3_path)
        
        return wav_buffer
        
    except Exception as e:
        logger.error(f"Error converting MP3 to WAV: {e}")
        raise HTTPException(status_code=400, detail=f"MP3 conversion failed: {str(e)}")

def load_mp3_as_tensor(mp3_file_content: bytes):
    """Load MP3 file content and convert to TensorFlow tensor"""
    try:
        wav_buffer = convert_mp3_to_wav(mp3_file_content)
        audio_binary = wav_buffer.read()
        
        # Decode WAV using TensorFlow
        wav, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        
        logger.info(f"MP3 loaded successfully. Length: {len(wav)} samples, Duration: {len(wav)/16000:.2f} seconds")
        return wav
        
    except Exception as e:
        logger.error(f"Error loading MP3 as tensor: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio file: {str(e)}")

def create_spectrogram(segment):
    """Create spectrogram from audio segment matching your preprocessing"""
    frame_length = 512
    frame_step = 256
    
    spectrogram = tf.signal.stft(segment, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    
    # Normalize
    mean = tf.math.reduce_mean(spectrogram)
    std = tf.math.reduce_std(spectrogram)
    spectrogram = (spectrogram - mean) / (std + 1e-6)
    
    # Expand dimensions for CNN input
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # Add channel dimension
    return spectrogram

def create_visualizations(wav, predictions, snoring_segments, filename, audio_duration):
    """Create comprehensive visualizations and return as base64 strings"""
    try:
        visualizations = {}
        
        # Create a larger figure for better mobile display
        plt.style.use('default')
        
        # Figure 1: Comprehensive Analysis Plot (similar to your original)
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle(f'Snoring Analysis: {filename}', fontsize=16, fontweight='bold')
        
        # Plot 1: Audio waveform
        time_axis = np.linspace(0, len(wav)/16000, len(wav))
        ax1.plot(time_axis, wav.numpy(), alpha=0.7, color='blue', linewidth=0.5)
        ax1.set_title('Audio Waveform')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlabel('Time (seconds)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Snoring probability over time
        segment_times = [(i * 0.5) for i in range(len(predictions))]
        ax2.plot(segment_times, predictions, 'o-', color='red', alpha=0.7, markersize=3, linewidth=1)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')
        ax2.set_title('Snoring Probability Over Time')
        ax2.set_ylabel('Snoring Probability')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Snoring segments visualization
        for i, is_snoring in enumerate(snoring_segments):
            color = 'red' if is_snoring else 'green'
            alpha = 0.6 if is_snoring else 0.3
            ax3.axvspan(i * 0.5, (i + 1) * 0.5, color=color, alpha=alpha)
        ax3.set_title('Snoring Detection Timeline')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Snoring')
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['No', 'Yes'])
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Snoring distribution pie chart
        snoring_count = sum(snoring_segments)
        non_snoring_count = len(snoring_segments) - snoring_count
        labels = ['Snoring', 'Non-Snoring']
        sizes = [snoring_count, non_snoring_count]
        colors = ['#ff6b6b', '#51cf66']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Snoring Distribution')
        
        plt.tight_layout()
        
        # Convert to base64
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
        buf1.seek(0)
        visualizations['analysis_plot'] = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)
        
        # Figure 2: Statistics and Metrics
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Confidence distribution
        ax1.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Snoring intervals duration
        interval_durations = []
        current_duration = 0
        for is_snoring in snoring_segments:
            if is_snoring:
                current_duration += 0.5
            else:
                if current_duration > 0:
                    interval_durations.append(current_duration)
                    current_duration = 0
        if current_duration > 0:
            interval_durations.append(current_duration)
            
        if interval_durations:
            ax2.hist(interval_durations, bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax2.set_title('Snoring Interval Durations')
            ax2.set_xlabel('Duration (seconds)')
            ax2.set_ylabel('Frequency')
        else:
            ax2.text(0.5, 0.5, 'No Snoring Intervals', ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Snoring pattern over time (smoothed)
        from scipy.ndimage import gaussian_filter1d
        if len(predictions) > 10:
            smoothed_predictions = gaussian_filter1d(predictions, sigma=2)
            ax3.plot(segment_times, smoothed_predictions, color='purple', linewidth=2)
            ax3.set_title('Smoothed Snoring Pattern')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Snoring Probability')
            ax3.set_ylim(0, 1)
        else:
            ax3.text(0.5, 0.5, 'Insufficient Data\nfor Smoothing', ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        stats_text = f"""
        Analysis Summary:
        
        Total Duration: {audio_duration:.1f}s
        Segments Analyzed: {len(predictions)}
        Snoring Segments: {snoring_count}
        Snoring Ratio: {snoring_count/len(predictions)*100:.1f}%
        Snoring Intervals: {len(interval_durations)}
        Avg Interval Duration: {np.mean(interval_durations):.1f}s
        Max Interval Duration: {np.max(interval_durations) if interval_durations else 0:.1f}s
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Analysis Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        visualizations['statistics_plot'] = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)
        
        # Figure 3: Simple timeline for mobile (vertical layout)
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        # Create a simple timeline
        ax.plot(segment_times, predictions, 'o-', color='red', alpha=0.7, markersize=2, linewidth=1)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        
        # Highlight snoring regions
        for i, is_snoring in enumerate(snoring_segments):
            if is_snoring:
                ax.axvspan(i * 0.5, (i + 1) * 0.5, color='red', alpha=0.3)
        
        ax.set_title('Snoring Detection Timeline')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Snoring Probability')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
        buf3.seek(0)
        visualizations['timeline_plot'] = base64.b64encode(buf3.read()).decode('utf-8')
        plt.close(fig3)
        
        logger.info("✅ All visualizations generated successfully")
        return visualizations
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return {}

def preprocess_mp3_for_prediction(mp3_file_content: bytes, filename: str):
    """Preprocess MP3 file and make predictions"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Load MP3 as tensor
    wav = load_mp3_as_tensor(mp3_file_content)
    
    if wav is None:
        raise HTTPException(status_code=400, detail="Failed to load MP3 file")
    
    # Process the entire audio file in segments
    segment_length = 16000  # 1 second segments
    hop_length = 8000       # 0.5 second hop (50% overlap)
    
    segments = []
    start_idx = 0
    
    while start_idx + segment_length <= len(wav):
        segment = wav[start_idx:start_idx + segment_length]
        segments.append(segment)
        start_idx += hop_length
    
    logger.info(f"Split audio into {len(segments)} segments for analysis")
    
    # Process each segment
    predictions = []
    confidences = []
    segment_predictions = []
    
    for i, segment in enumerate(segments):
        try:
            # Ensure segment is exactly 16000 samples
            if len(segment) < segment_length:
                padding = segment_length - len(segment)
                segment = tf.pad(segment, [[0, padding]], mode='CONSTANT')
            
            # Create spectrogram
            spectrogram = create_spectrogram(segment)
            
            # Expand batch dimension
            spectrogram = tf.expand_dims(spectrogram, axis=0)
            
            # Predict
            prediction = model.predict(spectrogram, verbose=0)
            
            # Handle prediction output
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                prediction_value = float(prediction[0][0]) if len(prediction.shape) > 1 else float(prediction[0])
            else:
                prediction_value = float(prediction)
                
            predictions.append(prediction_value)
            confidence = prediction_value if prediction_value > 0.5 else 1 - prediction_value
            confidences.append(confidence)
            
            # Store segment prediction details
            segment_prediction = {
                "segment": i + 1,
                "class": "Snoring" if prediction_value > 0.5 else "Not Snoring",
                "confidence": float(confidence),
                "probability": float(prediction_value)
            }
            segment_predictions.append(segment_prediction)
            
        except Exception as e:
            logger.error(f"Error processing segment {i}: {e}")
            # Default to not snoring if prediction fails
            predictions.append(0.0)
            confidences.append(0.0)
            segment_predictions.append({
                "segment": i + 1,
                "class": "Not Snoring",
                "confidence": 0.0,
                "probability": 0.0
            })
    
    return wav, predictions, confidences, segment_predictions

def analyze_snoring_pattern(predictions, confidences, threshold=0.5):
    """Analyze the snoring pattern across segments"""
    snoring_segments = [1 if p > threshold else 0 for p in predictions]
    
    # Count snoring segments
    total_segments = len(snoring_segments)
    snoring_count = sum(snoring_segments)
    snoring_ratio = snoring_count / total_segments if total_segments > 0 else 0
    
    # Find snoring intervals (consecutive snoring segments)
    snoring_intervals = []
    current_interval = []
    
    for i, is_snoring in enumerate(snoring_segments):
        if is_snoring:
            current_interval.append(i)
        else:
            if len(current_interval) >= 2:  # At least 2 consecutive segments (1 second)
                start_time = current_interval[0] * 0.5  # Convert to seconds
                end_time = (current_interval[-1] + 1) * 0.5
                duration = end_time - start_time
                
                snoring_intervals.append({
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "duration": round(duration, 2),
                    "segment_count": len(current_interval)
                })
            current_interval = []
    
    # Don't forget the last interval
    if len(current_interval) >= 2:
        start_time = current_interval[0] * 0.5
        end_time = (current_interval[-1] + 1) * 0.5
        duration = end_time - start_time
        
        snoring_intervals.append({
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(duration, 2),
            "segment_count": len(current_interval)
        })
    
    logger.info(f"Snoring Analysis: {total_segments} segments, {snoring_count} snoring ({snoring_ratio*100:.1f}%), {len(snoring_intervals)} intervals")
    
    return {
        "total_segments": total_segments,
        "snoring_segments": snoring_count,
        "snoring_ratio": float(snoring_ratio),
        "snoring_intervals": snoring_intervals,
        "interval_count": len(snoring_intervals)
    }, snoring_segments

def get_snoring_level_message(snoring_ratio: float) -> str:
    """Generate a user-friendly message based on snoring ratio"""
    if snoring_ratio > 0.3:
        return "🔊 HIGH snoring activity detected! Consider consulting a sleep specialist."
    elif snoring_ratio > 0.1:
        return "🔈 MODERATE snoring activity detected. Monitor your sleep patterns."
    else:
        return "🔇 LOW or no snoring detected. Your sleep appears to be quiet."

@app.post("/analyze-snoring")
async def analyze_snoring(file: UploadFile = File(...)):
    """Main endpoint for snoring analysis"""
    try:
        # Validate file type
        if not file.content_type or ("audio" not in file.content_type and not file.filename.lower().endswith('.mp3')):
            raise HTTPException(status_code=400, detail="Please upload an MP3 audio file")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        logger.info(f"Processing file: {file.filename}, Size: {len(file_content)} bytes")
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="AI model is not loaded. Please check server configuration.")
        
        # Process the audio
        wav, predictions, confidences, segment_predictions = preprocess_mp3_for_prediction(file_content, file.filename)
        
        # Analyze snoring pattern
        analysis_results, snoring_segments = analyze_snoring_pattern(predictions, confidences)
        
        # Calculate audio duration
        audio_duration = len(wav) / 16000  # seconds
        
        # Generate visualizations
        visualizations = create_visualizations(wav, predictions, snoring_segments, file.filename, audio_duration)
        
        # Generate conclusion message
        conclusion_message = get_snoring_level_message(analysis_results["snoring_ratio"])
        
        # Prepare response
        response_data = {
            "status": "success",
            "filename": file.filename,
            "audio_duration": round(audio_duration, 2),
            "analysis": analysis_results,
            "segment_predictions": segment_predictions,
            "message": conclusion_message,
            "summary": f"Snoring present in {analysis_results['snoring_ratio']*100:.1f}% of the audio",
            "visualizations": visualizations
        }
        
        logger.info(f"Analysis completed for {file.filename}: {analysis_results['snoring_ratio']*100:.1f}% snoring")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during snoring analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"model_loaded": False}
    
    try:
        return {
            "model_loaded": True,
            "model_layers": len(model.layers),
            "model_input_shape": str(model.input_shape),
            "model_output_shape": str(model.output_shape),
            "model_type": type(model).__name__
        }
    except Exception as e:
        return {
            "model_loaded": True,
            "model_type": type(model).__name__,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)