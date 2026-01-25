"""
Debug visualization routes

Returns extracted frames, detected faces, transcribed text, and audio waveform
for debugging and frontend visualization purposes.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
import tempfile
import os
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import cv2

from src.services.multimodal_service import get_multimodal_service
from src.services.audio_analysis_service import get_audio_analysis_service
from src.core.face_detector import FaceDetector
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/debug", tags=["Debug"])


class BoundingBoxResponse(BaseModel):
    """Face bounding box"""
    x: int
    y: int
    width: int
    height: int


class FrameDebugInfo(BaseModel):
    """Debug info for a single extracted frame"""
    frame_index: int
    timestamp_sec: float
    image_base64: str = Field(description="Base64 encoded JPEG image")
    width: int
    height: int
    face_detected: bool
    face_bbox: Optional[BoundingBoxResponse] = None


class WaveformData(BaseModel):
    """Audio waveform data for visualization"""
    sample_rate: int
    duration_seconds: float
    # Downsampled waveform for visualization (e.g., 1000 points)
    waveform_points: List[float] = Field(description="Normalized amplitude values (-1 to 1)")
    rms_envelope: List[float] = Field(description="RMS energy envelope for smoother visualization")
    num_points: int


class DebugVisualizationResponse(BaseModel):
    """Complete debug visualization data"""
    success: bool
    frames: List[FrameDebugInfo]
    transcript: str
    transcript_length: int
    has_audio: bool
    waveform: Optional[WaveformData] = None
    audio_metrics_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


def extract_waveform_for_visualization(
    video_path: str,
    num_points: int = 500
) -> Optional[WaveformData]:
    """
    Extract audio waveform data optimized for frontend visualization.

    Args:
        video_path: Path to video file
        num_points: Number of waveform points to return (for chart rendering)

    Returns:
        WaveformData or None if no audio
    """
    try:
        from moviepy import VideoFileClip
        import librosa

        # Extract audio
        clip = VideoFileClip(video_path)

        if clip.audio is None:
            clip.close()
            return None

        # Create temp file for audio
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            # Extract audio to WAV
            clip.audio.write_audiofile(
                audio_path,
                fps=22050,
                nbytes=2,
                codec='pcm_s16le',
                logger=None
            )
            clip.close()

            # Load with librosa
            y, sr = librosa.load(audio_path, sr=22050)
            duration = len(y) / sr

            # Downsample waveform for visualization
            # Take evenly spaced samples
            if len(y) > num_points:
                indices = np.linspace(0, len(y) - 1, num_points, dtype=int)
                waveform_downsampled = y[indices]
            else:
                waveform_downsampled = y
                num_points = len(y)

            # Normalize to -1 to 1
            max_val = np.max(np.abs(waveform_downsampled)) + 1e-10
            waveform_normalized = (waveform_downsampled / max_val).tolist()

            # Calculate RMS envelope (smoother visualization)
            hop_length = max(1, len(y) // num_points)
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

            # Resize RMS to match num_points
            if len(rms) > num_points:
                rms_indices = np.linspace(0, len(rms) - 1, num_points, dtype=int)
                rms_downsampled = rms[rms_indices]
            else:
                rms_downsampled = rms

            # Normalize RMS to 0-1
            rms_max = np.max(rms_downsampled) + 1e-10
            rms_normalized = (rms_downsampled / rms_max).tolist()

            return WaveformData(
                sample_rate=sr,
                duration_seconds=round(duration, 2),
                waveform_points=waveform_normalized,
                rms_envelope=rms_normalized,
                num_points=num_points
            )

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    except Exception as e:
        logger.error(f"Waveform extraction failed: {e}")
        return None


def frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """Convert numpy frame to base64 JPEG"""
    # Convert to PIL
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        img = Image.fromarray(frame)
    else:
        img = Image.fromarray(frame).convert('RGB')

    # Resize if too large
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Encode to base64
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode('ascii')

    return f"data:image/jpeg;base64,{b64}"


def extract_frames_with_timestamps(video_path: str, num_frames: int = 10) -> List[Dict[str, Any]]:
    """
    Extract frames from video with timestamp information.

    Returns list of dicts with:
    - frame: numpy array (RGB)
    - timestamp: float (seconds)
    - index: int
    """
    try:
        from moviepy import VideoFileClip

        frames_data = []
        clip = VideoFileClip(video_path)

        try:
            duration = clip.duration

            for i in range(num_frames):
                if num_frames > 1:
                    timestamp = (i / (num_frames - 1)) * duration
                else:
                    timestamp = 0

                timestamp = min(timestamp, duration - 0.1)

                frame_array = clip.get_frame(timestamp)

                frames_data.append({
                    "frame": frame_array,
                    "timestamp": timestamp,
                    "index": i
                })

            return frames_data

        finally:
            clip.close()

    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return []


@router.post("/visualize", response_model=DebugVisualizationResponse)
async def get_debug_visualization(
    file: UploadFile = File(..., description="Video file to analyze"),
    num_frames: int = 10,
    waveform_points: int = 500
):
    """
    Extract debugging visualization data from a video.

    Returns:
    - **frames**: List of extracted frames with face detection bounding boxes
    - **transcript**: Transcribed text from audio
    - **waveform**: Audio waveform data for chart visualization
    - **audio_metrics_summary**: Summary of acoustic analysis

    Use this for debugging the analysis pipeline and displaying
    intermediate results in the frontend.
    """
    logger.info(f"Debug visualization request: {file.filename}")

    temp_video_path = None

    try:
        # Validate file type
        ext = Path(file.filename).suffix.lower() if file.filename else ""
        if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video required. Supported: {settings.ALLOWED_VIDEO_EXTENSIONS}"
            )

        # Read and save to temp file
        content = await file.read()

        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB"
            )

        fd, temp_video_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(temp_video_path, 'wb') as f:
            f.write(content)

        # Initialize services
        multimodal_service = get_multimodal_service()
        face_detector = FaceDetector()

        # 1. Extract frames with timestamps
        logger.info("Extracting frames...")
        frames_data = extract_frames_with_timestamps(temp_video_path, num_frames)

        # 2. Process each frame: detect faces, encode to base64
        frames_response = []
        for frame_info in frames_data:
            frame = frame_info["frame"]
            timestamp = frame_info["timestamp"]
            idx = frame_info["index"]

            # Detect face
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_bbox = face_detector.detect_largest_face(frame_bgr)

            # Convert to base64
            frame_b64 = frame_to_base64(frame)

            frame_debug = FrameDebugInfo(
                frame_index=idx,
                timestamp_sec=round(timestamp, 2),
                image_base64=frame_b64,
                width=frame.shape[1],
                height=frame.shape[0],
                face_detected=face_bbox is not None,
                face_bbox=BoundingBoxResponse(
                    x=face_bbox.x,
                    y=face_bbox.y,
                    width=face_bbox.width,
                    height=face_bbox.height
                ) if face_bbox else None
            )
            frames_response.append(frame_debug)

        logger.info(f"Processed {len(frames_response)} frames, faces detected: {sum(1 for f in frames_response if f.face_detected)}")

        # 3. Extract transcript
        logger.info("Extracting transcript...")
        transcript = ""
        has_audio = False
        try:
            audio_result = multimodal_service.analyze_video(temp_video_path)
            transcript = audio_result.get("transcript", "")
            has_audio = audio_result.get("metadata", {}).get("has_audio", False)
        except Exception as e:
            logger.warning(f"Transcript extraction failed: {e}")

        # 4. Extract waveform for visualization
        logger.info("Extracting waveform...")
        waveform = extract_waveform_for_visualization(temp_video_path, waveform_points)

        # 5. Get audio metrics summary
        logger.info("Extracting audio metrics...")
        audio_metrics_summary = None
        try:
            audio_service = get_audio_analysis_service()
            audio_result = audio_service.analyze_video(temp_video_path)
            if audio_result:
                # Extract key metrics for summary
                audio_metrics_summary = {
                    "personality_indicators": audio_result.get("personality_indicators", {}),
                    "interpretations": audio_result.get("interpretations", {}),
                    "normalized_metrics": audio_result.get("normalized_metrics", {})
                }
        except Exception as e:
            logger.warning(f"Audio metrics extraction failed: {e}")

        return DebugVisualizationResponse(
            success=True,
            frames=frames_response,
            transcript=transcript,
            transcript_length=len(transcript),
            has_audio=has_audio,
            waveform=waveform,
            audio_metrics_summary=audio_metrics_summary,
            metadata={
                "filename": file.filename,
                "file_size_bytes": len(content),
                "num_frames_extracted": len(frames_response),
                "faces_detected_count": sum(1 for f in frames_response if f.face_detected),
                "waveform_points": waveform.num_points if waveform else 0
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug visualization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visualization failed: {str(e)}"
        )
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except Exception:
                pass


@router.get("/demo")
async def demo_debug_response():
    """
    Demo endpoint showing the structure of debug visualization response.
    Use this to understand the data format before integrating.
    """
    return {
        "success": True,
        "frames": [
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...(truncated)",
                "width": 640,
                "height": 480,
                "face_detected": True,
                "face_bbox": {
                    "x": 150,
                    "y": 100,
                    "width": 200,
                    "height": 200
                }
            }
        ],
        "transcript": "Hello, this is a sample transcript from the video audio...",
        "transcript_length": 58,
        "has_audio": True,
        "waveform": {
            "sample_rate": 22050,
            "duration_seconds": 10.5,
            "waveform_points": [0.1, 0.3, -0.2, 0.5, -0.4],
            "rms_envelope": [0.2, 0.4, 0.3, 0.5, 0.4],
            "num_points": 500
        },
        "audio_metrics_summary": {
            "personality_indicators": {
                "vocal_extraversion": {"score": 72, "level": "High"},
                "vocal_stability": {"score": 65, "level": "Moderate"},
                "vocal_confidence": {"score": 68, "level": "Moderate"},
                "vocal_warmth": {"score": 55, "level": "Moderate"}
            },
            "interpretations": {
                "pitch": "Moderate pitch, balanced vocal tone",
                "expressiveness": "Highly expressive speech",
                "volume": "Moderate volume, conversational tone"
            }
        },
        "metadata": {
            "filename": "sample_video.mp4",
            "file_size_bytes": 15000000,
            "num_frames_extracted": 10,
            "faces_detected_count": 8,
            "waveform_points": 500
        },
        "note": "This is a demo response. POST to /debug/visualize with a video file for real data."
    }
