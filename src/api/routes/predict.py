"""
Prediction routes

This API always accepts VIDEO uploads regardless of model type:

1. **ResNet** (MODEL_TYPE=resnet):
   - Extracts 10 evenly-spaced frames from video (configurable via RESNET_NUM_FRAMES)
   - Applies face detection on each frame
   - Runs ResNet prediction on all frames
   - Averages the scores for more robust results
   - Image size: 256x256

2. **VAT** (MODEL_TYPE=vat):
   - Extracts 32 frames using k-segment sampling
   - Image size: 224x224

**Multimodal Analysis (both models):**
When generate_report=true, the API will:
1. Extract 10 frames from the video for LLM analysis
2. Transcribe the audio using OpenAI
3. Generate personality insights using both frames and transcript

Supported video formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm
"""
from fastapi import APIRouter, UploadFile, File, Depends, Query, Form, HTTPException, status
from typing import Optional
from pathlib import Path
import gc
import hashlib
import tempfile
import os
import asyncio
import json

from src.api.schemas.request import FileUploadParams
from src.api.schemas.response import PredictionResponse, AssessmentMetadata
from src.services.model_manager import ModelManager, get_model_manager
from src.services.prediction_service import PredictionService
from src.services.preprocessing_service import PreprocessingService
from src.services.multimodal_service import get_multimodal_service
from src.services.audio_analysis_service import get_audio_analysis_service
from src.infrastructure.cache import CacheService, get_cache_service
from src.infrastructure.storage import StorageService, get_storage_service
from src.core.video_processor import compress_video
from src.core.face_detector import FaceDetector
import cv2
import numpy as np
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import (
    FileUploadError,
    FileSizeExceededError,
    UnsupportedFileTypeError
)

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])


def format_audio_metrics_for_response(audio_analysis: dict) -> dict:
    """
    Format audio analysis results into the AudioMetrics response schema.

    Args:
        audio_analysis: Raw audio analysis from AudioAnalysisService

    Returns:
        Formatted dict matching AudioMetrics schema
    """
    if not audio_analysis:
        return None

    indicators = audio_analysis.get("personality_indicators", {})
    interpretations = audio_analysis.get("interpretations", {})

    # Get scores from the 3 indicators: vocal_extraversion, vocal_expressiveness, vocal_fluency
    vocal_ext = indicators.get("vocal_extraversion", {}).get("score", 50)
    vocal_expr = indicators.get("vocal_expressiveness", {}).get("score", 50)
    vocal_fluency = indicators.get("vocal_fluency", {}).get("score", 50)

    # Generate coach recommendation based on indicator scores
    if vocal_ext > 65 and vocal_expr > 60:
        coach_rec = "Your voice projects energy and expressiveness. The varied intonation engages listeners effectively. Consider using strategic pauses to emphasize key points."
    elif vocal_fluency > 65:
        coach_rec = "Your speech flows smoothly and confidently. This fluent delivery style conveys competence and helps maintain listener attention."
    elif vocal_expr > 65:
        coach_rec = "Your expressive vocal style adds color and emphasis to your communication. This helps convey emotion and keep listeners engaged."
    elif vocal_fluency < 40:
        coach_rec = "Consider reducing pauses and filler words to improve speech flow. Practice speaking in complete thoughts to convey more confidence."
    elif vocal_expr < 40:
        coach_rec = "Adding more vocal variety could make your speech more engaging. Try varying your pitch and volume to emphasize important points."
    else:
        coach_rec = "Your vocal patterns are balanced. Focus on matching your vocal energy to the context - more expressive for persuasion, calmer for difficult conversations."

    # Generate actionable steps based on areas for improvement
    steps = []
    if vocal_fluency < 50:
        steps.append({"emoji": "🎤", "text": "Practice speaking without filler words"})
    if vocal_expr < 50:
        steps.append({"emoji": "🎭", "text": "Add more pitch variation when speaking"})
    if vocal_ext < 40:
        steps.append({"emoji": "🔊", "text": "Project your voice with more energy"})

    # Default steps if none added
    if not steps:
        steps = [
            {"emoji": "🎯", "text": "Record yourself to identify patterns"},
            {"emoji": "⏸️", "text": "Use strategic pauses for emphasis"},
            {"emoji": "🔊", "text": "Match vocal energy to your message"}
        ]

    return {
        "indicators": indicators,
        "interpretations": interpretations,
        "coach_recommendation": coach_rec,
        "actionable_steps": steps
    }


async def run_audio_analysis_async(video_path: str) -> dict:
    """Run audio analysis in a thread pool to avoid blocking."""
    audio_service = get_audio_analysis_service()
    return await asyncio.to_thread(audio_service.analyze_video, video_path)


def extract_debug_visualization(
    video_path: str,
    raw_frames: list,
    transcript: str,
    audio_analysis: dict
) -> dict:
    """
    Extract debug visualization data for frontend display.

    Returns dict with:
    - frames: List of frame info with base64 images and face bboxes
    - transcript: Transcribed text
    - waveform: Audio waveform data for visualization
    - audio_summary: Key audio metrics
    """
    from PIL import Image
    from io import BytesIO
    import base64

    debug_data = {
        "frames": [],
        "transcript": transcript or "",
        "transcript_length": len(transcript) if transcript else 0,
        "waveform": None,
        "audio_summary": None
    }

    # Process frames with face detection
    if raw_frames:
        face_detector = FaceDetector()

        for i, frame in enumerate(raw_frames):
            try:
                # Convert to BGR for OpenCV face detection
                if isinstance(frame, np.ndarray):
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_rgb = frame
                else:
                    frame_rgb = np.array(frame)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Detect face
                face_bbox = face_detector.detect_largest_face(frame_bgr)

                # Convert frame to base64 (resized for efficiency)
                img = Image.fromarray(frame_rgb)
                if img.width > 320 or img.height > 320:
                    img.thumbnail((320, 320), Image.Resampling.LANCZOS)

                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=75)
                b64 = base64.b64encode(buffer.getvalue()).decode('ascii')

                frame_info = {
                    "index": i,
                    "image_base64": f"data:image/jpeg;base64,{b64}",
                    "width": frame_rgb.shape[1],
                    "height": frame_rgb.shape[0],
                    "face_detected": face_bbox is not None,
                    "face_bbox": {
                        "x": face_bbox.x,
                        "y": face_bbox.y,
                        "width": face_bbox.width,
                        "height": face_bbox.height
                    } if face_bbox else None
                }
                debug_data["frames"].append(frame_info)

            except Exception as e:
                logger.warning(f"Failed to process debug frame {i}: {e}")

    # Extract waveform for visualization
    if video_path:
        try:
            waveform = extract_waveform_for_debug(video_path)
            if waveform:
                debug_data["waveform"] = waveform
        except Exception as e:
            logger.warning(f"Waveform extraction failed: {e}")

    # Add audio metrics summary
    if audio_analysis:
        debug_data["audio_summary"] = {
            "indicators": audio_analysis.get("personality_indicators", {}),
            "interpretations": audio_analysis.get("interpretations", {})
        }

    return debug_data


def extract_waveform_for_debug(video_path: str, num_points: int = 200) -> dict:
    """Extract audio waveform data for visualization."""
    import tempfile

    # Try moviepy first, fall back to ffmpeg for WebM and other problematic formats
    audio_path = _extract_audio_for_waveform_moviepy(video_path)
    if not audio_path:
        audio_path = _extract_audio_for_waveform_ffmpeg(video_path)

    if not audio_path:
        return None

    try:
        import librosa

        # Load with librosa
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr

        # Downsample for visualization
        if len(y) > num_points:
            indices = np.linspace(0, len(y) - 1, num_points, dtype=int)
            waveform = y[indices]
        else:
            waveform = y
            num_points = len(y)

        # Normalize
        max_val = np.max(np.abs(waveform)) + 1e-10
        waveform_normalized = (waveform / max_val).tolist()

        # RMS envelope
        hop_length = max(1, len(y) // num_points)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        if len(rms) > num_points:
            rms_indices = np.linspace(0, len(rms) - 1, num_points, dtype=int)
            rms = rms[rms_indices]
        rms_max = np.max(rms) + 1e-10
        rms_normalized = (rms / rms_max).tolist()

        return {
            "duration_seconds": round(duration, 2),
            "waveform_points": waveform_normalized,
            "rms_envelope": rms_normalized,
            "num_points": num_points
        }
    except Exception as e:
        logger.warning(f"Waveform processing error: {e}")
        return None
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass


def _extract_audio_for_waveform_moviepy(video_path: str) -> str:
    """Extract audio using moviepy (preferred for most formats)."""
    import tempfile

    try:
        from moviepy import VideoFileClip

        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None

        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        clip.audio.write_audiofile(audio_path, fps=22050, nbytes=2, codec='pcm_s16le', logger=None)
        clip.close()

        return audio_path
    except Exception as e:
        logger.debug(f"moviepy audio extraction failed: {e}")
        return None


def _extract_audio_for_waveform_ffmpeg(video_path: str) -> str:
    """Extract audio using ffmpeg CLI (fallback for WebM and other formats)."""
    import subprocess
    import shutil
    import tempfile

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.warning("ffmpeg not found in PATH, cannot extract audio for waveform")
        return None

    audio_path = None
    try:
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        cmd = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",
            "-ac", "1",
            audio_path
        ]

        logger.info(f"Extracting audio for waveform with ffmpeg")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            logger.warning(f"ffmpeg waveform audio extraction failed: {result.stderr[:200]}")
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return None

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logger.info("Audio extracted for waveform using ffmpeg")
            return audio_path
        else:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return None
    except Exception as e:
        logger.warning(f"ffmpeg waveform extraction error: {e}")
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
        return None


def validate_upload_file(file: UploadFile) -> str:
    """
    Validate uploaded file - always requires video input

    Args:
        file: Uploaded file

    Returns:
        str: Always "video"

    Raises:
        UnsupportedFileTypeError: If file type is not a supported video format
    """
    if not file.filename:
        raise UnsupportedFileTypeError("", [], message="Filename is required")

    ext = Path(file.filename).suffix.lower()

    # Always require video input (for both ResNet and VAT models)
    allowed_extensions = settings.ALLOWED_VIDEO_EXTENSIONS
    if ext not in allowed_extensions:
        raise UnsupportedFileTypeError(
            ext,
            allowed_extensions,
            message=f"Video input required. Supported formats: {', '.join(allowed_extensions)}"
        )
    return "video"


async def get_prediction_service(
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionService:
    """Dependency to get prediction service"""
    return PredictionService(
        model_manager=model_manager,
        preprocessing_service=PreprocessingService()
    )


@router.post("/upload", response_model=PredictionResponse)
async def predict_from_upload(
    file: UploadFile = File(..., description="Video file for personality analysis"),
    method: Optional[str] = Query(
        default=None,
        description="Face detection method for ResNet: 'face', 'middle', or 'face+middle' (ignored for VAT)"
    ),
    use_tta: bool = Query(default=True, description="Use Test Time Augmentation"),
    include_interpretations: bool = Query(default=True, description="Include T-score interpretations"),
    generate_report: bool = Query(default=False, description="Generate AI narrative report"),
    question_responses: Optional[str] = Form(
        default=None,
        description="JSON string containing assessment question responses with timestamps (from gamified assessment)"
    ),
    prediction_service: PredictionService = Depends(get_prediction_service),
    cache_service: CacheService = Depends(get_cache_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    Predict personality from uploaded VIDEO file

    **Always requires video input** - supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm

    **Model Type determines prediction method:**

    - **ResNet** (MODEL_TYPE=resnet): Extracts 10 frames, applies face detection, averages predictions
    - **VAT** (MODEL_TYPE=vat): Extracts 32 frames using k-segment sampling

    **Test Time Augmentation (TTA):**
    Applies horizontal flip and averages predictions for better accuracy

    **AI Report Generation (Multimodal - both models):**
    When `generate_report=true`, the API will:
    - Extract 10 frames from the video for LLM visual analysis
    - Transcribe the audio using OpenAI's gpt-4o-transcribe
    - Generate personality insights based on visual and verbal cues
    """
    model_type = "VAT" if settings.is_vat_model else "ResNet"
    logger.info(f"Upload prediction request: {file.filename} (model={model_type}, report={generate_report})")

    temp_video_path = None
    compressed_video_path = None
    compression_metadata = None
    try:
        # Validate file - always requires video
        validate_upload_file(file)

        # Read file content
        content = await file.read()
        original_size = len(content)

        # Check file size
        if original_size > settings.MAX_UPLOAD_SIZE:
            raise FileSizeExceededError(settings.MAX_UPLOAD_SIZE)

        # Generate hash for caching
        content_hash = hashlib.md5(content).hexdigest()

        # Use effective face detection method for ResNet
        preprocessing_method = method or settings.effective_face_detection

        # Check cache (include report flag in cache key)
        cache_key_params = {"use_tta": use_tta, "report": generate_report, "method": preprocessing_method}
        cached_result = cache_service.get_prediction(content_hash, **cache_key_params)

        if cached_result:
            logger.info(f"Returning cached prediction for {file.filename}")
            return PredictionResponse(**cached_result)

        # Save to storage (optional)
        if settings.STORAGE_BACKEND == "local":
            try:
                storage_path = storage_service.save_upload(file.filename, content, subfolder="uploads")
                logger.info(f"Saved upload to: {storage_path}")
            except Exception as e:
                logger.warning(f"Failed to save upload: {e}")

        # Determine video extension
        ext = ".mp4"
        if file.filename:
            path = Path(file.filename)
            if path.suffix.lower() in settings.ALLOWED_VIDEO_EXTENSIONS:
                ext = path.suffix.lower()

        # Save video to temp file (needed for both prediction and multimodal)
        fd, temp_video_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(temp_video_path, 'wb') as f:
            f.write(content)

        # Auto-compress video if it exceeds threshold
        if (settings.VIDEO_COMPRESSION_ENABLED and
            original_size > settings.VIDEO_COMPRESSION_THRESHOLD):
            logger.info(
                f"Video size ({original_size / 1024 / 1024:.1f}MB) exceeds threshold "
                f"({settings.VIDEO_COMPRESSION_THRESHOLD / 1024 / 1024:.1f}MB), compressing..."
            )
            try:
                compressed_video_path, compression_metadata = compress_video(
                    input_path=temp_video_path,
                    target_bitrate=settings.VIDEO_COMPRESSION_TARGET_BITRATE,
                    max_resolution=settings.VIDEO_COMPRESSION_MAX_RESOLUTION,
                    audio_bitrate=settings.VIDEO_COMPRESSION_AUDIO_BITRATE
                )
                # Use compressed video for processing
                temp_video_path, compressed_video_path = compressed_video_path, temp_video_path
                logger.info(f"Using compressed video for processing")
            except Exception as e:
                logger.warning(f"Video compression failed, using original: {e}")
                compression_metadata = {"error": str(e), "used_original": True}

        multimodal_data = None
        audio_analysis_data = None
        result = None
        debug_raw_frames = None  # For debug visualization
        assessment_metadata = None  # For gamified assessment

        # Parse question_responses if provided (gamified assessment)
        if question_responses:
            try:
                assessment_data = json.loads(question_responses)
                # Validate with Pydantic schema
                assessment_metadata = AssessmentMetadata(**assessment_data)
                logger.info(f"Assessment metadata received: {assessment_metadata.questions_answered} answered, {assessment_metadata.questions_skipped} skipped")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse question_responses JSON: {e}")
            except Exception as e:
                logger.warning(f"Failed to validate assessment metadata: {e}")

        # Optimized parallel pipeline when report is requested with ResNet
        if generate_report and not settings.is_vat_model:
            try:
                # Run ResNet prediction, audio transcription, and audio analysis IN PARALLEL
                # ResNet extracts frames -> shared with multimodal service
                multimodal_service = get_multimodal_service()

                # Start all tasks concurrently
                resnet_task = prediction_service.predict_from_video_resnet_async(
                    video_path=temp_video_path,
                    use_tta=use_tta,
                    preprocessing_method=preprocessing_method,
                    return_raw_frames=True  # Get raw frames for multimodal reuse
                )
                audio_transcribe_task = multimodal_service.transcribe_audio_only_async(temp_video_path)
                audio_analysis_task = run_audio_analysis_async(temp_video_path)

                # Wait for all to complete in parallel
                resnet_result, audio_result, audio_analysis_result = await asyncio.gather(
                    resnet_task,
                    audio_transcribe_task,
                    audio_analysis_task,
                    return_exceptions=True
                )

                # Handle ResNet result
                if isinstance(resnet_result, Exception):
                    logger.error(f"ResNet prediction failed: {resnet_result}")
                    raise resnet_result
                result = resnet_result

                # Handle audio analysis result
                if isinstance(audio_analysis_result, Exception):
                    logger.warning(f"Audio analysis failed: {audio_analysis_result}")
                elif audio_analysis_result:
                    audio_analysis_data = audio_analysis_result
                    logger.info(f"Audio analysis complete: {len(audio_analysis_data.get('normalized_metrics', {}))} metrics")

                # Get raw frames from ResNet result for multimodal (keep a copy for debug)
                raw_frames = result.pop("raw_frames", None)
                debug_raw_frames = raw_frames  # Keep for debug visualization

                # Build multimodal data using shared frames + audio transcript
                if raw_frames:
                    # Convert frames to base64 for LLM
                    from PIL import Image
                    frames_base64 = multimodal_service.frames_to_base64(
                        [Image.fromarray(f) for f in raw_frames]
                    )
                    multimodal_data = {
                        "frames_base64": frames_base64,
                        "transcript": "",
                        "metadata": {
                            "num_frames_extracted": len(raw_frames),
                            "has_audio": False,
                            "transcript_length": 0,
                            "used_shared_frames": True,
                            "parallel_execution": True
                        }
                    }

                    # Add audio transcript if available
                    if not isinstance(audio_result, Exception) and audio_result:
                        multimodal_data["transcript"] = audio_result.get("transcript", "")
                        multimodal_data["metadata"]["has_audio"] = audio_result.get("metadata", {}).get("has_audio", False)
                        multimodal_data["metadata"]["transcript_length"] = len(audio_result.get("transcript", ""))
                    else:
                        logger.warning(f"Audio transcription failed or empty: {audio_result}")

                logger.info(f"Parallel pipeline complete: ResNet + Audio transcription + Audio analysis in parallel")

            except Exception as e:
                logger.error(f"Parallel pipeline failed, falling back to sequential: {e}", exc_info=True)
                # Fallback to sequential processing
                result = None
                multimodal_data = None
                audio_analysis_data = None

        # Fallback: Sequential processing (VAT model or parallel failed)
        if result is None:
            if generate_report:
                try:
                    # Run multimodal analysis (extract frames and transcribe)
                    multimodal_service = get_multimodal_service()
                    multimodal_data = await multimodal_service.analyze_video_async(temp_video_path)
                    logger.info(f"Multimodal analysis complete: {multimodal_data.get('metadata', {})}")
                except Exception as e:
                    logger.warning(f"Multimodal analysis failed, falling back to score-based: {e}")
                    multimodal_data = None

                # Run audio analysis if not already done
                if audio_analysis_data is None:
                    try:
                        audio_analysis_data = await run_audio_analysis_async(temp_video_path)
                        if audio_analysis_data:
                            logger.info(f"Audio analysis complete (sequential)")
                    except Exception as e:
                        logger.warning(f"Audio analysis failed: {e}")

            # Run video prediction
            result = prediction_service.predict_from_video_bytes(
                video_bytes=content,
                filename=file.filename,
                use_tta=use_tta
            )

        # Add interpretations and optional AI report (with multimodal data for both models)
        if include_interpretations:
            # Debug logging for multimodal data
            if generate_report:
                if multimodal_data:
                    frames_count = len(multimodal_data.get("frames_base64", []))
                    logger.info(f"Passing multimodal_data to interpretations: frames={frames_count}, has_transcript={bool(multimodal_data.get('transcript'))}")

                    # Add audio metrics to multimodal_data for LLM analysis
                    if audio_analysis_data:
                        multimodal_data["audio_metrics"] = audio_analysis_data
                        logger.info(f"Added audio_metrics to multimodal_data for LLM")

                    # Add assessment metadata (gamified questions) to multimodal_data for LLM
                    if assessment_metadata:
                        multimodal_data["assessment_metadata"] = assessment_metadata.model_dump()
                        logger.info(f"Added assessment_metadata to multimodal_data for LLM: {assessment_metadata.questions_answered} questions")
                else:
                    logger.warning("generate_report=True but multimodal_data is None!")

            result = await prediction_service.add_interpretations_async(
                result,
                include_summary=True,
                generate_report=generate_report,
                multimodal_data=multimodal_data
            )

        # Add audio metrics if available
        if audio_analysis_data:
            audio_metrics = format_audio_metrics_for_response(audio_analysis_data)
            if audio_metrics:
                result["audio_metrics"] = audio_metrics
                # Also add raw metrics to metadata for debugging/analysis
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["audio_analysis"] = {
                    "raw_metrics": audio_analysis_data.get("raw_metrics", {}),
                    "normalized_metrics": audio_analysis_data.get("normalized_metrics", {})
                }

        # Add compression metadata if video was compressed
        if compression_metadata:
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["compression"] = compression_metadata

        # Add debug visualization data (frames, faces, transcript, waveform)
        if debug_raw_frames or multimodal_data:
            try:
                transcript = multimodal_data.get("transcript", "") if multimodal_data else ""
                debug_data = extract_debug_visualization(
                    video_path=temp_video_path,
                    raw_frames=debug_raw_frames,
                    transcript=transcript,
                    audio_analysis=audio_analysis_data
                )
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["debug_visualization"] = debug_data
                logger.info(f"Debug visualization added: {len(debug_data.get('frames', []))} frames")
            except Exception as e:
                logger.warning(f"Failed to extract debug visualization: {e}")

        # Cache result
        cache_service.set_prediction(content_hash, result, **cache_key_params)

        return PredictionResponse(**result)

    except (FileUploadError, FileSizeExceededError, UnsupportedFileTypeError):
        raise
    except Exception as e:
        logger.error(f"Upload prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Clean up temp video files
        for path in [temp_video_path, compressed_video_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        # Force garbage collection to free memory after video processing
        gc.collect()


@router.post("/file", response_model=PredictionResponse)
async def predict_from_file_path(
    file_path: str = Query(..., description="Path to video file on server"),
    method: Optional[str] = Query(
        default=None,
        description="Face detection method for ResNet: 'face', 'middle', or 'face+middle' (ignored for VAT)"
    ),
    use_tta: bool = Query(default=True, description="Use Test Time Augmentation"),
    include_interpretations: bool = Query(default=True, description="Include T-score interpretations"),
    generate_report: bool = Query(default=False, description="Generate AI narrative report"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict personality from video file path (server-side files only)

    Useful for batch processing or when files are already on the server.

    **Always requires video input** - supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm

    **Model Type determines prediction method:**

    - **ResNet** (MODEL_TYPE=resnet): Extracts 10 frames, applies face detection, averages predictions
    - **VAT** (MODEL_TYPE=vat): Extracts 32 frames using k-segment sampling

    **AI Report Generation (Multimodal - both models):**
    When `generate_report=true`, extracts 10 frames and transcript for LLM analysis.
    """
    model_type = "VAT" if settings.is_vat_model else "ResNet"
    logger.info(f"File path prediction request: {file_path} (model={model_type}, report={generate_report})")

    # Validate path exists
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}"
        )

    ext = path.suffix.lower()

    # Always require video input
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video input required. Supported formats: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}"
        )

    multimodal_data = None
    result = None

    # Optimized parallel pipeline when report is requested with ResNet
    if generate_report and not settings.is_vat_model:
        try:
            # Run ResNet prediction and audio transcription IN PARALLEL
            multimodal_service = get_multimodal_service()

            # Start both tasks concurrently
            resnet_task = prediction_service.predict_from_video_resnet_async(
                video_path=str(path),
                use_tta=use_tta,
                preprocessing_method=method,
                return_raw_frames=True
            )
            audio_task = multimodal_service.transcribe_audio_only_async(str(path))

            # Wait for both to complete in parallel
            resnet_result, audio_result = await asyncio.gather(
                resnet_task,
                audio_task,
                return_exceptions=True
            )

            # Handle ResNet result
            if isinstance(resnet_result, Exception):
                logger.error(f"ResNet prediction failed: {resnet_result}")
                raise resnet_result
            result = resnet_result

            # Get raw frames from ResNet result for multimodal
            raw_frames = result.pop("raw_frames", None)

            # Build multimodal data using shared frames + audio transcript
            if raw_frames:
                from PIL import Image
                frames_base64 = multimodal_service.frames_to_base64(
                    [Image.fromarray(f) for f in raw_frames]
                )
                multimodal_data = {
                    "frames_base64": frames_base64,
                    "transcript": "",
                    "metadata": {
                        "num_frames_extracted": len(raw_frames),
                        "has_audio": False,
                        "transcript_length": 0,
                        "used_shared_frames": True,
                        "parallel_execution": True
                    }
                }

                # Add audio transcript if available
                if not isinstance(audio_result, Exception) and audio_result:
                    multimodal_data["transcript"] = audio_result.get("transcript", "")
                    multimodal_data["metadata"]["has_audio"] = audio_result.get("metadata", {}).get("has_audio", False)
                    multimodal_data["metadata"]["transcript_length"] = len(audio_result.get("transcript", ""))
                else:
                    logger.warning(f"Audio transcription failed or empty: {audio_result}")

            logger.info(f"Parallel pipeline complete: ResNet + Audio in parallel, frames shared")

        except Exception as e:
            logger.warning(f"Parallel pipeline failed, falling back to sequential: {e}")
            result = None
            multimodal_data = None

    # Fallback: Sequential processing (VAT model or parallel failed)
    if result is None:
        if generate_report:
            try:
                multimodal_service = get_multimodal_service()
                multimodal_data = await multimodal_service.analyze_video_async(str(path))
                logger.info(f"Multimodal analysis complete: {multimodal_data.get('metadata', {})}")
            except Exception as e:
                logger.warning(f"Multimodal analysis failed, falling back to score-based: {e}")
                multimodal_data = None

        # Run video prediction based on model type
        if settings.is_vat_model:
            result = prediction_service.predict_from_video(
                video_path=str(path),
                use_tta=use_tta
            )
        else:
            result = prediction_service.predict_from_video_resnet(
                video_path=str(path),
                use_tta=use_tta,
                preprocessing_method=method
            )

    # Add interpretations and optional AI report (with multimodal data for both models)
    if include_interpretations:
        result = await prediction_service.add_interpretations_async(
            result,
            include_summary=True,
            generate_report=generate_report,
            multimodal_data=multimodal_data
        )

    return PredictionResponse(**result)


@router.get("/demo")
async def demo_prediction():
    """
    Demo endpoint with example response
    Shows what a typical prediction looks like based on the configured model type

    **Both models always require VIDEO input** - the API extracts frames appropriately:
    - VAT: Extracts 32 frames using k-segment sampling
    - ResNet: Extracts 10 frames, applies face detection, averages predictions
    """
    model_type = settings.MODEL_TYPE

    if settings.is_vat_model:
        return {
            "success": True,
            "model_type": "vat",
            "input_type": "video",
            "predictions": {
                "openness": 0.65,
                "conscientiousness": 0.78,
                "extraversion": 0.45,
                "agreeableness": 0.52,
                "neuroticism": -0.23
            },
            "metadata": {
                "preprocessing": {
                    "total_frames": 300,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                    "duration_seconds": 10.0,
                    "num_frames_extracted": 32,
                    "extraction_method": "k_segment"
                },
                "prediction": {
                    "model": "vat",
                    "num_frames": 32,
                    "tta_used": True,
                    "device": "cuda"
                }
            },
            "note": "VAT model active. Use POST /predict/upload with a VIDEO file (.mp4, .avi, .mov, etc.)"
        }
    else:
        return {
            "success": True,
            "model_type": "resnet",
            "input_type": "video",
            "predictions": {
                "extraversion": 0.45,
                "neuroticism": -0.23,
                "agreeableness": 0.52,
                "conscientiousness": 0.78,
                "openness": 0.65
            },
            "metadata": {
                "preprocessing": {
                    "preprocessing_method": "resnet_multi_frame",
                    "method": "face+middle",
                    "num_frames_requested": 10,
                    "num_frames_extracted": 10,
                    "num_frames_processed": 10,
                    "faces_detected_count": 8,
                    "face_detection_rate": 0.8,
                    "video_duration": 10.0,
                    "video_fps": 30.0,
                    "original_size": [640, 480],
                    "processed_size": [256, 256]
                },
                "prediction": {
                    "model": "resnet",
                    "backbone": "resnet18",
                    "image_size": 256,
                    "tta_used": True,
                    "device": "cuda",
                    "num_frames_predicted": 10,
                    "aggregation_method": "mean",
                    "per_trait_std": {
                        "extraversion": 0.05,
                        "neuroticism": 0.03,
                        "agreeableness": 0.04,
                        "conscientiousness": 0.02,
                        "openness": 0.06
                    }
                }
            },
            "note": "ResNet model active. Use POST /predict/upload with a VIDEO file (.mp4, .avi, .mov, etc.). 10 frames extracted and averaged for prediction."
        }
