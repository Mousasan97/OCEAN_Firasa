"""
Streaming routes with Server-Sent Events (SSE)

Provides real-time progress updates during video analysis.
The frontend can connect to these endpoints to receive progress updates
as the analysis pipeline processes the video.

Usage:
    POST /api/v1/stream/analyze

    Returns: text/event-stream with progress updates like:
    data: {"stage": "uploading", "progress": 0, "message": "Receiving video..."}
    data: {"stage": "compressing", "progress": 10, "message": "Compressing video..."}
    data: {"stage": "extracting_frames", "progress": 25, "message": "Extracting frames..."}
    data: {"stage": "transcribing", "progress": 40, "message": "Transcribing audio..."}
    data: {"stage": "analyzing_video", "progress": 55, "message": "Analyzing video frames..."}
    data: {"stage": "analyzing_audio", "progress": 70, "message": "Analyzing voice patterns..."}
    data: {"stage": "generating_report", "progress": 85, "message": "Generating insights..."}
    data: {"stage": "complete", "progress": 100, "message": "Analysis complete", "result": {...}}
"""
from fastapi import APIRouter, UploadFile, File, Query, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import Optional, AsyncGenerator
from pathlib import Path
import hashlib
import tempfile
import os
import asyncio
import json
import time

from src.api.schemas.response import PredictionResponse, AssessmentMetadata
from src.services.model_manager import ModelManager, get_model_manager
from src.services.prediction_service import PredictionService
from src.services.preprocessing_service import PreprocessingService
from src.services.multimodal_service import get_multimodal_service
from src.services.audio_analysis_service import get_audio_analysis_service
from src.infrastructure.cache import CacheService, get_cache_service
from src.infrastructure.storage import StorageService, get_storage_service
from src.core.video_processor import compress_video
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import FileSizeExceededError

logger = get_logger(__name__)
router = APIRouter(prefix="/stream", tags=["Streaming"])


async def get_prediction_service(
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionService:
    """Dependency to get prediction service"""
    return PredictionService(
        model_manager=model_manager,
        preprocessing_service=PreprocessingService()
    )


def create_progress_event(stage: str, progress: int, message: str, result: dict = None) -> str:
    """Create a JSON event for SSE streaming"""
    event = {
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": time.time()
    }
    if result is not None:
        event["result"] = result
    return json.dumps(event)


async def run_audio_analysis_async(video_path: str) -> dict:
    """Run audio analysis in a separate thread"""
    try:
        audio_service = get_audio_analysis_service()
        return await asyncio.to_thread(audio_service.analyze_video, video_path)
    except Exception as e:
        logger.warning(f"Audio analysis failed: {e}")
        return None


def format_audio_metrics_for_response(audio_analysis: dict) -> dict:
    """Format audio analysis results into the AudioMetrics response schema."""
    if not audio_analysis:
        return None

    indicators = audio_analysis.get("personality_indicators", {})

    vocal_ext = indicators.get("vocal_extraversion", {}).get("score", 50)
    vocal_expr = indicators.get("vocal_expressiveness", {}).get("score", 50)
    vocal_fluency = indicators.get("vocal_fluency", {}).get("score", 50)

    if vocal_ext > 65 and vocal_expr > 60:
        coach_rec = "Your voice projects energy and expressiveness."
    elif vocal_fluency > 65:
        coach_rec = "Your speech flows smoothly and confidently."
    else:
        coach_rec = "Your vocal patterns are balanced."

    steps = []
    if vocal_fluency < 50:
        steps.append({"emoji": "🎤", "text": "Practice speaking without filler words"})
    if vocal_expr < 50:
        steps.append({"emoji": "🎭", "text": "Add more pitch variation when speaking"})
    if len(steps) < 3:
        steps.extend([
            {"emoji": "🎯", "text": "Record yourself to identify patterns"},
            {"emoji": "⏸️", "text": "Use strategic pauses for emphasis"},
            {"emoji": "📣", "text": "Practice projecting your voice"}
        ][:3 - len(steps)])

    return {
        "indicators": indicators,
        "interpretations": audio_analysis.get("interpretations", {}),
        "coach_recommendation": coach_rec,
        "actionable_steps": steps
    }


def select_top_frames_per_segment(scores: dict, num_segments: int = 10) -> dict:
    """
    Divide the recording into segments and select the top frame from each segment.

    For each segment, the "top" frame is the one with the highest average OCEAN score,
    representing the most expressive/confident moment in that time window.

    Args:
        scores: Dictionary with trait arrays {"openness": [...], "extraversion": [...], ...}
        num_segments: Number of segments to divide the video into (default 10)

    Returns:
        Dictionary with selected frame indices and their scores
    """
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    # Get the number of frames (use first available trait)
    first_trait = next((t for t in traits if scores.get(t)), None)
    if not first_trait:
        return {"selected_indices": [], "scores": {}}

    n_frames = len(scores[first_trait])
    if n_frames == 0:
        return {"selected_indices": [], "scores": {}}

    # Calculate average score for each frame (across all traits)
    frame_avg_scores = []
    for i in range(n_frames):
        trait_scores = [scores[t][i] for t in traits if scores.get(t) and len(scores[t]) > i]
        if trait_scores:
            frame_avg_scores.append(sum(trait_scores) / len(trait_scores))
        else:
            frame_avg_scores.append(0)

    # Determine segment size
    actual_segments = min(num_segments, n_frames)
    segment_size = n_frames / actual_segments

    # Select top frame from each segment
    selected_indices = []
    for seg in range(actual_segments):
        start_idx = int(seg * segment_size)
        end_idx = int((seg + 1) * segment_size)
        if end_idx > n_frames:
            end_idx = n_frames

        # Find the frame with highest average score in this segment
        if start_idx < end_idx:
            segment_scores = [(i, frame_avg_scores[i]) for i in range(start_idx, end_idx)]
            best_frame = max(segment_scores, key=lambda x: x[1])
            selected_indices.append(best_frame[0])

    return {
        "selected_indices": selected_indices,
        "n_frames": n_frames,
        "n_segments": actual_segments
    }


def process_realtime_scores(realtime_scores_json: str) -> Optional[dict]:
    """
    Process real-time scores collected during recording.

    Uses segment-based selection: divides the recording into 10 segments
    and picks the best frame from each segment, then averages those top frames.

    Args:
        realtime_scores_json: JSON string containing scores history

    Returns:
        Dictionary with aggregated scores or None if invalid
    """
    try:
        data = json.loads(realtime_scores_json)
        scores = data.get("scores", {})
        sample_count = data.get("sampleCount", 0)

        # Need at least 5 samples for reliable aggregation
        if sample_count < 5:
            logger.info(f"Not enough realtime samples ({sample_count}), falling back to video extraction")
            return None

        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

        # Select top frames from each segment (10 segments by default)
        selection = select_top_frames_per_segment(scores, num_segments=10)
        selected_indices = selection["selected_indices"]

        if not selected_indices:
            logger.warning("No frames selected, falling back to video extraction")
            return None

        # Calculate average from selected top frames
        aggregated = {}
        for trait in traits:
            trait_values = scores.get(trait, [])
            if trait_values:
                # Get scores only from selected frames
                selected_scores = [trait_values[i] for i in selected_indices if i < len(trait_values)]
                if selected_scores:
                    # Real-time scores are in 0-100 scale, convert to 0-1 for consistency
                    avg_score = sum(selected_scores) / len(selected_scores)
                    aggregated[trait] = avg_score / 100.0
                else:
                    aggregated[trait] = 0.5
            else:
                aggregated[trait] = 0.5  # Default to middle if no data

        # Log detailed info for debugging
        logger.info(f"=== REALTIME SCORES DEBUG ===")
        logger.info(f"Total samples received: {sample_count}")
        logger.info(f"Segments: {len(selected_indices)}, indices: {selected_indices}")

        # Log raw input scores (0-100 scale) statistics
        for trait in traits:
            trait_values = scores.get(trait, [])
            if trait_values:
                logger.info(f"  {trait}: min={min(trait_values):.1f}, max={max(trait_values):.1f}, avg={sum(trait_values)/len(trait_values):.1f} (0-100 scale)")

        # Log selected frame scores
        logger.info(f"Selected frame scores (0-100 scale):")
        for trait in traits:
            trait_values = scores.get(trait, [])
            if trait_values:
                selected = [trait_values[i] for i in selected_indices if i < len(trait_values)]
                if selected:
                    logger.info(f"  {trait}: {selected} -> avg={sum(selected)/len(selected):.1f}")

        logger.info(f"Final aggregated scores (0-1 scale): {aggregated}")
        logger.info(f"=== END REALTIME SCORES DEBUG ===")

        return {
            "predictions": aggregated,
            "sample_count": sample_count,
            "selected_frames": len(selected_indices),
            "total_frames": selection["n_frames"],
            "source": "segment_top_selection"
        }

    except Exception as e:
        logger.warning(f"Failed to process realtime scores: {e}")
        return None


@router.post("/analyze")
async def analyze_with_progress(
    file: UploadFile = File(..., description="Video file for personality analysis"),
    method: Optional[str] = Query(default=None, description="Face detection method"),
    use_tta: bool = Query(default=True, description="Use Test Time Augmentation"),
    generate_report: bool = Query(default=True, description="Generate AI narrative report"),
    question_responses: Optional[str] = Form(default=None, description="Assessment question responses JSON"),
    realtime_scores: Optional[str] = Form(default=None, description="Real-time scores collected during recording"),
    model_manager: ModelManager = Depends(get_model_manager),
    cache_service: CacheService = Depends(get_cache_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """
    Analyze video with real-time progress streaming via SSE.

    Returns a Server-Sent Events stream with progress updates.
    The final event contains the complete analysis result.

    **Progress Stages:**
    - `uploading` (0-5%): Receiving and validating video
    - `compressing` (5-15%): Compressing video if needed
    - `extracting_frames` (15-30%): Extracting frames for analysis
    - `transcribing` (30-50%): Transcribing audio to text
    - `analyzing_video` (50-65%): Running OCEAN prediction on frames
    - `analyzing_audio` (65-75%): Analyzing voice patterns
    - `generating_report` (75-95%): Generating AI insights
    - `complete` (100%): Analysis complete with full result

    **Example Usage (JavaScript):**
    ```javascript
    const formData = new FormData();
    formData.append('file', videoFile);

    const eventSource = new EventSource('/api/v1/stream/analyze', {
        method: 'POST',
        body: formData
    });

    // Note: EventSource only supports GET. Use fetch with ReadableStream instead:
    const response = await fetch('/api/v1/stream/analyze', {
        method: 'POST',
        body: formData
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                console.log(`Progress: ${data.progress}% - ${data.message}`);

                if (data.stage === 'complete') {
                    console.log('Result:', data.result);
                }
            }
        }
    }
    ```
    """

    async def generate_progress_events() -> AsyncGenerator[str, None]:
        temp_video_path = None
        compressed_video_path = None

        try:
            # Stage 1: Uploading (0-5%)
            yield f"data: {create_progress_event('uploading', 0, 'Receiving video...')}\n\n"

            # Validate file type
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")

            ext = Path(file.filename).suffix.lower()
            if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported video format: {ext}")

            yield f"data: {create_progress_event('uploading', 3, 'Validating video format...')}\n\n"

            # Read file content
            content = await file.read()
            original_size = len(content)

            if original_size > settings.MAX_UPLOAD_SIZE:
                raise FileSizeExceededError(settings.MAX_UPLOAD_SIZE)

            yield f"data: {create_progress_event('uploading', 5, f'Video received ({original_size / 1024 / 1024:.1f}MB)')}\n\n"

            # Generate cache key
            content_hash = hashlib.md5(content).hexdigest()
            preprocessing_method = method or settings.effective_face_detection
            cache_key_params = {"use_tta": use_tta, "report": generate_report, "method": preprocessing_method}

            # Check cache
            cached_result = cache_service.get_prediction(content_hash, **cache_key_params)
            if cached_result:
                yield f"data: {create_progress_event('complete', 100, 'Analysis complete (cached)', cached_result)}\n\n"
                return

            # Save to temp file
            fd, temp_video_path = tempfile.mkstemp(suffix=ext)
            os.close(fd)
            with open(temp_video_path, 'wb') as f:
                f.write(content)

            # Stage 2: Compression (5-15%)
            if settings.VIDEO_COMPRESSION_ENABLED and original_size > settings.VIDEO_COMPRESSION_THRESHOLD:
                yield f"data: {create_progress_event('compressing', 8, 'Compressing video...')}\n\n"
                try:
                    compressed_video_path, _ = compress_video(
                        input_path=temp_video_path,
                        target_bitrate=settings.VIDEO_COMPRESSION_TARGET_BITRATE,
                        max_resolution=settings.VIDEO_COMPRESSION_MAX_RESOLUTION,
                        audio_bitrate=settings.VIDEO_COMPRESSION_AUDIO_BITRATE
                    )
                    temp_video_path, compressed_video_path = compressed_video_path, temp_video_path
                    yield f"data: {create_progress_event('compressing', 15, 'Video compressed')}\n\n"
                except Exception as e:
                    logger.warning(f"Compression failed: {e}")
                    yield f"data: {create_progress_event('compressing', 15, 'Using original video')}\n\n"
            else:
                yield f"data: {create_progress_event('compressing', 15, 'No compression needed')}\n\n"

            # Parse assessment metadata
            assessment_metadata = None
            if question_responses:
                try:
                    assessment_data = json.loads(question_responses)
                    assessment_metadata = AssessmentMetadata(**assessment_data)
                except Exception as e:
                    logger.warning(f"Failed to parse assessment metadata: {e}")

            # Create services
            prediction_service = PredictionService(
                model_manager=model_manager,
                preprocessing_service=PreprocessingService()
            )
            multimodal_service = get_multimodal_service()

            # Stage 3: Extract frames (15-30%)
            yield f"data: {create_progress_event('extracting_frames', 18, 'Extracting video frames...')}\n\n"
            await asyncio.sleep(0.1)  # Allow event to be sent
            yield f"data: {create_progress_event('extracting_frames', 25, 'Processing video data...')}\n\n"

            # Stage 4: Transcription (30-50%)
            yield f"data: {create_progress_event('transcribing', 32, 'Transcribing audio...')}\n\n"
            await asyncio.sleep(0.1)

            # Stage 5: Video analysis (50-65%) - This is the longest stage
            yield f"data: {create_progress_event('analyzing_video', 40, 'Starting video analysis...')}\n\n"

            # Check if we have valid realtime scores to use instead of re-extracting frames
            realtime_aggregated = None
            if realtime_scores:
                realtime_aggregated = process_realtime_scores(realtime_scores)
                if realtime_aggregated:
                    logger.info(f"Using {realtime_aggregated['sample_count']} realtime samples for final analysis")

            # Run parallel analysis with progress simulation
            # Create tasks - only run ResNet if we don't have realtime scores
            resnet_task = None
            if not realtime_aggregated:
                resnet_task = asyncio.create_task(
                    prediction_service.predict_from_video_resnet_async(
                        video_path=temp_video_path,
                        use_tta=use_tta,
                        preprocessing_method=preprocessing_method,
                        return_raw_frames=True
                    )
                )

            audio_transcribe_task = asyncio.create_task(
                multimodal_service.transcribe_audio_only_async(temp_video_path)
            )
            audio_analysis_task = asyncio.create_task(
                run_audio_analysis_async(temp_video_path)
            )

            # Send progress updates while waiting for tasks
            if realtime_aggregated:
                progress_messages = [
                    (45, 'Using real-time scores...'),
                    (50, 'Processing audio...'),
                    (55, 'Aggregating personality data...'),
                    (60, 'Finalizing predictions...'),
                ]
            else:
                progress_messages = [
                    (45, 'Analyzing facial expressions...'),
                    (50, 'Processing video frames...'),
                    (55, 'Extracting personality cues...'),
                    (60, 'Analyzing behavioral patterns...'),
                ]

            all_tasks = [t for t in [resnet_task, audio_transcribe_task, audio_analysis_task] if t is not None]
            msg_index = 0

            while not all(t.done() for t in all_tasks):
                # Send progress update
                if msg_index < len(progress_messages):
                    prog, msg = progress_messages[msg_index]
                    yield f"data: {create_progress_event('analyzing_video', prog, msg)}\n\n"
                    msg_index += 1
                await asyncio.sleep(1.5)  # Check every 1.5 seconds

            # Get results with exception handling
            resnet_result = None
            if resnet_task:
                try:
                    resnet_result = resnet_task.result()
                except Exception as e:
                    logger.error(f"ResNet prediction failed: {e}")
                    raise e

            try:
                audio_result = audio_transcribe_task.result()
            except Exception as e:
                logger.warning(f"Audio transcription failed: {e}")
                audio_result = None

            try:
                audio_analysis_result = audio_analysis_task.result()
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}")
                audio_analysis_result = None

            yield f"data: {create_progress_event('analyzing_video', 65, 'Video analysis complete')}\n\n"

            # Use realtime aggregated scores or ResNet result
            if realtime_aggregated:
                # Build result structure similar to ResNet result
                # Predictions are in 0-1 scale, T-score normalization happens in add_interpretations
                raw_predictions = realtime_aggregated["predictions"]

                result = {
                    "predictions": raw_predictions,
                    "metadata": {
                        "source": realtime_aggregated.get("source", "realtime_aggregation"),
                        "sample_count": realtime_aggregated["sample_count"],
                        "selected_frames": realtime_aggregated.get("selected_frames"),
                        "total_frames": realtime_aggregated.get("total_frames"),
                        "preprocessing": {
                            "method": "realtime_websocket",
                            "face_detected": True  # Assume face was detected during live recording
                        }
                    }
                }
                logger.info(f"Final result using {realtime_aggregated.get('selected_frames', 'N/A')} top frames from {realtime_aggregated.get('total_frames', 'N/A')} total (0-1 scale): {raw_predictions}")
            else:
                result = resnet_result

            # Stage 6: Audio analysis (65-75%)
            yield f"data: {create_progress_event('analyzing_audio', 70, 'Analyzing voice patterns...')}\n\n"

            audio_analysis_data = None
            if audio_analysis_result:
                audio_analysis_data = audio_analysis_result

            yield f"data: {create_progress_event('analyzing_audio', 75, 'Voice analysis complete')}\n\n"

            # Build multimodal data
            raw_frames = result.pop("raw_frames", None)
            multimodal_data = None
            pil_frames = None

            # If using realtime scores and need to generate report, extract frames separately
            if not raw_frames and realtime_aggregated and generate_report:
                try:
                    logger.info("Extracting frames for report (realtime scores mode)")
                    # Extract PIL frames for the AI report
                    pil_frames = await asyncio.to_thread(
                        multimodal_service.extract_frames_from_video,
                        temp_video_path
                    )
                except Exception as e:
                    logger.warning(f"Frame extraction for report failed: {e}")
                    pil_frames = None

            # Convert raw frames to PIL if needed
            if raw_frames and generate_report:
                from PIL import Image
                pil_frames = [Image.fromarray(f) for f in raw_frames]

            if pil_frames and generate_report:
                frames_base64 = multimodal_service.frames_to_base64(pil_frames)
                multimodal_data = {
                    "frames_base64": frames_base64,
                    "transcript": "",
                    "metadata": {
                        "num_frames_extracted": len(pil_frames),
                        "has_audio": False,
                        "transcript_length": 0
                    }
                }

                if not isinstance(audio_result, Exception) and audio_result:
                    multimodal_data["transcript"] = audio_result.get("transcript", "")
                    multimodal_data["metadata"]["has_audio"] = audio_result.get("metadata", {}).get("has_audio", False)
                    multimodal_data["metadata"]["transcript_length"] = len(audio_result.get("transcript", ""))

            # Stage 7: Generate report (75-95%)
            yield f"data: {create_progress_event('generating_report', 80, 'Generating personality insights...')}\n\n"

            # Add interpretations and AI report
            if multimodal_data:
                if audio_analysis_data:
                    multimodal_data["audio_metrics"] = audio_analysis_data
                if assessment_metadata:
                    multimodal_data["assessment_metadata"] = assessment_metadata.model_dump()

            yield f"data: {create_progress_event('generating_report', 85, 'Running AI analysis...')}\n\n"

            result = await prediction_service.add_interpretations_async(
                result,
                include_summary=True,
                generate_report=generate_report,
                multimodal_data=multimodal_data
            )

            yield f"data: {create_progress_event('generating_report', 95, 'Finalizing results...')}\n\n"

            # Format audio metrics - merge with LLM voice coaching if present
            logger.info(f"[STREAM VOICE DEBUG] audio_analysis_data exists: {bool(audio_analysis_data)}")
            logger.info(f"[STREAM VOICE DEBUG] result audio_metrics before merge: {result.get('audio_metrics')}")
            if audio_analysis_data:
                formatted_audio = format_audio_metrics_for_response(audio_analysis_data)
                # Preserve LLM coaching fields if they exist in result["audio_metrics"]
                if result.get("audio_metrics"):
                    llm_coaching_fields = ["coach_recommendation", "actionable_steps", "snapshot_insight",
                                          "behavioral_patterns", "how_others_experience", "strength",
                                          "tradeoff", "growth_lever", "suitable_for"]
                    for field in llm_coaching_fields:
                        if result["audio_metrics"].get(field):
                            formatted_audio[field] = result["audio_metrics"][field]
                            logger.info(f"[STREAM VOICE DEBUG] Merged field {field}: {result['audio_metrics'].get(field)}")
                result["audio_metrics"] = formatted_audio
                logger.info(f"[STREAM VOICE DEBUG] result audio_metrics after merge: {result.get('audio_metrics')}")

            # Add metadata
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["streaming"] = True

            # Include transcript for chat agent context
            if multimodal_data and multimodal_data.get("transcript"):
                result["user_transcript"] = multimodal_data["transcript"]

            # Cache result
            cache_service.set_prediction(content_hash, result, **cache_key_params)

            # Stage 8: Complete (100%)
            yield f"data: {create_progress_event('complete', 100, 'Analysis complete', result)}\n\n"

        except Exception as e:
            logger.error(f"Streaming analysis failed: {e}", exc_info=True)
            error_event = create_progress_event('error', -1, str(e))
            yield f"data: {error_event}\n\n"

        finally:
            # Cleanup temp files
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except Exception:
                    pass
            if compressed_video_path and os.path.exists(compressed_video_path):
                try:
                    os.unlink(compressed_video_path)
                except Exception:
                    pass

    return StreamingResponse(
        generate_progress_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/test")
async def test_sse():
    """Test SSE endpoint - sends progress updates every second"""

    async def generate_test_events():
        stages = [
            ("uploading", 5, "Receiving video..."),
            ("compressing", 15, "Compressing video..."),
            ("extracting_frames", 30, "Extracting frames..."),
            ("transcribing", 50, "Transcribing audio..."),
            ("analyzing_video", 65, "Analyzing video..."),
            ("analyzing_audio", 75, "Analyzing voice..."),
            ("generating_report", 90, "Generating insights..."),
            ("complete", 100, "Analysis complete")
        ]

        for stage, progress, message in stages:
            yield f"data: {create_progress_event(stage, progress, message)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate_test_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
