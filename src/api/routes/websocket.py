"""
WebSocket route for real-time personality prediction
Handles streaming video frames and returning OCEAN scores in real-time
Now includes audio feature analysis for multimodal scoring
"""
import asyncio
import base64
import json
import time
from io import BytesIO
from typing import Optional, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

from src.services.model_manager import get_model_manager
from src.services.prediction_service import PredictionService
from src.services.preprocessing_service import PreprocessingService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


# ============================================
# Audio Feature → Trait Score Mapping
# ============================================

# Population norms for audio features
# Calibrated for Web Audio API's AnalyserNode output
AUDIO_NORMS = {
    "rms_energy": {"mean": 0.03, "std": 0.02},        # Web audio RMS is typically lower
    "energy_variance": {"mean": 0.01, "std": 0.008},  # Energy consistency
    "zcr": {"mean": 0.15, "std": 0.08},               # Zero crossing rate (higher in web audio)
    "spectral_centroid": {"mean": 1500, "std": 600},  # Brightness in Hz
    "spectral_spread": {"mean": 1000, "std": 400},    # Spread around centroid
}

# How much each audio feature influences each trait
# Weights are balanced to produce scores centered around 50
# Positive = higher feature → higher trait score
# Note: Using smaller weights to avoid extreme deviations from video scores
AUDIO_TRAIT_WEIGHTS = {
    "extraversion": {
        "rms_energy": 0.40,          # Louder = more extraverted
        "energy_variance": 0.20,     # More dynamic = expressive
        "spectral_centroid": 0.20,   # Brighter voice = energetic
        "spectral_spread": 0.20,     # More varied = animated
    },
    "neuroticism": {
        "energy_variance": 0.30,     # Unstable volume = anxious
        "spectral_spread": 0.30,     # Erratic spectral = nervous
        "zcr": 0.40,                 # High ZCR can indicate tension
    },
    "agreeableness": {
        # Inverted: lower energy variance = more agreeable (steady voice)
        # We achieve this by using positive weights but the raw feature inversely correlates
        "rms_energy": 0.30,          # Moderate projection = balanced
        "spectral_centroid": 0.35,   # Warmth matters
        "spectral_spread": 0.35,     # Consistency matters
    },
    "conscientiousness": {
        "rms_energy": 0.35,          # Clear projection = confident
        "spectral_centroid": 0.30,   # Measured speech
        "spectral_spread": 0.35,     # Controlled variation
    },
    "openness": {
        "spectral_spread": 0.35,     # Varied = creative expression
        "energy_variance": 0.25,     # Dynamic = expressive
        "spectral_centroid": 0.25,   # Brighter = animated
        "rms_energy": 0.15,          # Some projection
    }
}

# How much audio should influence final score (per trait)
# Keep these low so video (ResNet) remains the primary signal
# Audio provides a small adjustment, not a replacement
AUDIO_FUSION_WEIGHTS = {
    "extraversion": 0.15,      # Audio provides small boost/penalty
    "neuroticism": 0.10,       # Mostly visual
    "agreeableness": 0.10,     # Mostly visual
    "conscientiousness": 0.05, # Almost entirely visual
    "openness": 0.10,          # Mostly visual
}


def normalize_audio_feature(value: float, feature_name: str) -> float:
    """
    Normalize an audio feature to z-score using population norms.
    Returns value in roughly -2 to +2 range (standard deviations from mean).
    """
    if feature_name not in AUDIO_NORMS:
        return 0.0

    norm = AUDIO_NORMS[feature_name]
    if norm["std"] == 0:
        return 0.0

    z_score = (value - norm["mean"]) / norm["std"]
    # Clip to reasonable range
    return max(-3.0, min(3.0, z_score))


def audio_features_to_trait_scores(audio_features: Dict) -> Dict[str, float]:
    """
    Convert audio features to trait scores (0-100 scale).

    Uses weighted combination of normalized features.
    Returns scores centered around 50 (population mean).
    """
    if not audio_features or not audio_features.get("voice_active", False):
        # No voice detected - return None to skip audio fusion
        return None

    trait_scores = {}

    for trait, weights in AUDIO_TRAIT_WEIGHTS.items():
        score = 0.0
        total_weight = 0.0

        for feature, weight in weights.items():
            if feature in audio_features:
                # Get normalized feature value
                z_score = normalize_audio_feature(audio_features[feature], feature)

                # Apply weight (positive or negative)
                score += weight * z_score
                total_weight += abs(weight)

        if total_weight > 0:
            # Normalize by total weight
            score = score / total_weight

            # Convert z-score to 0-100 scale (50 = mean, ~15 per std)
            # A z-score of +1 → ~65, z-score of -1 → ~35
            trait_scores[trait] = round(50 + (score * 15), 1)

            # Clip to valid range
            trait_scores[trait] = max(0, min(100, trait_scores[trait]))
        else:
            trait_scores[trait] = 50.0  # Default to average

    return trait_scores


def fuse_video_audio_scores(
    video_scores: Dict[str, float],
    audio_scores: Optional[Dict[str, float]],
    voice_active: bool = False
) -> Dict[str, float]:
    """
    Fuse video and audio scores using trait-specific weights.

    If no audio (voice not active), returns video scores unchanged.
    """
    if audio_scores is None or not voice_active:
        return video_scores

    fused = {}
    for trait in video_scores:
        video_score = video_scores[trait]
        audio_score = audio_scores.get(trait, 50.0)

        # Get fusion weight for this trait
        audio_weight = AUDIO_FUSION_WEIGHTS.get(trait, 0.2)
        video_weight = 1.0 - audio_weight

        # Weighted average
        fused[trait] = round(video_weight * video_score + audio_weight * audio_score, 1)

    return fused


class RealtimePredictionHandler:
    """
    Handles real-time frame prediction over WebSocket.
    Reuses existing model and services for prediction.
    """

    def __init__(self):
        """Initialize handler with shared model and services."""
        self.model_manager = get_model_manager()
        self.preprocessing_service = PreprocessingService()
        self.prediction_service = PredictionService(
            model_manager=self.model_manager,
            preprocessing_service=self.preprocessing_service
        )
        self._predictor = None

    def _ensure_predictor(self):
        """Lazily initialize the predictor."""
        if self._predictor is None and self.model_manager.is_loaded:
            from src.core.personality_predictor import PersonalityPredictor
            self._predictor = PersonalityPredictor(
                model=self.model_manager.model,
                device=self.model_manager.device,
                trait_labels=self.model_manager.trait_labels,
                image_size=256  # ResNet uses 256x256
            )
        return self._predictor

    async def process_frame(
        self,
        frame_data: str,
        frame_index: int,
        audio_features: Optional[Dict] = None,
        include_debug: bool = False
    ) -> dict:
        """
        Process a single base64-encoded JPEG frame and return prediction.
        Optionally fuses with audio features if provided.

        Args:
            frame_data: Base64-encoded JPEG image data
            frame_index: Index of the frame for tracking
            audio_features: Optional audio features from browser
            include_debug: If True, include face image and bbox in response

        Returns:
            Dictionary with prediction results or error
        """
        start_time = time.perf_counter()

        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(frame_data)

            # Run prediction in thread pool (non-blocking)
            result = await asyncio.to_thread(
                self._predict_from_bytes,
                image_bytes
            )

            inference_time = (time.perf_counter() - start_time) * 1000

            # Get raw model predictions (0-1 scale from ResNet)
            predictions = result.get("predictions", {})

            # Convert predictions to percentages (0-100 scale)
            video_scores = {}
            for trait, value in predictions.items():
                # Model outputs are in 0-1 range, convert to percentage
                video_scores[trait] = round(value * 100, 1)

            # Get face detection status from metadata
            preprocessing_meta = result.get("metadata", {}).get("preprocessing", {})
            face_detected = preprocessing_meta.get("face_detected", False)

            # Process audio features if provided
            audio_scores = None
            voice_detected = False

            # DISABLED: Audio fusion temporarily disabled - using pure ResNet predictions
            # if audio_features:
            #     voice_detected = audio_features.get("voice_active", False)
            #     if voice_detected:
            #         audio_scores = audio_features_to_trait_scores(audio_features)

            # Use video scores directly (no audio fusion)
            final_scores = video_scores

            # Log for debugging
            if frame_index % 5 == 0:
                logger.info(f"[FRAME {frame_index}] Video scores (pure ResNet): {video_scores}")

            response = {
                "type": "prediction",
                "frame_index": frame_index,
                "scores": final_scores,
                "video_scores": video_scores,    # For debugging UI
                "audio_scores": audio_scores,    # For debugging UI (None if no voice)
                "face_detected": face_detected,
                "voice_detected": voice_detected,
                "audio_features": audio_features if audio_features else None,
                "inference_time_ms": round(inference_time, 1)
            }

            # Include debug info if requested (face image and bounding box)
            if include_debug:
                response["debug"] = {
                    "face_bbox": preprocessing_meta.get("face_bbox"),
                    "face_image": preprocessing_meta.get("face_image_base64"),
                    "original_size": preprocessing_meta.get("original_size"),
                    "processed_size": preprocessing_meta.get("processed_size")
                }

            return response

        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            return {
                "type": "error",
                "frame_index": frame_index,
                "message": str(e)
            }

    def _predict_from_bytes(self, image_bytes: bytes) -> dict:
        """
        Synchronous prediction from image bytes.

        Args:
            image_bytes: JPEG image as bytes

        Returns:
            Prediction result dictionary
        """
        return self.prediction_service.predict_from_bytes(
            image_bytes,
            use_tta=False,  # Disable TTA for speed
            preprocessing_method="face+middle"  # Keep face detection for accuracy
        )


@router.websocket("/realtime-predict")
async def websocket_realtime_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time personality prediction.

    Protocol:
    1. Client connects
    2. Server sends "ready" message with model info
    3. Client sends frames as JSON: {"type": "frame", "data": "<base64>", "frame_index": 0, "timestamp_ms": 123}
    4. Server responds with predictions: {"type": "prediction", "scores": {...}, "face_detected": true}
    5. Client sends "stop" to end session
    """
    await websocket.accept()
    logger.info("WebSocket client connected for real-time prediction")

    # Check if model is loaded
    model_manager = get_model_manager()
    if not model_manager.is_loaded:
        await websocket.send_json({
            "type": "error",
            "message": "Model not loaded. Please try again later."
        })
        await websocket.close()
        logger.warning("WebSocket closed: Model not loaded")
        return

    # Initialize handler
    handler = RealtimePredictionHandler()

    # Send ready message
    try:
        model_info = model_manager.get_model_info()
        await websocket.send_json({
            "type": "ready",
            "model_info": {
                "backbone": model_info.get("backbone", "resnet18"),
                "traits": model_info.get("trait_labels", []),
                "device": str(model_info.get("device", "cpu"))
            }
        })
    except Exception as e:
        logger.error(f"Error sending ready message: {e}")
        await websocket.send_json({
            "type": "ready",
            "model_info": {}
        })

    # Main message loop
    try:
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second timeout for inactivity
                )
                message = json.loads(data)
                msg_type = message.get("type", "")

                if msg_type == "frame":
                    # Process frame and send prediction
                    frame_data = message.get("data", "")
                    frame_index = message.get("frame_index", 0)
                    timestamp_ms = message.get("timestamp_ms", 0)
                    audio_features = message.get("audio_features", None)
                    include_debug = message.get("debug", False)  # Client can request debug info

                    if not frame_data:
                        await websocket.send_json({
                            "type": "error",
                            "frame_index": frame_index,
                            "message": "Empty frame data"
                        })
                        continue

                    # Process frame with optional audio features
                    result = await handler.process_frame(
                        frame_data,
                        frame_index,
                        audio_features=audio_features,
                        include_debug=include_debug
                    )
                    result["timestamp_ms"] = timestamp_ms

                    await websocket.send_json(result)

                elif msg_type == "stop":
                    logger.info("Client requested stop")
                    break

                elif msg_type == "ping":
                    # Heartbeat response
                    await websocket.send_json({"type": "pong"})

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })

            except asyncio.TimeoutError:
                # Send heartbeat to check if client is still connected
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except:
                    break

            except json.JSONDecodeError as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid JSON: {str(e)}"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass

    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info("WebSocket connection closed")
