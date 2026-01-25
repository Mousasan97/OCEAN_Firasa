"""
Image and video preprocessing service with face detection
"""
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import Literal, Tuple, Optional, List

import torch

from src.core.face_detector import FaceDetector
from src.core.video_processor import VideoProcessor
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import (
    InvalidImageError,
    InvalidVideoError,
    FaceDetectionError,
    PreprocessingError
)

logger = get_logger(__name__)


class PreprocessingService:
    """
    Preprocessing service for images and videos
    Handles face detection, cropping, and format conversions
    """

    def __init__(self):
        """Initialize preprocessing service"""
        self.face_detector = FaceDetector()
        self.video_processor = VideoProcessor()
        logger.info("PreprocessingService initialized")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file

        Args:
            image_path: Path to image file

        Returns:
            Image as RGB numpy array

        Raises:
            InvalidImageError: If image cannot be loaded
        """
        try:
            pil_image = Image.open(image_path).convert("RGB")
            image_rgb = np.array(pil_image)
            logger.debug(f"Loaded image: {image_rgb.shape}")
            return image_rgb
        except Exception as e:
            raise InvalidImageError(f"Failed to load image: {str(e)}")

    def load_video_frame(self, video_path: str) -> np.ndarray:
        """
        Load middle frame from video

        Args:
            video_path: Path to video file

        Returns:
            Frame as RGB numpy array

        Raises:
            InvalidVideoError: If video cannot be loaded
        """
        try:
            frame_rgb = self.video_processor.extract_middle_frame(video_path)
            logger.debug(f"Loaded video frame: {frame_rgb.shape}")
            return frame_rgb
        except Exception as e:
            raise InvalidVideoError(f"Failed to load video: {str(e)}")

    def preprocess_image(
        self,
        image_rgb: np.ndarray,
        method: Literal["face", "middle", "face+middle"] = "face+middle",
        expand_ratio: Optional[float] = None
    ) -> Tuple[Image.Image, dict]:
        """
        Preprocess image with face detection

        Args:
            image_rgb: RGB image as numpy array
            method: Preprocessing method
                - "face": Require face detection, error if no face
                - "middle": Use full image, no face detection
                - "face+middle": Try face detection, fallback to full image
            expand_ratio: Bounding box expansion ratio (uses config if None)

        Returns:
            Tuple of (PIL Image, metadata dict)

        Raises:
            FaceDetectionError: If face required but not detected
            PreprocessingError: If preprocessing fails
        """
        try:
            expand_ratio = expand_ratio or settings.FACE_EXPAND_RATIO

            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            img_h, img_w = image_bgr.shape[:2]

            processed_image = image_bgr
            metadata = {
                "original_size": (img_w, img_h),
                "method": method,
                "face_detected": False,
                "face_bbox": None,
                "processed_size": None
            }

            # Try face detection if requested
            if method in ["face", "face+middle"]:
                try:
                    bbox = self.face_detector.detect_largest_face(image_bgr)

                    if bbox is not None:
                        # Crop face region
                        face_crop = self.face_detector.crop_face(
                            image_bgr,
                            bbox,
                            expand_ratio=expand_ratio
                        )

                        processed_image = face_crop
                        metadata["face_detected"] = True
                        metadata["face_bbox"] = {
                            "x": bbox.x,
                            "y": bbox.y,
                            "width": bbox.width,
                            "height": bbox.height
                        }
                        logger.info(f"Face detected and cropped: {bbox.width}x{bbox.height}")

                    else:
                        # No face detected
                        if method == "face":
                            raise FaceDetectionError("No face detected in image")
                        else:
                            logger.warning("No face detected, using full image")

                except FaceDetectionError:
                    raise
                except Exception as e:
                    logger.error(f"Face detection failed: {e}")
                    if method == "face":
                        raise FaceDetectionError(f"Face detection failed: {str(e)}")
                    logger.warning("Using full image as fallback")

            # Convert back to PIL Image (RGB)
            processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            metadata["processed_size"] = processed_pil.size

            # Add base64-encoded face image if face was detected
            if metadata["face_detected"]:
                buffered = BytesIO()
                processed_pil.save(buffered, format="JPEG", quality=85)
                face_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                metadata["face_image_base64"] = f"data:image/jpeg;base64,{face_base64}"
                logger.debug("Face image encoded as base64")

            logger.info(f"Preprocessing completed: {metadata['original_size']} -> {metadata['processed_size']}")
            return processed_pil, metadata

        except (FaceDetectionError, PreprocessingError):
            raise
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            raise PreprocessingError(f"Preprocessing failed: {str(e)}")

    def preprocess_from_file(
        self,
        file_path: str,
        method: Optional[str] = None
    ) -> Tuple[Image.Image, dict]:
        """
        Preprocess image or video from file

        Args:
            file_path: Path to image or video file
            method: Preprocessing method (uses config if None)

        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        method = method or settings.FACE_DETECTION_METHOD
        path = Path(file_path)

        # Determine if file is video
        is_video = path.suffix.lower() in settings.ALLOWED_VIDEO_EXTENSIONS

        if is_video:
            logger.info(f"Processing video: {file_path}")
            image_rgb = self.load_video_frame(str(path))
        else:
            logger.info(f"Processing image: {file_path}")
            image_rgb = self.load_image(str(path))

        # Preprocess
        processed_image, metadata = self.preprocess_image(image_rgb, method=method)
        metadata["file_path"] = str(path)
        metadata["file_type"] = "video" if is_video else "image"

        return processed_image, metadata

    def preprocess_from_bytes(
        self,
        image_bytes: bytes,
        method: Optional[str] = None
    ) -> Tuple[Image.Image, dict]:
        """
        Preprocess image from bytes (for API uploads)

        Args:
            image_bytes: Image data as bytes
            method: Preprocessing method (uses config if None)

        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        method = method or settings.FACE_DETECTION_METHOD

        try:
            # Load image from bytes
            from io import BytesIO
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_rgb = np.array(pil_image)

            # Preprocess
            processed_image, metadata = self.preprocess_image(image_rgb, method=method)
            metadata["file_type"] = "upload"

            return processed_image, metadata

        except Exception as e:
            raise InvalidImageError(f"Failed to load image from bytes: {str(e)}")

    def preprocess_video_for_vat(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
        image_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Preprocess video for VAT model using K-segment sampling.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (uses config if None)
            image_size: Frame size (uses config if None)

        Returns:
            Tuple of:
                - torch.Tensor: [num_frames, 3, image_size, image_size]
                - dict: Video metadata

        Raises:
            InvalidVideoError: If video cannot be processed
        """
        num_frames = num_frames or settings.MODEL_NUM_FRAMES
        image_size = image_size or settings.MODEL_IMAGE_SIZE

        try:
            logger.info(f"Preprocessing video for VAT: {video_path}")

            frames, metadata = self.video_processor.extract_frames_k_segment(
                video_path,
                num_frames=num_frames,
                image_size=image_size
            )

            metadata["file_path"] = str(video_path)
            metadata["file_type"] = "video"
            metadata["preprocessing_method"] = "vat_k_segment"

            logger.info(
                f"Video preprocessed: {frames.shape} "
                f"(duration: {metadata.get('duration_seconds', 0):.2f}s)"
            )

            return frames, metadata

        except Exception as e:
            logger.error(f"Video preprocessing failed: {e}", exc_info=True)
            raise InvalidVideoError(f"Failed to preprocess video: {str(e)}")

    def preprocess_video_for_resnet(
        self,
        video_path: str,
        method: Optional[str] = None,
        num_frames: Optional[int] = None,
        return_raw_frames: bool = False,
        skip_no_face: bool = True
    ) -> Tuple[List[Image.Image], dict, Optional[List[np.ndarray]]]:
        """
        Preprocess video for ResNet model by extracting multiple frames and applying face detection.

        Extracts evenly-spaced frames from the video and applies face detection to each.
        Only frames with detected faces are used for prediction (when skip_no_face=True).
        The predictions from valid frames are averaged for more robust results.

        Args:
            video_path: Path to video file
            method: Face detection method (uses config if None)
            num_frames: Number of frames to extract (uses config if None)
            return_raw_frames: If True, also return raw frames for multimodal reuse
            skip_no_face: If True, only include frames where face was detected (default True)

        Returns:
            Tuple of:
                - List[PIL.Image]: List of preprocessed frames (only face-detected if skip_no_face=True)
                - dict: Video and preprocessing metadata
                - Optional[List[np.ndarray]]: Raw frames if return_raw_frames=True, else None

        Raises:
            InvalidVideoError: If video cannot be processed
        """
        method = method or settings.effective_face_detection
        num_frames = num_frames or settings.RESNET_NUM_FRAMES

        try:
            logger.info(f"Preprocessing video for ResNet: {video_path} (extracting {num_frames} frames, skip_no_face={skip_no_face})")

            # Get video metadata
            video_metadata = self.video_processor.get_video_info(video_path)

            # Extract multiple evenly-spaced frames
            raw_frames = self.video_processor.extract_multiple_frames(video_path, num_frames)

            if not raw_frames:
                raise InvalidVideoError("No frames could be extracted from video")

            # Process each frame with face detection
            processed_images = []
            frames_metadata = []
            faces_detected = 0
            skipped_no_face = 0

            for i, frame_rgb in enumerate(raw_frames):
                try:
                    processed_image, preprocess_metadata = self.preprocess_image(
                        frame_rgb,
                        method=method
                    )

                    face_detected = preprocess_metadata.get('face_detected', False)

                    if face_detected:
                        faces_detected += 1
                        processed_images.append(processed_image)
                        frames_metadata.append(preprocess_metadata)
                    elif not skip_no_face:
                        # Include frames without face detection if skip_no_face is False
                        processed_images.append(processed_image)
                        frames_metadata.append(preprocess_metadata)
                    else:
                        # Skip frames without face detection
                        skipped_no_face += 1
                        logger.debug(f"Skipping frame {i}: no face detected")

                except Exception as e:
                    logger.warning(f"Failed to process frame {i}: {e}")
                    # Skip failed frames
                    continue

            if not processed_images:
                raise InvalidVideoError(
                    f"No valid frames for prediction. "
                    f"Extracted {len(raw_frames)} frames but no faces detected. "
                    f"Try a video with clearer frontal face views."
                )

            # Use the first frame's metadata as base, but include aggregate info
            base_metadata = frames_metadata[0] if frames_metadata else {}

            # Combine metadata
            metadata = {
                **video_metadata,
                **base_metadata,
                "file_path": str(video_path),
                "file_type": "video",
                "preprocessing_method": "resnet_multi_frame",
                "num_frames_requested": num_frames,
                "num_frames_extracted": len(raw_frames),
                "num_frames_processed": len(processed_images),
                "faces_detected_count": faces_detected,
                "frames_skipped_no_face": skipped_no_face,
                "skip_no_face_enabled": skip_no_face,
                "face_detection_rate": faces_detected / len(raw_frames) if raw_frames else 0
            }

            logger.info(
                f"Video preprocessed for ResNet: {len(processed_images)} frames used for prediction "
                f"(faces detected: {faces_detected}/{len(raw_frames)}, skipped: {skipped_no_face})"
            )

            # Return raw frames if requested (for multimodal reuse)
            if return_raw_frames:
                return processed_images, metadata, raw_frames
            return processed_images, metadata, None

        except Exception as e:
            logger.error(f"Video preprocessing for ResNet failed: {e}", exc_info=True)
            raise InvalidVideoError(f"Failed to preprocess video for ResNet: {str(e)}")
