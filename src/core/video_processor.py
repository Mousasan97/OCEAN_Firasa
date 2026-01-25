"""
Video processing utilities
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import tempfile
import os
import subprocess
import shutil

import torch
from torchvision import transforms

from src.utils.logger import get_logger
from src.utils.exceptions import InvalidVideoError

logger = get_logger(__name__)


def compress_video(
    input_path: str,
    output_path: Optional[str] = None,
    target_bitrate: str = "1M",
    max_resolution: int = 720,
    audio_bitrate: str = "128k"
) -> Tuple[str, dict]:
    """
    Compress a video file to reduce size.

    Uses moviepy for compression with fallback to ffmpeg CLI.

    Args:
        input_path: Path to input video
        output_path: Path for output video (creates temp file if None)
        target_bitrate: Target video bitrate (e.g., "1M", "2M", "500k")
        max_resolution: Maximum height in pixels (e.g., 720, 480)
        audio_bitrate: Audio bitrate (e.g., "128k", "96k")

    Returns:
        Tuple of (output_path, compression_metadata)
    """
    from src.utils.config import settings

    input_path = str(input_path)
    original_size = os.path.getsize(input_path)

    # Create output path if not provided
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    logger.info(f"Compressing video: {original_size / 1024 / 1024:.1f}MB -> target bitrate {target_bitrate}")

    try:
        # Try moviepy first (already a dependency)
        compressed_path, metadata = _compress_with_moviepy(
            input_path, output_path, target_bitrate, max_resolution, audio_bitrate
        )
    except Exception as e:
        logger.warning(f"moviepy compression failed: {e}, trying ffmpeg CLI")
        try:
            # Fallback to ffmpeg CLI if available
            compressed_path, metadata = _compress_with_ffmpeg(
                input_path, output_path, target_bitrate, max_resolution, audio_bitrate
            )
        except Exception as e2:
            logger.error(f"Both compression methods failed: {e2}")
            raise ValueError(f"Video compression failed: {e2}")

    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = (1 - compressed_size / original_size) * 100

    metadata.update({
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio_percent": round(compression_ratio, 1),
        "target_bitrate": target_bitrate,
        "max_resolution": max_resolution
    })

    logger.info(
        f"Video compressed: {original_size / 1024 / 1024:.1f}MB -> "
        f"{compressed_size / 1024 / 1024:.1f}MB ({compression_ratio:.1f}% reduction)"
    )

    return compressed_path, metadata


def _compress_with_moviepy(
    input_path: str,
    output_path: str,
    target_bitrate: str,
    max_resolution: int,
    audio_bitrate: str
) -> Tuple[str, dict]:
    """Compress video using moviepy"""
    from moviepy import VideoFileClip

    clip = VideoFileClip(input_path)

    try:
        original_height = clip.h
        original_width = clip.w
        original_duration = clip.duration

        # Resize if needed
        if clip.h > max_resolution:
            scale_factor = max_resolution / clip.h
            new_width = int(clip.w * scale_factor)
            new_height = max_resolution
            clip = clip.resized(height=new_height)
            logger.info(f"Resized video: {original_width}x{original_height} -> {new_width}x{new_height}")
        else:
            new_width, new_height = original_width, original_height

        # Write compressed video
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate=target_bitrate,
            audio_bitrate=audio_bitrate,
            preset="medium",
            logger=None  # Suppress moviepy's verbose logging
        )

        metadata = {
            "method": "moviepy",
            "original_resolution": f"{original_width}x{original_height}",
            "output_resolution": f"{new_width}x{new_height}",
            "duration_seconds": original_duration
        }

        return output_path, metadata

    finally:
        clip.close()


def _compress_with_ffmpeg(
    input_path: str,
    output_path: str,
    target_bitrate: str,
    max_resolution: int,
    audio_bitrate: str
) -> Tuple[str, dict]:
    """Compress video using ffmpeg CLI (fallback)"""
    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found in PATH")

    # Build ffmpeg command
    cmd = [
        ffmpeg_path,
        "-y",  # Overwrite output
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-b:v", target_bitrate,
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-vf", f"scale=-2:'min({max_resolution},ih)'",  # Scale height to max, keep aspect ratio
        "-movflags", "+faststart",  # Optimize for streaming
        output_path
    ]

    logger.info(f"Running ffmpeg: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    metadata = {
        "method": "ffmpeg",
        "ffmpeg_path": ffmpeg_path
    }

    return output_path, metadata


class VideoProcessor:
    """Video frame extraction and processing"""

    @staticmethod
    def extract_middle_frame(video_path: str) -> np.ndarray:
        """
        Extract middle frame from video

        Args:
            video_path: Path to video file

        Returns:
            Frame as RGB numpy array

        Raises:
            InvalidVideoError: If video cannot be opened or read
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")

        try:
            # Get total frame count
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.debug(f"Video has {n_frames} frames")

            # Handle WebM and other formats with missing frame count metadata
            if n_frames <= 0 or n_frames > 100000:
                logger.info(f"Frame count unavailable ({n_frames}), reading frames sequentially for middle frame")
                return VideoProcessor._extract_middle_frame_sequential(cap)

            # Calculate middle frame index
            mid_idx = max(0, n_frames // 2)

            # Seek to middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)

            # Read frame
            ret, frame = cap.read()

            if not ret:
                # Fallback to first frame
                logger.warning("Could not read middle frame, falling back to first frame")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

                if not ret:
                    raise InvalidVideoError("Cannot read any frames from video")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            logger.info(f"Extracted frame {mid_idx}/{n_frames}: {frame_rgb.shape}")
            return frame_rgb

        finally:
            cap.release()

    @staticmethod
    def _extract_middle_frame_sequential(cap: cv2.VideoCapture) -> np.ndarray:
        """
        Extract middle frame by reading sequentially (for videos without frame count metadata).

        Args:
            cap: OpenCV VideoCapture object (already opened)

        Returns:
            Middle frame as RGB numpy array
        """
        logger.info("Reading video sequentially to find middle frame")

        # Read all frames to find the middle one
        all_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
            frame_idx += 1

            # Safety limit
            if frame_idx > 10000:
                logger.warning("Video too long, stopping at 10000 frames")
                break

        total_frames = len(all_frames)
        logger.info(f"Read {total_frames} total frames")

        if total_frames == 0:
            raise InvalidVideoError("Cannot read any frames from video")

        # Return middle frame
        mid_idx = total_frames // 2
        logger.info(f"Returning middle frame {mid_idx}/{total_frames}")
        return all_frames[mid_idx]

    @staticmethod
    def extract_frame_at_position(
        video_path: str,
        position: float = 0.5
    ) -> np.ndarray:
        """
        Extract frame at specific position in video

        Args:
            video_path: Path to video file
            position: Position in video (0.0 to 1.0)

        Returns:
            Frame as RGB numpy array
        """
        if not 0.0 <= position <= 1.0:
            raise ValueError("Position must be between 0.0 and 1.0")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")

        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = int(n_frames * position)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                raise InvalidVideoError(f"Cannot read frame at position {position}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.info(f"Extracted frame {frame_idx}/{n_frames} at position {position}")

            return frame_rgb

        finally:
            cap.release()

    @staticmethod
    def extract_multiple_frames(
        video_path: str,
        num_frames: int = 5
    ) -> list[np.ndarray]:
        """
        Extract multiple evenly-spaced frames from video

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract

        Returns:
            List of frames as RGB numpy arrays
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")

        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Handle WebM and other formats with missing frame count metadata
            # (e.g., browser-recorded WebM files have Duration: N/A)
            if n_frames <= 0 or n_frames > 100000:
                logger.info(f"Frame count unavailable ({n_frames}), reading frames sequentially")
                return VideoProcessor._extract_frames_sequential(cap, num_frames, fps)

            # Calculate frame indices
            if num_frames >= n_frames:
                indices = list(range(n_frames))
            else:
                indices = [int(i * n_frames / num_frames) for i in range(num_frames)]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            logger.info(f"Extracted {len(frames)} frames from video")
            return frames

        finally:
            cap.release()

    @staticmethod
    def _extract_frames_sequential(
        cap: cv2.VideoCapture,
        num_frames: int,
        fps: float
    ) -> list[np.ndarray]:
        """
        Extract frames by reading sequentially (for videos without frame count metadata).
        Used for WebM and other formats where CAP_PROP_FRAME_COUNT is unavailable.

        Args:
            cap: OpenCV VideoCapture object (already opened)
            num_frames: Number of frames to extract
            fps: Frames per second (may be 0 if unavailable)

        Returns:
            List of frames as RGB numpy arrays
        """
        logger.info(f"Sequential frame extraction: requesting {num_frames} frames, fps={fps}")

        # First pass: read ALL frames to get total count
        # For WebM from browser, we need to read everything first
        all_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
            frame_idx += 1

            # Safety limit: don't read more than 10000 frames
            if frame_idx > 10000:
                logger.warning("Video too long, stopping at 10000 frames")
                break

        total_frames = len(all_frames)
        logger.info(f"Read {total_frames} total frames from video")

        if total_frames == 0:
            raise InvalidVideoError("No frames could be extracted from video")

        # Select evenly-spaced frames from what we have
        if num_frames >= total_frames:
            frames = all_frames
            logger.info(f"Returning all {len(frames)} frames (requested {num_frames})")
        else:
            # Calculate indices for evenly-spaced frames
            indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
            frames = [all_frames[i] for i in indices]
            logger.info(f"Selected {len(frames)} evenly-spaced frames from {total_frames} total")

        return frames

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video metadata

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")

        try:
            info = {
                "path": str(video_path),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": None
            }

            # Calculate duration
            if info["fps"] > 0:
                info["duration_seconds"] = info["frame_count"] / info["fps"]

            return info

        finally:
            cap.release()

    @staticmethod
    def extract_frames_k_segment(
        video_path: str,
        num_frames: int = 32,
        image_size: int = 224
    ) -> Tuple[torch.Tensor, dict]:
        """
        Extract frames using K-segment sampling for VAT model.

        Divides the video into K segments and takes the first frame of each segment.
        Applies ImageNet normalization.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: 32)
            image_size: Size to resize frames to (default: 224)

        Returns:
            Tuple of:
                - torch.Tensor: [num_frames, 3, image_size, image_size]
                - dict: Video metadata

        Raises:
            InvalidVideoError: If video cannot be opened or has too few frames
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames < num_frames:
                raise InvalidVideoError(
                    f"Video too short: {total_frames} frames (need at least {num_frames})"
                )

            # K-segment sampling: divide video into K segments, take first frame of each
            segment_size = total_frames / num_frames
            frame_indices = [int(i * segment_size) for i in range(num_frames)]

            # Transform for preprocessing (ImageNet normalization)
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret:
                    # Fallback: use last valid frame
                    if frames:
                        frames.append(frames[-1])
                        logger.warning(f"Could not read frame {idx}, using previous frame")
                        continue
                    else:
                        raise InvalidVideoError(f"Could not read frame {idx} from video")

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image and apply transforms
                pil_image = Image.fromarray(frame_rgb)
                tensor = transform(pil_image)
                frames.append(tensor)

            # Video metadata
            duration = total_frames / fps if fps > 0 else None
            metadata = {
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_seconds": duration,
                "num_frames_extracted": len(frames),
                "extraction_method": "k_segment",
                "frame_indices": frame_indices
            }

            logger.info(
                f"Extracted {len(frames)} frames using K-segment sampling "
                f"(video: {total_frames} frames, {duration:.2f}s)"
            )

            return torch.stack(frames), metadata  # [num_frames, 3, H, W]

        finally:
            cap.release()
