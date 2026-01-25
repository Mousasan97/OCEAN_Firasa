"""
Multimodal Analysis Service
Extracts frames and transcribes audio from video for AI-powered personality insights

Optimized for parallel processing - can accept pre-extracted frames to avoid
redundant video reads when used alongside ResNet prediction.
"""
import os
import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from openai import OpenAI

from src.utils.logger import get_logger
from src.utils.config import settings
from src.utils.exceptions import PredictionError

logger = get_logger(__name__)

# Thread pool for parallel audio extraction
_audio_executor = ThreadPoolExecutor(max_workers=2)


class MultimodalAnalysisService:
    """
    Service for extracting frames and transcribing audio from videos
    for multimodal personality analysis
    """

    def __init__(self, num_frames: int = 10):
        """
        Initialize the multimodal analysis service

        Args:
            num_frames: Number of frames to extract (default 10)
        """
        self.num_frames = num_frames
        self._client: Optional[OpenAI] = None

        logger.info(f"MultimodalAnalysisService initialized (num_frames={num_frames})")

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise PredictionError("OPENAI_API_KEY not configured for multimodal analysis")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def extract_frames_from_video(self, video_path: str) -> List[Image.Image]:
        """
        Extract evenly-spaced frames from a video file

        Args:
            video_path: Path to the video file

        Returns:
            List of PIL Images
        """
        # Try moviepy first, fall back to OpenCV for problematic formats (e.g., WebM)
        try:
            return self._extract_frames_moviepy(video_path)
        except Exception as e:
            logger.warning(f"moviepy frame extraction failed: {e}, trying OpenCV fallback")
            return self._extract_frames_opencv(video_path)

    def _extract_frames_moviepy(self, video_path: str) -> List[Image.Image]:
        """Extract frames using moviepy (preferred for most formats)"""
        try:
            from moviepy import VideoFileClip
        except ImportError:
            raise PredictionError("moviepy not installed. Run: pip install moviepy")

        frames = []
        video = None

        try:
            video = VideoFileClip(video_path)
            duration = video.duration

            logger.info(f"Extracting {self.num_frames} frames from video (duration={duration:.2f}s)")

            for i in range(self.num_frames):
                # Calculate evenly-spaced timestamp
                if self.num_frames > 1:
                    timestamp = (i / (self.num_frames - 1)) * duration
                else:
                    timestamp = 0

                # Ensure we don't go past the end
                timestamp = min(timestamp, duration - 0.1)

                # Get frame at timestamp
                frame_array = video.get_frame(timestamp)

                # Convert to PIL Image
                img = Image.fromarray(frame_array)
                frames.append(img)

                logger.debug(f"Extracted frame {i+1} at {timestamp:.2f}s")

            logger.info(f"Successfully extracted {len(frames)} frames using moviepy")
            return frames

        finally:
            if video is not None:
                video.close()

    def _extract_frames_opencv(self, video_path: str) -> List[Image.Image]:
        """Extract frames using OpenCV (fallback for WebM and other problematic formats)"""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise PredictionError(f"Cannot open video: {video_path}")

        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Handle videos with missing frame count metadata (e.g., WebM from browser)
            if n_frames <= 0 or n_frames > 100000:
                logger.info(f"Frame count unavailable ({n_frames}), reading sequentially")
                return self._extract_frames_opencv_sequential(cap, fps)

            # Calculate frame indices for evenly-spaced extraction
            if self.num_frames >= n_frames:
                indices = list(range(n_frames))
            else:
                indices = [int(i * n_frames / self.num_frames) for i in range(self.num_frames)]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    frames.append(img)

            logger.info(f"Successfully extracted {len(frames)} frames using OpenCV")
            return frames

        finally:
            cap.release()

    def _extract_frames_opencv_sequential(self, cap, fps: float) -> List[Image.Image]:
        """Extract frames by reading sequentially (for videos without frame count)"""
        import cv2

        logger.info(f"Sequential frame extraction: requesting {self.num_frames} frames, fps={fps}")

        # Read ALL frames first
        all_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            all_frames.append(img)
            frame_idx += 1

            if frame_idx > 10000:
                logger.warning("Video too long, stopping at 10000 frames")
                break

        total_frames = len(all_frames)
        logger.info(f"Read {total_frames} total frames from video")

        if total_frames == 0:
            raise PredictionError("No frames could be extracted from video")

        # Select evenly-spaced frames
        if self.num_frames >= total_frames:
            frames = all_frames
            logger.info(f"Returning all {len(frames)} frames (requested {self.num_frames})")
        else:
            indices = [int(i * (total_frames - 1) / (self.num_frames - 1)) for i in range(self.num_frames)]
            frames = [all_frames[i] for i in indices]
            logger.info(f"Selected {len(frames)} evenly-spaced frames from {total_frames} total")

        return frames

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video and save to temp file

        Args:
            video_path: Path to the video file

        Returns:
            Path to the extracted audio file, or None if no audio
        """
        # Try moviepy first, fall back to ffmpeg CLI for problematic formats (e.g., WebM)
        audio_path = self._extract_audio_moviepy(video_path)
        if audio_path:
            return audio_path

        # Fallback to ffmpeg CLI
        logger.info("moviepy audio extraction failed, trying ffmpeg CLI")
        return self._extract_audio_ffmpeg(video_path)

    def _extract_audio_moviepy(self, video_path: str) -> Optional[str]:
        """Extract audio using moviepy (preferred for most formats)"""
        try:
            from moviepy import VideoFileClip
        except ImportError:
            logger.warning("moviepy not installed")
            return None

        video = None
        audio_path = None

        try:
            video = VideoFileClip(video_path)

            # Check if video has audio
            if video.audio is None:
                logger.warning("Video has no audio track")
                return None

            # Create temp file for audio
            audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
            os.close(audio_fd)

            logger.info(f"Extracting audio to: {audio_path}")
            video.audio.write_audiofile(audio_path, logger=None)

            logger.info("Audio extraction complete using moviepy")
            return audio_path

        except Exception as e:
            logger.warning(f"moviepy audio extraction failed: {e}")
            # Clean up temp file on error
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass
            return None
        finally:
            if video is not None:
                video.close()

    def _extract_audio_ffmpeg(self, video_path: str) -> Optional[str]:
        """Extract audio using ffmpeg CLI (fallback for WebM and other formats)"""
        import subprocess
        import shutil

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.warning("ffmpeg not found in PATH, cannot extract audio")
            return None

        audio_path = None
        try:
            # Create temp file for audio
            audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
            os.close(audio_fd)

            # Use ffmpeg to extract audio
            cmd = [
                ffmpeg_path,
                "-y",  # Overwrite output
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "libmp3lame",
                "-ab", "128k",
                "-ar", "44100",
                audio_path
            ]

            logger.info(f"Extracting audio with ffmpeg: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.warning(f"ffmpeg audio extraction failed: {result.stderr}")
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                return None

            # Verify the output file exists and has content
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info("Audio extraction complete using ffmpeg")
                return audio_path
            else:
                logger.warning("ffmpeg produced empty audio file")
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                return None

        except Exception as e:
            logger.warning(f"ffmpeg audio extraction error: {e}")
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass
            return None

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file using OpenAI's transcription API

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")

            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file
                )

            transcript = transcription.text
            logger.info(f"Transcription complete ({len(transcript)} chars)")
            return transcript

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise PredictionError(f"Audio transcription failed: {str(e)}")

    def frames_to_base64(self, frames: List[Image.Image], max_size: int = 512) -> List[str]:
        """
        Convert frames to base64-encoded data URLs for API submission

        Args:
            frames: List of PIL Images
            max_size: Maximum dimension to resize frames (for API efficiency)

        Returns:
            List of base64 data URLs
        """
        data_urls = []

        for i, frame in enumerate(frames):
            try:
                # Resize if too large
                if frame.width > max_size or frame.height > max_size:
                    frame.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Convert to RGB if necessary
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')

                # Encode to base64
                buffer = BytesIO()
                frame.save(buffer, format='JPEG', quality=85)
                b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
                data_url = f"data:image/jpeg;base64,{b64}"
                data_urls.append(data_url)

            except Exception as e:
                logger.warning(f"Failed to encode frame {i}: {e}")
                continue

        logger.info(f"Encoded {len(data_urls)} frames to base64")
        return data_urls

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Full multimodal analysis: extract frames, transcribe audio

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with frames (base64), transcript, and metadata
        """
        logger.info(f"Starting multimodal analysis: {video_path}")

        result = {
            "frames_base64": [],
            "transcript": "",
            "metadata": {
                "num_frames_extracted": 0,
                "has_audio": False,
                "transcript_length": 0
            }
        }

        # Extract frames
        try:
            frames = self.extract_frames_from_video(video_path)
            result["frames_base64"] = self.frames_to_base64(frames)
            result["metadata"]["num_frames_extracted"] = len(frames)
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            # Continue even if frame extraction fails

        # Extract and transcribe audio
        audio_path = None
        try:
            audio_path = self.extract_audio_from_video(video_path)
            if audio_path:
                result["metadata"]["has_audio"] = True
                transcript = self.transcribe_audio(audio_path)
                result["transcript"] = transcript
                result["metadata"]["transcript_length"] = len(transcript)
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            # Continue even if audio fails
        finally:
            # Clean up temp audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

        logger.info(f"Multimodal analysis complete: {result['metadata']}")
        return result

    def analyze_with_preextracted_frames(
        self,
        video_path: str,
        frames: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Multimodal analysis using pre-extracted frames (avoids redundant video read).

        Use this when frames were already extracted for ResNet prediction.

        Args:
            video_path: Path to video file (only used for audio extraction)
            frames: Pre-extracted frames as numpy arrays (RGB)

        Returns:
            Dictionary with frames (base64), transcript, and metadata
        """
        logger.info(f"Starting multimodal analysis with {len(frames)} pre-extracted frames")

        result = {
            "frames_base64": [],
            "transcript": "",
            "metadata": {
                "num_frames_extracted": len(frames),
                "has_audio": False,
                "transcript_length": 0,
                "used_preextracted_frames": True
            }
        }

        # Convert pre-extracted frames to base64
        try:
            # Convert numpy arrays to PIL Images
            pil_frames = [Image.fromarray(f) for f in frames]
            result["frames_base64"] = self.frames_to_base64(pil_frames)
            logger.info(f"Converted {len(result['frames_base64'])} pre-extracted frames to base64")
        except Exception as e:
            logger.error(f"Frame conversion failed: {e}")

        # Extract and transcribe audio (still need to read video for this)
        audio_path = None
        try:
            audio_path = self.extract_audio_from_video(video_path)
            if audio_path:
                result["metadata"]["has_audio"] = True
                transcript = self.transcribe_audio(audio_path)
                result["transcript"] = transcript
                result["metadata"]["transcript_length"] = len(transcript)
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

        logger.info(f"Multimodal analysis complete: {result['metadata']}")
        return result

    async def transcribe_audio_only_async(self, video_path: str) -> Dict[str, Any]:
        """
        Extract and transcribe audio only (for parallel processing).

        Use this when frames will be handled separately.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with transcript and audio metadata
        """
        logger.info(f"Starting audio-only transcription: {video_path}")

        result = {
            "transcript": "",
            "metadata": {
                "has_audio": False,
                "transcript_length": 0
            }
        }

        audio_path = None
        try:
            # Extract audio in thread pool
            loop = asyncio.get_event_loop()
            audio_path = await loop.run_in_executor(
                _audio_executor,
                self.extract_audio_from_video,
                video_path
            )

            if audio_path:
                result["metadata"]["has_audio"] = True
                # Transcribe in thread pool (API call is blocking)
                transcript = await loop.run_in_executor(
                    _audio_executor,
                    self.transcribe_audio,
                    audio_path
                )
                result["transcript"] = transcript
                result["metadata"]["transcript_length"] = len(transcript)
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

        return result

    async def analyze_video_async(self, video_path: str) -> Dict[str, Any]:
        """
        Async wrapper for multimodal analysis

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with frames (base64), transcript, and metadata
        """
        return await asyncio.to_thread(self.analyze_video, video_path)

    async def analyze_with_preextracted_frames_async(
        self,
        video_path: str,
        frames: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Async version of analyze_with_preextracted_frames.

        Args:
            video_path: Path to video file (for audio extraction)
            frames: Pre-extracted frames as numpy arrays

        Returns:
            Dictionary with frames (base64), transcript, and metadata
        """
        return await asyncio.to_thread(
            self.analyze_with_preextracted_frames,
            video_path,
            frames
        )


# Singleton instance
_multimodal_service: Optional[MultimodalAnalysisService] = None


def get_multimodal_service() -> MultimodalAnalysisService:
    """Get or create singleton multimodal service instance"""
    global _multimodal_service
    if _multimodal_service is None:
        _multimodal_service = MultimodalAnalysisService(num_frames=10)
    return _multimodal_service
