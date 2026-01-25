"""
Audio Analysis Service for Personality Insights

Extracts acoustic features from video audio that correlate with personality traits.
Uses librosa for signal processing and provides normalized metrics for LLM analysis.

Research-backed acoustic correlates:
- Pitch (F0): Higher pitch & variability → Extraversion, Neuroticism
- Speaking rate: Faster → Extraversion, lower Conscientiousness
- Loudness: Higher → Extraversion, Dominance
- Voice stability: Lower jitter/shimmer → Emotional stability (low Neuroticism)
- Spectral brightness: Brighter voice → Extraversion, Energy
"""
import os
import tempfile
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


class AudioAnalysisService:
    """
    Service for extracting personality-relevant acoustic features from audio.
    """

    def __init__(self):
        self._librosa = None
        self._sr = 22050  # Standard sample rate for librosa

    @property
    def librosa(self):
        """Lazy load librosa"""
        if self._librosa is None:
            try:
                import librosa
                self._librosa = librosa
                logger.info("librosa loaded successfully")
            except ImportError:
                raise ImportError(
                    "librosa not installed. Run: pip install librosa"
                )
        return self._librosa

    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio track from video file.

        Args:
            video_path: Path to video file

        Returns:
            Path to temporary WAV file, or None if no audio
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

            clip = VideoFileClip(video_path)

            if clip.audio is None:
                logger.warning(f"Video has no audio track: {video_path}")
                clip.close()
                return None

            # Create temp file for audio
            fd, audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            # Extract audio
            clip.audio.write_audiofile(
                audio_path,
                fps=self._sr,
                nbytes=2,
                codec='pcm_s16le',
                logger=None
            )
            clip.close()

            logger.info(f"Extracted audio using moviepy to: {audio_path}")
            return audio_path

        except Exception as e:
            logger.warning(f"moviepy audio extraction failed: {e}")
            return None

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
            fd, audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            # Use ffmpeg to extract audio as WAV
            cmd = [
                ffmpeg_path,
                "-y",  # Overwrite output
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", str(self._sr),
                "-ac", "1",  # Mono
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
                logger.info(f"Extracted audio using ffmpeg to: {audio_path}")
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

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio file and extract personality-relevant acoustic features.

        Args:
            audio_path: Path to audio file (WAV format preferred)

        Returns:
            Dictionary with acoustic metrics and interpretations
        """
        try:
            # Load audio
            y, sr = self.librosa.load(audio_path, sr=self._sr)
            duration = len(y) / sr

            logger.info(f"Analyzing audio: {duration:.1f}s at {sr}Hz")

            # Extract all features
            pitch_metrics = self._extract_pitch_features(y, sr)
            energy_metrics = self._extract_energy_features(y, sr)
            tempo_metrics = self._extract_tempo_features(y, sr)
            spectral_metrics = self._extract_spectral_features(y, sr)
            voice_quality = self._extract_voice_quality(y, sr)

            # Combine all metrics
            raw_metrics = {
                **pitch_metrics,
                **energy_metrics,
                **tempo_metrics,
                **spectral_metrics,
                **voice_quality,
                "duration_seconds": round(duration, 2)
            }

            # Normalize and interpret
            normalized_metrics = self._normalize_metrics(raw_metrics)
            interpretations = self._interpret_metrics(normalized_metrics)

            result = {
                "raw_metrics": raw_metrics,
                "normalized_metrics": normalized_metrics,
                "interpretations": interpretations,
                "personality_indicators": self._derive_personality_indicators(normalized_metrics),
                "metadata": {
                    "sample_rate": sr,
                    "duration_seconds": round(duration, 2),
                    "analysis_version": "1.0"
                }
            }

            logger.info(f"Audio analysis complete: {len(normalized_metrics)} metrics extracted")
            return result

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            raise

    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract fundamental frequency (F0) features"""
        try:
            # Use pyin for more robust pitch detection
            f0, voiced_flag, voiced_probs = self.librosa.pyin(
                y,
                fmin=self.librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=self.librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=sr
            )

            # Filter to voiced segments only
            f0_voiced = f0[~np.isnan(f0)]

            if len(f0_voiced) == 0:
                return {
                    "pitch_mean_hz": 0,
                    "pitch_std_hz": 0,
                    "pitch_range_hz": 0,
                    "pitch_variability": 0,
                    "voiced_ratio": 0
                }

            pitch_mean = float(np.mean(f0_voiced))
            pitch_std = float(np.std(f0_voiced))
            pitch_range = float(np.max(f0_voiced) - np.min(f0_voiced))
            voiced_ratio = len(f0_voiced) / len(f0)

            return {
                "pitch_mean_hz": round(pitch_mean, 1),
                "pitch_std_hz": round(pitch_std, 1),
                "pitch_range_hz": round(pitch_range, 1),
                "pitch_variability": round(pitch_std / pitch_mean if pitch_mean > 0 else 0, 3),
                "voiced_ratio": round(voiced_ratio, 3)
            }

        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return {
                "pitch_mean_hz": 0,
                "pitch_std_hz": 0,
                "pitch_range_hz": 0,
                "pitch_variability": 0,
                "voiced_ratio": 0
            }

    def _extract_energy_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract loudness and energy features"""
        try:
            # RMS energy
            rms = self.librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))

            # Convert to dB
            rms_db = self.librosa.amplitude_to_db(rms, ref=np.max)
            loudness_mean = float(np.mean(rms_db))
            loudness_range = float(np.max(rms_db) - np.min(rms_db))

            # Dynamic range (difference between loud and quiet parts)
            loudness_percentile_90 = float(np.percentile(rms_db, 90))
            loudness_percentile_10 = float(np.percentile(rms_db, 10))
            dynamic_range = loudness_percentile_90 - loudness_percentile_10

            return {
                "rms_energy": round(rms_mean, 6),
                "rms_variability": round(rms_std / rms_mean if rms_mean > 0 else 0, 3),
                "loudness_mean_db": round(loudness_mean, 1),
                "loudness_range_db": round(loudness_range, 1),
                "dynamic_range_db": round(dynamic_range, 1)
            }

        except Exception as e:
            logger.warning(f"Energy extraction failed: {e}")
            return {
                "rms_energy": 0,
                "rms_variability": 0,
                "loudness_mean_db": 0,
                "loudness_range_db": 0,
                "dynamic_range_db": 0
            }

    def _extract_tempo_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract tempo and rhythm features (proxy for speaking rate)"""
        try:
            # Onset detection for speech rhythm
            onset_env = self.librosa.onset.onset_strength(y=y, sr=sr)
            tempo, beats = self.librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

            # Speech rate proxy: onsets per second
            onsets = self.librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
            duration = len(y) / sr
            onset_rate = len(onsets) / duration if duration > 0 else 0

            # Pause detection (low energy segments)
            rms = self.librosa.feature.rms(y=y)[0]
            silence_threshold = np.percentile(rms, 20)
            silent_frames = np.sum(rms < silence_threshold)
            pause_ratio = silent_frames / len(rms)

            return {
                "tempo_bpm": round(float(tempo), 1) if not isinstance(tempo, np.ndarray) else round(float(tempo[0]), 1),
                "onset_rate_per_sec": round(onset_rate, 2),
                "pause_ratio": round(pause_ratio, 3),
                "speech_density": round(1 - pause_ratio, 3)
            }

        except Exception as e:
            logger.warning(f"Tempo extraction failed: {e}")
            return {
                "tempo_bpm": 0,
                "onset_rate_per_sec": 0,
                "pause_ratio": 0,
                "speech_density": 0
            }

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features (voice brightness, clarity)"""
        try:
            # Spectral centroid (brightness)
            centroid = self.librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_mean = float(np.mean(centroid))

            # Spectral rolloff (frequency below which 85% of energy)
            rolloff = self.librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            rolloff_mean = float(np.mean(rolloff))

            # Spectral bandwidth (spread of frequencies)
            bandwidth = self.librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            bandwidth_mean = float(np.mean(bandwidth))

            # Spectral flatness (tonal vs noisy)
            flatness = self.librosa.feature.spectral_flatness(y=y)[0]
            flatness_mean = float(np.mean(flatness))

            return {
                "spectral_centroid_hz": round(centroid_mean, 1),
                "spectral_rolloff_hz": round(rolloff_mean, 1),
                "spectral_bandwidth_hz": round(bandwidth_mean, 1),
                "spectral_flatness": round(flatness_mean, 4),
                "voice_brightness": round(centroid_mean / 4000, 2)  # Normalized brightness score
            }

        except Exception as e:
            logger.warning(f"Spectral extraction failed: {e}")
            return {
                "spectral_centroid_hz": 0,
                "spectral_rolloff_hz": 0,
                "spectral_bandwidth_hz": 0,
                "spectral_flatness": 0,
                "voice_brightness": 0
            }

    def _extract_voice_quality(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract voice quality metrics (stability, clarity)"""
        try:
            # Zero crossing rate (indicator of noisiness/breathiness)
            zcr = self.librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            zcr_std = float(np.std(zcr))

            # MFCC variance as voice consistency measure
            mfccs = self.librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

            # Harmonic-to-noise ratio approximation
            harmonic, percussive = self.librosa.effects.hpss(y)
            hnr_approx = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-10))

            return {
                "zero_crossing_rate": round(zcr_mean, 4),
                "zcr_variability": round(zcr_std, 4),
                "mfcc_variance": round(mfcc_var, 2),
                "harmonic_ratio": round(min(hnr_approx, 10), 2),  # Cap at 10
                "voice_stability": round(1 / (1 + zcr_std), 3)  # Higher = more stable
            }

        except Exception as e:
            logger.warning(f"Voice quality extraction failed: {e}")
            return {
                "zero_crossing_rate": 0,
                "zcr_variability": 0,
                "mfcc_variance": 0,
                "harmonic_ratio": 0,
                "voice_stability": 0
            }

    def _normalize_metrics(self, raw_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize raw metrics to 0-100 scale for easier interpretation.
        Based on typical human speech ranges.
        """
        # Reference ranges for normalization (based on research)
        normalization_params = {
            # Pitch: typical range 85-255 Hz for adults
            "pitch_mean_hz": (85, 255),
            "pitch_variability": (0.05, 0.4),
            # Energy
            "loudness_mean_db": (-40, -5),
            "dynamic_range_db": (5, 40),
            # Tempo
            "onset_rate_per_sec": (2, 8),
            "speech_density": (0.3, 0.9),
            # Spectral
            "voice_brightness": (0.2, 0.8),
            "spectral_centroid_hz": (500, 3000),
            # Voice quality
            "voice_stability": (0.3, 0.9),
            "harmonic_ratio": (1, 8),
        }

        normalized = {}
        for key, (min_val, max_val) in normalization_params.items():
            if key in raw_metrics:
                value = raw_metrics[key]
                # Clamp and normalize to 0-100
                clamped = max(min_val, min(max_val, value))
                score = ((clamped - min_val) / (max_val - min_val)) * 100
                normalized[key] = round(score, 1)

        return normalized

    def _interpret_metrics(self, normalized: Dict[str, float]) -> Dict[str, str]:
        """
        Provide human-readable interpretations of normalized metrics.
        """
        interpretations = {}

        # Pitch interpretation
        if "pitch_mean_hz" in normalized:
            score = normalized["pitch_mean_hz"]
            if score < 33:
                interpretations["pitch"] = "Low pitch voice, often perceived as calm and authoritative"
            elif score < 66:
                interpretations["pitch"] = "Moderate pitch, balanced vocal tone"
            else:
                interpretations["pitch"] = "Higher pitch voice, often perceived as energetic and expressive"

        # Pitch variability
        if "pitch_variability" in normalized:
            score = normalized["pitch_variability"]
            if score < 33:
                interpretations["expressiveness"] = "Monotone speech pattern, measured and consistent"
            elif score < 66:
                interpretations["expressiveness"] = "Moderate vocal variety, balanced expression"
            else:
                interpretations["expressiveness"] = "Highly expressive speech with wide pitch variation"

        # Loudness
        if "loudness_mean_db" in normalized:
            score = normalized["loudness_mean_db"]
            if score < 33:
                interpretations["volume"] = "Soft-spoken, quieter voice"
            elif score < 66:
                interpretations["volume"] = "Moderate volume, conversational tone"
            else:
                interpretations["volume"] = "Loud and projecting voice"

        # Speaking rate
        if "speech_density" in normalized:
            score = normalized["speech_density"]
            if score < 33:
                interpretations["pace"] = "Slower speech with many pauses, deliberate communication"
            elif score < 66:
                interpretations["pace"] = "Moderate speaking pace"
            else:
                interpretations["pace"] = "Fast-paced speech with few pauses, rapid communication"

        # Voice brightness
        if "voice_brightness" in normalized:
            score = normalized["voice_brightness"]
            if score < 33:
                interpretations["brightness"] = "Warm, mellow voice quality"
            elif score < 66:
                interpretations["brightness"] = "Balanced voice clarity"
            else:
                interpretations["brightness"] = "Bright, clear, and sharp voice quality"

        # Voice stability
        if "voice_stability" in normalized:
            score = normalized["voice_stability"]
            if score < 33:
                interpretations["stability"] = "Variable voice quality, possible tension or emotion"
            elif score < 66:
                interpretations["stability"] = "Moderately stable voice"
            else:
                interpretations["stability"] = "Very stable and controlled voice"

        return interpretations

    def _derive_personality_indicators(self, normalized: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Derive personality-relevant indicators from acoustic features.
        Based on research correlating voice with personality.

        Three indicators:
        1. Vocal Extraversion - How animated/projecting the voice is
        2. Vocal Expressiveness - Monotone vs expressive intonation
        3. Vocal Fluency - Smooth confident delivery vs hesitant
        """
        indicators = {}

        # ===========================================
        # 1. VOCAL EXTRAVERSION (keep - working well)
        # Based on: loudness, speech density, pitch variability, brightness
        # ===========================================
        extraversion_signals = []
        extraversion_score = 50  # Start neutral

        if "loudness_mean_db" in normalized:
            extraversion_score += (normalized["loudness_mean_db"] - 50) * 0.3
            if normalized["loudness_mean_db"] > 60:
                extraversion_signals.append("projects voice confidently")

        if "speech_density" in normalized:
            extraversion_score += (normalized["speech_density"] - 50) * 0.25
            if normalized["speech_density"] > 60:
                extraversion_signals.append("speaks rapidly with few pauses")

        if "pitch_variability" in normalized:
            extraversion_score += (normalized["pitch_variability"] - 50) * 0.25
            if normalized["pitch_variability"] > 60:
                extraversion_signals.append("uses expressive intonation")

        if "voice_brightness" in normalized:
            extraversion_score += (normalized["voice_brightness"] - 50) * 0.2
            if normalized["voice_brightness"] > 60:
                extraversion_signals.append("has bright, energetic voice")

        indicators["vocal_extraversion"] = {
            "score": round(max(0, min(100, extraversion_score)), 1),
            "level": self._get_level(extraversion_score),
            "signals": extraversion_signals or ["neutral vocal pattern"]
        }

        # ===========================================
        # 2. VOCAL EXPRESSIVENESS (new - replaces warmth)
        # Based on: pitch variance + dynamic range
        # Captures: monotone vs animated intonation
        # ===========================================
        expressiveness_signals = []
        expressiveness_score = 50

        # Pitch variability is the primary driver
        if "pitch_variability" in normalized:
            expressiveness_score += (normalized["pitch_variability"] - 50) * 0.5
            if normalized["pitch_variability"] > 65:
                expressiveness_signals.append("wide pitch range")
            elif normalized["pitch_variability"] < 35:
                expressiveness_signals.append("monotone delivery")

        # Dynamic range adds to expressiveness
        if "dynamic_range_db" in normalized:
            expressiveness_score += (normalized["dynamic_range_db"] - 50) * 0.3
            if normalized["dynamic_range_db"] > 60:
                expressiveness_signals.append("varied volume emphasis")

        # Spectral variation indicates tonal expressiveness
        if "spectral_centroid_hz" in normalized:
            # Higher centroid variance = more expressive
            expressiveness_score += (normalized["spectral_centroid_hz"] - 50) * 0.2
            if normalized["spectral_centroid_hz"] > 60:
                expressiveness_signals.append("varied tonal quality")

        indicators["vocal_expressiveness"] = {
            "score": round(max(0, min(100, expressiveness_score)), 1),
            "level": self._get_level(expressiveness_score),
            "signals": expressiveness_signals or ["moderate expression"]
        }

        # ===========================================
        # 3. VOCAL FLUENCY (new - replaces confidence/stability)
        # Based on: low pause ratio + speech density + low hesitation
        # Captures: smooth confident delivery vs hesitant/nervous
        # ===========================================
        fluency_signals = []
        fluency_score = 50

        # Speech density (inverse of pauses) - primary driver
        if "speech_density" in normalized:
            fluency_score += (normalized["speech_density"] - 50) * 0.4
            if normalized["speech_density"] > 65:
                fluency_signals.append("continuous speech flow")
            elif normalized["speech_density"] < 35:
                fluency_signals.append("frequent pauses")

        # Voice stability contributes to fluency
        if "voice_stability" in normalized:
            fluency_score += (normalized["voice_stability"] - 50) * 0.3
            if normalized["voice_stability"] > 60:
                fluency_signals.append("steady voice quality")

        # Onset rate (syllable rate proxy) - higher = more fluent
        if "onset_rate_per_sec" in normalized:
            # Normalize onset rate: typical range 2-8 per second
            onset_norm = normalized.get("onset_rate_per_sec", 50)
            fluency_score += (onset_norm - 50) * 0.3
            if onset_norm > 60:
                fluency_signals.append("good articulation pace")

        indicators["vocal_fluency"] = {
            "score": round(max(0, min(100, fluency_score)), 1),
            "level": self._get_level(fluency_score),
            "signals": fluency_signals or ["moderate fluency"]
        }

        return indicators

    def _get_level(self, score: float) -> str:
        """Convert score to level label"""
        if score < 35:
            return "Low"
        elif score < 65:
            return "Moderate"
        else:
            return "High"

    def analyze_video(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Complete pipeline: extract audio from video and analyze it.

        Args:
            video_path: Path to video file

        Returns:
            Analysis results or None if no audio
        """
        audio_path = None
        try:
            # Extract audio
            audio_path = self.extract_audio_from_video(video_path)
            if audio_path is None:
                return None

            # Analyze
            results = self.analyze_audio(audio_path)
            return results

        finally:
            # Cleanup
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass


# Singleton instance
_audio_service: Optional[AudioAnalysisService] = None


def get_audio_analysis_service() -> AudioAnalysisService:
    """Get or create singleton audio analysis service"""
    global _audio_service
    if _audio_service is None:
        _audio_service = AudioAnalysisService()
    return _audio_service
