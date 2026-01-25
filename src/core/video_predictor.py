"""
Video-based personality predictor for VAT model
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


class VideoPersonalityPredictor:
    """
    Video-based personality prediction using VAT (Video Attention Transformer) model.

    Takes preprocessed video frames and returns OCEAN personality trait predictions.
    """

    # Fixed trait order for VAT model (matches training)
    TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        trait_labels: Optional[List[str]] = None,
        use_tta: bool = True
    ):
        """
        Initialize the video predictor.

        Args:
            model: Loaded VAT model (SemiTransformer)
            device: Device to run inference on
            trait_labels: List of trait labels (uses default TRAITS if not provided)
            use_tta: Whether to use test-time augmentation (horizontal flip)
        """
        self.model = model
        self.device = device
        self.trait_labels = trait_labels or self.TRAITS
        self.use_tta = use_tta

        logger.info(f"VideoPersonalityPredictor initialized (TTA: {use_tta})")

    def predict(self, frames: torch.Tensor) -> Dict[str, float]:
        """
        Predict personality traits from video frames.

        Args:
            frames: Tensor of shape [num_frames, 3, H, W] (preprocessed frames)

        Returns:
            Dictionary mapping trait names to scores
        """
        # Add batch dimension: [num_frames, 3, H, W] -> [1, num_frames, 3, H, W]
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)

        frames = frames.to(self.device)

        with torch.no_grad():
            if self.use_tta:
                # Test-time augmentation: average predictions with horizontal flip
                output_normal = self.model(frames)

                # Horizontal flip
                frames_flipped = torch.flip(frames, dims=[-1])
                output_flipped = self.model(frames_flipped)

                # Average predictions
                output = (output_normal + output_flipped) / 2
            else:
                output = self.model(frames)

            predictions = output.cpu().numpy()[0]

        # Create result dictionary
        result = {}
        for trait, score in zip(self.trait_labels, predictions):
            result[trait] = float(score)

        logger.debug(f"Predictions: {result}")
        return result

    def predict_batch(self, frames_batch: torch.Tensor) -> List[Dict[str, float]]:
        """
        Predict personality traits for a batch of videos.

        Args:
            frames_batch: Tensor of shape [batch, num_frames, 3, H, W]

        Returns:
            List of dictionaries mapping trait names to scores
        """
        frames_batch = frames_batch.to(self.device)

        with torch.no_grad():
            if self.use_tta:
                output_normal = self.model(frames_batch)
                frames_flipped = torch.flip(frames_batch, dims=[-1])
                output_flipped = self.model(frames_flipped)
                output = (output_normal + output_flipped) / 2
            else:
                output = self.model(frames_batch)

            predictions = output.cpu().numpy()

        results = []
        for pred in predictions:
            result = {}
            for trait, score in zip(self.trait_labels, pred):
                result[trait] = float(score)
            results.append(result)

        return results
