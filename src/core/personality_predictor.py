"""
Core personality prediction logic
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.exceptions import PredictionError

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Personality prediction result"""
    traits: Dict[str, float]
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {"traits": self.traits}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class PersonalityPredictor:
    """Core personality prediction using trained model"""

    # Default trait labels (Big-5 personality traits)
    DEFAULT_TRAITS = [
        "extraversion",
        "neuroticism",
        "agreeableness",
        "conscientiousness",
        "openness"
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        trait_labels: Optional[List[str]] = None,
        image_size: int = 256
    ):
        """
        Initialize personality predictor

        Args:
            model: Trained PyTorch model
            device: Device to run inference on
            trait_labels: List of personality trait labels
            image_size: Target image size for model input
        """
        self.model = model
        self.device = device
        self.trait_labels = trait_labels or self.DEFAULT_TRAITS
        self.image_size = image_size

        # Set model to eval mode
        self.model.eval()

        # Define image transforms (same as training validation pipeline)
        self.transform = T.Compose([
            T.Resize(int(image_size * 1.05)),  # Slightly larger for center crop
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info(f"PersonalityPredictor initialized with {len(self.trait_labels)} traits")

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for model input

        Args:
            pil_image: PIL Image in RGB format

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Apply transforms
        tensor = self.transform(pil_image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

        return tensor

    def predict(
        self,
        pil_image: Image.Image,
        use_tta: bool = True
    ) -> PredictionResult:
        """
        Predict personality traits from image

        Args:
            pil_image: PIL Image in RGB format
            use_tta: Whether to use Test Time Augmentation

        Returns:
            PredictionResult with trait scores

        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(pil_image)
            input_tensor = input_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                prediction = self.model(input_tensor)

                # Test Time Augmentation (horizontal flip)
                if use_tta:
                    prediction_flip = self.model(torch.flip(input_tensor, dims=[-1]))
                    prediction = (prediction + prediction_flip) / 2.0
                    logger.debug("Applied Test Time Augmentation")

                # Convert to numpy
                scores = prediction.cpu().numpy().flatten()

            # Create results dictionary
            traits = {
                trait: float(score)
                for trait, score in zip(self.trait_labels, scores)
            }

            result = PredictionResult(
                traits=traits,
                metadata={
                    "image_size": pil_image.size,
                    "tta_used": use_tta,
                    "device": str(self.device)
                }
            )

            logger.info(f"Prediction completed: {traits}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Prediction failed: {str(e)}")

    def predict_batch(
        self,
        pil_images: List[Image.Image],
        use_tta: bool = True
    ) -> List[PredictionResult]:
        """
        Predict personality traits for multiple images

        Args:
            pil_images: List of PIL Images in RGB format
            use_tta: Whether to use Test Time Augmentation

        Returns:
            List of PredictionResults
        """
        try:
            # Preprocess all images
            tensors = [self.preprocess_image(img) for img in pil_images]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)

            # Run inference
            with torch.no_grad():
                predictions = self.model(batch_tensor)

                # Test Time Augmentation
                if use_tta:
                    predictions_flip = self.model(torch.flip(batch_tensor, dims=[-1]))
                    predictions = (predictions + predictions_flip) / 2.0

                # Convert to numpy
                scores_batch = predictions.cpu().numpy()

            # Create results for each image
            results = []
            for i, scores in enumerate(scores_batch):
                traits = {
                    trait: float(score)
                    for trait, score in zip(self.trait_labels, scores)
                }

                result = PredictionResult(
                    traits=traits,
                    metadata={
                        "image_size": pil_images[i].size,
                        "tta_used": use_tta,
                        "device": str(self.device),
                        "batch_index": i
                    }
                )
                results.append(result)

            logger.info(f"Batch prediction completed for {len(results)} images")
            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Batch prediction failed: {str(e)}")

    @staticmethod
    def get_trait_description(trait: str) -> str:
        """Get human-readable description of personality trait"""
        descriptions = {
            "extraversion": "Outgoing, social, energetic",
            "neuroticism": "Anxious, moody, emotionally unstable",
            "agreeableness": "Cooperative, trusting, helpful",
            "conscientiousness": "Organized, disciplined, responsible",
            "openness": "Creative, curious, open to new experiences"
        }
        return descriptions.get(trait.lower(), "Unknown trait")
