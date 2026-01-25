"""
Model management and loading (Singleton pattern)
Ensures model is loaded once and cached in memory
"""
import torch
from pathlib import Path
from typing import Optional, List
from threading import Lock

from src.core.models.resnet_regressor import ResNetRegressor
from src.core.models.vat_model import SemiTransformer
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import ModelNotLoadedException, ModelLoadError

logger = get_logger(__name__)

# Fixed trait order for VAT model (matches training)
VAT_TRAIT_ORDER = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']


class ModelManager:
    """
    Singleton model manager
    Loads and caches the model in memory for fast inference
    """

    _instance: Optional['ModelManager'] = None
    _lock: Lock = Lock()

    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model manager (only runs once)"""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._trait_labels: Optional[List[str]] = None
        self._checkpoint_path: Optional[str] = None
        self._backbone: Optional[str] = None

        logger.info("ModelManager initialized")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None

    @property
    def device(self) -> torch.device:
        """Get model device"""
        if not self.is_loaded:
            raise ModelNotLoadedException()
        return self._device

    @property
    def trait_labels(self) -> List[str]:
        """Get personality trait labels"""
        if not self.is_loaded:
            raise ModelNotLoadedException()
        return self._trait_labels

    @property
    def model(self) -> torch.nn.Module:
        """Get loaded model"""
        if not self.is_loaded:
            raise ModelNotLoadedException()
        return self._model

    @property
    def is_vat(self) -> bool:
        """Check if the loaded model is a VAT model"""
        return self._backbone == "vat"

    @property
    def num_frames(self) -> int:
        """Get number of frames for VAT model"""
        return settings.MODEL_NUM_FRAMES if self.is_vat else 1

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to model checkpoint (uses config if not provided)
            device: Device to load model on (uses config if not provided)

        Raises:
            ModelLoadError: If model loading fails
        """
        # Use config defaults if not provided
        # Use effective_checkpoint_path which respects MODEL_TYPE setting
        checkpoint_path = checkpoint_path or settings.effective_checkpoint_path
        device_str = device or settings.model_device_resolved

        logger.info(f"Model type configured: {settings.MODEL_TYPE}")
        logger.info(f"Expected input: {'video (32 frames)' if settings.is_vat_model else 'single image'}")

        try:
            logger.info(f"Loading model from: {checkpoint_path}")
            logger.info(f"Target device: {device_str}")

            # Check if checkpoint exists
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                raise ModelLoadError(
                    f"Checkpoint not found: {checkpoint_path}",
                    details={"path": str(ckpt_path)}
                )

            # Setup device
            self._device = torch.device(device_str)

            # Load checkpoint
            try:
                ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load checkpoint: {str(e)}",
                    details={"path": checkpoint_path, "error": str(e)}
                )

            # Detect model type based on checkpoint structure
            is_vat_checkpoint = "model_state_dict" in ckpt

            if is_vat_checkpoint:
                # VAT model checkpoint
                self._backbone = "vat"
                self._trait_labels = VAT_TRAIT_ORDER

                logger.info(f"Detected VAT model checkpoint")
                logger.info(f"Model backbone: {self._backbone}")
                logger.info(f"Predicting traits: {self._trait_labels}")

                # Initialize VAT model
                model = SemiTransformer(
                    num_classes=len(self._trait_labels),
                    seq_len=settings.MODEL_NUM_FRAMES,
                    return_feature=False
                )

                # Load weights
                try:
                    model.load_state_dict(ckpt["model_state_dict"], strict=True)
                    logger.info("VAT model weights loaded successfully")
                    if 'val_loss' in ckpt:
                        logger.info(f"Checkpoint val_loss: {ckpt['val_loss']:.4f}")
                except Exception as e:
                    raise ModelLoadError(
                        f"Failed to load VAT model weights: {str(e)}",
                        details={"error": str(e)}
                    )
            else:
                # ResNet model checkpoint (legacy)
                self._trait_labels = ckpt.get("label_cols", [
                    "extraversion",
                    "neuroticism",
                    "agreeableness",
                    "conscientiousness",
                    "openness"
                ])

                config = ckpt.get("config", {})
                self._backbone = config.get("BACKBONE", settings.MODEL_BACKBONE)

                logger.info(f"Detected ResNet model checkpoint")
                logger.info(f"Model backbone: {self._backbone}")
                logger.info(f"Predicting traits: {self._trait_labels}")

                # Initialize ResNet model
                model = ResNetRegressor(
                    backbone=self._backbone,
                    out_dim=len(self._trait_labels),
                    pretrained=False
                )

                # Load weights
                try:
                    model.load_state_dict(ckpt["model"], strict=True)
                    logger.info("ResNet model weights loaded successfully")
                except Exception as e:
                    raise ModelLoadError(
                        f"Failed to load model weights: {str(e)}",
                        details={"error": str(e)}
                    )

            # Move to device and set to eval mode
            model = model.to(self._device)
            model.eval()

            self._model = model
            self._checkpoint_path = checkpoint_path

            # Log model info
            logger.info(f"Model loaded successfully")
            if self.is_vat:
                num_params = sum(p.numel() for p in model.parameters())
                num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Parameters: {num_params:,}")
                logger.info(f"Trainable parameters: {num_trainable:,}")
            else:
                logger.info(f"Parameters: {model.num_parameters:,}")
                logger.info(f"Trainable parameters: {model.num_trainable_parameters:,}")

            # Warm up model if configured
            if settings.MODEL_WARMUP:
                self._warmup_model()

        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}", exc_info=True)
            raise ModelLoadError(
                f"Unexpected error loading model: {str(e)}",
                details={"error": str(e)}
            )

    def _warmup_model(self) -> None:
        """Warm up model with dummy input"""
        try:
            logger.info("Warming up model...")
            image_size = settings.effective_image_size

            if self.is_vat:
                # VAT model expects [batch, num_frames, 3, H, W]
                dummy_input = torch.randn(
                    1, settings.MODEL_NUM_FRAMES, 3,
                    image_size, image_size
                ).to(self._device)
            else:
                # ResNet model expects [batch, 3, H, W]
                dummy_input = torch.randn(
                    1, 3, image_size, image_size
                ).to(self._device)

            with torch.no_grad():
                _ = self._model(dummy_input)

            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def unload_model(self) -> None:
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            self._model = None

            # Clear CUDA cache if using GPU
            if self._device and self._device.type == 'cuda':
                torch.cuda.empty_cache()

            logger.info("Model unloaded from memory")

    def reload_model(
        self,
        checkpoint_path: Optional[str] = None
    ) -> None:
        """
        Reload model (useful for model updates)

        Args:
            checkpoint_path: Path to new checkpoint (uses current if not provided)
        """
        logger.info("Reloading model...")
        checkpoint_path = checkpoint_path or self._checkpoint_path
        self.unload_model()
        self.load_model(checkpoint_path)
        logger.info("Model reloaded successfully")

    def get_model_info(self) -> dict:
        """Get model information"""
        if not self.is_loaded:
            return {"loaded": False}

        info = {
            "loaded": True,
            "checkpoint_path": self._checkpoint_path,
            "backbone": self._backbone,
            "device": str(self._device),
            "trait_labels": self._trait_labels,
            "is_vat": self.is_vat,
        }

        # Add parameter counts (different attribute access for different models)
        if self.is_vat:
            info["num_parameters"] = sum(p.numel() for p in self._model.parameters())
            info["num_trainable_parameters"] = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )
            info["num_frames"] = settings.MODEL_NUM_FRAMES
            info["image_size"] = settings.MODEL_IMAGE_SIZE
        else:
            info["num_parameters"] = self._model.num_parameters
            info["num_trainable_parameters"] = self._model.num_trainable_parameters

        return info


# Global singleton instance
_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get model manager instance (dependency injection helper)"""
    return _model_manager
