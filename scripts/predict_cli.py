#!/usr/bin/env python3
"""
CLI tool for personality prediction
Refactored version maintaining compatibility with original script
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.model_manager import get_model_manager
from src.services.prediction_service import PredictionService
from src.services.preprocessing_service import PreprocessingService
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def display_results(results: dict, input_path: str):
    """Display prediction results in a nice format"""
    predictions = results.get("predictions", {})
    metadata = results.get("metadata", {})

    print(f"\n{'='*70}")
    print(f"PERSONALITY PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"Input: {input_path}")

    preprocessing = metadata.get("preprocessing", {})
    if preprocessing:
        print(f"Face detected: {preprocessing.get('face_detected', False)}")
        print(f"Processed size: {preprocessing.get('processed_size', 'N/A')}")

    print(f"{'='*70}")

    # Trait descriptions
    descriptions = {
        "extraversion": "Outgoing, social, energetic",
        "neuroticism": "Anxious, moody, emotionally unstable",
        "agreeableness": "Cooperative, trusting, helpful",
        "conscientiousness": "Organized, disciplined, responsible",
        "openness": "Creative, curious, open to new experiences"
    }

    print(f"\nBig-5 Personality Traits:")
    print(f"{'-'*70}")

    for trait, score in predictions.items():
        desc = descriptions.get(trait, "")

        # Add indicator based on score
        if score > 0.7:
            indicator = "HIGH"
        elif score > 0.4:
            indicator = "MED "
        else:
            indicator = "LOW "

        print(f"{indicator} {trait.capitalize():>17}: {score:+.3f}  ({desc})")

    print(f"\n{'='*70}")
    print(f"IMPORTANT DISCLAIMER:")
    print(f"   These predictions are based on facial appearance alone")
    print(f"   and should be interpreted with caution. Personality is")
    print(f"   complex and cannot be accurately determined from photos alone.")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict personality from images or videos using trained ResNet model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from image
  python scripts/predict_cli.py image.jpg

  # Predict from video
  python scripts/predict_cli.py video.mp4

  # Save processed image
  python scripts/predict_cli.py image.jpg --save-processed

  # Disable face detection
  python scripts/predict_cli.py image.jpg --no-face-detection

  # Disable Test Time Augmentation
  python scripts/predict_cli.py image.jpg --no-tta

  # Use different model checkpoint
  python scripts/predict_cli.py image.jpg --checkpoint path/to/model.pt
        """
    )

    parser.add_argument("input_path", help="Path to image or video file")
    parser.add_argument("--checkpoint", type=str,
                       default=None,
                       help="Path to model checkpoint (uses config default if not specified)")
    parser.add_argument("--method", choices=["face", "middle", "face+middle"],
                       default="face+middle",
                       help="Face detection method (default: face+middle)")
    parser.add_argument("--no-face-detection", action="store_true",
                       help="Disable face detection (use full image)")
    parser.add_argument("--no-tta", action="store_true",
                       help="Disable Test Time Augmentation")
    parser.add_argument("--save-processed", action="store_true",
                       help="Save the processed image after face detection")
    parser.add_argument("--output-dir", default="processed_outputs",
                       help="Directory to save processed images")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                       default="auto",
                       help="Device to use (default: auto)")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Using device: {args.device}")

    # Load model
    try:
        print("Loading model...")
        model_manager = get_model_manager()

        if not model_manager.is_loaded:
            checkpoint_path = args.checkpoint or settings.MODEL_CHECKPOINT_PATH
            model_manager.load_model(
                checkpoint_path=checkpoint_path,
                device=args.device
            )

        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Determine method
    if args.no_face_detection:
        method = "middle"
        print("Face detection disabled - using full image")
    else:
        method = args.method
        print(f"Face detection method: {method}")

    # Create prediction service
    prediction_service = PredictionService(
        model_manager=model_manager,
        preprocessing_service=PreprocessingService()
    )

    # Run prediction
    try:
        print(f"\nProcessing: {input_path}")
        results = prediction_service.predict_from_file(
            file_path=str(input_path),
            use_tta=not args.no_tta,
            preprocessing_method=method
        )

        # Display results
        display_results(results, str(input_path))

        # Save processed image if requested
        if args.save_processed:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"processed_{input_path.name}"
            print(f"Processed image would be saved to: {output_path}")
            # TODO: Implement saving preprocessed image

        return 0

    except Exception as e:
        print(f"\nError during prediction: {e}")
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
