"""
High-level prediction service
Orchestrates preprocessing, model inference, and result formatting
"""
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, List, Union
from PIL import Image
import hashlib

from src.services.model_manager import ModelManager
from src.services.preprocessing_service import PreprocessingService
from src.services.interpretation_service import get_interpretation_service
from src.services.derived_metrics_service import get_derived_metrics_service
from src.core.personality_predictor import PersonalityPredictor, PredictionResult
from src.core.video_predictor import VideoPersonalityPredictor
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import PredictionError, ModelNotLoadedException

logger = get_logger(__name__)


class PredictionService:
    """
    High-level prediction service
    Orchestrates the full prediction pipeline
    """

    def __init__(
        self,
        model_manager: ModelManager,
        preprocessing_service: Optional[PreprocessingService] = None
    ):
        """
        Initialize prediction service

        Args:
            model_manager: Model manager instance
            preprocessing_service: Preprocessing service (creates new if None)
        """
        self.model_manager = model_manager
        self.preprocessing_service = preprocessing_service or PreprocessingService()

        # Will be initialized when model is loaded
        self._predictor: Optional[PersonalityPredictor] = None
        self._video_predictor: Optional[VideoPersonalityPredictor] = None

        logger.info("PredictionService initialized")

    def _ensure_predictor(self) -> Union[PersonalityPredictor, VideoPersonalityPredictor]:
        """Ensure predictor is initialized with loaded model"""
        if not self.model_manager.is_loaded:
            raise ModelNotLoadedException("Model must be loaded before prediction")

        # Check if VAT model - use video predictor
        if self.model_manager.is_vat:
            if self._video_predictor is None:
                self._video_predictor = VideoPersonalityPredictor(
                    model=self.model_manager.model,
                    device=self.model_manager.device,
                    trait_labels=self.model_manager.trait_labels,
                    use_tta=settings.USE_TTA
                )
                logger.info("VideoPersonalityPredictor initialized for VAT model")
            return self._video_predictor
        else:
            # ResNet model - use image predictor
            if self._predictor is None:
                self._predictor = PersonalityPredictor(
                    model=self.model_manager.model,
                    device=self.model_manager.device,
                    trait_labels=self.model_manager.trait_labels,
                    image_size=settings.effective_image_size  # Use effective size (256 for ResNet)
                )
                logger.info(f"PersonalityPredictor initialized for ResNet model (image_size={settings.effective_image_size})")
            return self._predictor

    def predict_from_file(
        self,
        file_path: str,
        use_tta: Optional[bool] = None,
        preprocessing_method: Optional[str] = None
    ) -> Dict:
        """
        Predict personality from image or video file

        Args:
            file_path: Path to image or video file
            use_tta: Whether to use Test Time Augmentation (uses config if None)
            preprocessing_method: Preprocessing method (uses config if None)

        Returns:
            Dictionary with prediction results and metadata
        """
        use_tta = use_tta if use_tta is not None else settings.USE_TTA
        predictor = self._ensure_predictor()

        try:
            # Preprocess
            logger.info(f"Preprocessing file: {file_path}")
            processed_image, preprocessing_meta = self.preprocessing_service.preprocess_from_file(
                file_path,
                method=preprocessing_method
            )

            # Predict
            logger.info("Running prediction...")
            prediction_result = predictor.predict(processed_image, use_tta=use_tta)

            # Combine results
            result = {
                "success": True,
                "predictions": prediction_result.traits,
                "metadata": {
                    "file_path": str(file_path),
                    "preprocessing": preprocessing_meta,
                    "prediction": prediction_result.metadata
                }
            }

            logger.info(f"Prediction completed: {prediction_result.traits}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Prediction failed: {str(e)}")

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        use_tta: Optional[bool] = None,
        preprocessing_method: Optional[str] = None
    ) -> Dict:
        """
        Predict personality from image bytes (API uploads)

        Args:
            image_bytes: Image data as bytes
            filename: Original filename (for metadata)
            use_tta: Whether to use Test Time Augmentation
            preprocessing_method: Preprocessing method

        Returns:
            Dictionary with prediction results and metadata
        """
        use_tta = use_tta if use_tta is not None else settings.USE_TTA
        predictor = self._ensure_predictor()

        try:
            # Generate hash for caching
            image_hash = hashlib.md5(image_bytes).hexdigest()

            # Preprocess
            logger.info(f"Preprocessing uploaded image (hash: {image_hash[:8]})")
            processed_image, preprocessing_meta = self.preprocessing_service.preprocess_from_bytes(
                image_bytes,
                method=preprocessing_method
            )

            # Predict
            logger.info("Running prediction...")
            prediction_result = predictor.predict(processed_image, use_tta=use_tta)

            # Combine results
            result = {
                "success": True,
                "predictions": prediction_result.traits,
                "metadata": {
                    "filename": filename,
                    "image_hash": image_hash,
                    "preprocessing": preprocessing_meta,
                    "prediction": prediction_result.metadata
                }
            }

            logger.info(f"Prediction completed: {prediction_result.traits}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Prediction failed: {str(e)}")

    def predict_from_pil(
        self,
        pil_image: Image.Image,
        use_tta: Optional[bool] = None,
        preprocessing_method: Optional[str] = None
    ) -> Dict:
        """
        Predict personality from PIL Image (already loaded)

        Args:
            pil_image: PIL Image in RGB format
            use_tta: Whether to use Test Time Augmentation
            preprocessing_method: Preprocessing method

        Returns:
            Dictionary with prediction results and metadata
        """
        use_tta = use_tta if use_tta is not None else settings.USE_TTA
        predictor = self._ensure_predictor()

        try:
            import numpy as np

            # Convert PIL to numpy
            image_rgb = np.array(pil_image)

            # Preprocess
            logger.info("Preprocessing PIL image...")
            processed_image, preprocessing_meta = self.preprocessing_service.preprocess_image(
                image_rgb,
                method=preprocessing_method or settings.FACE_DETECTION_METHOD
            )

            # Predict
            logger.info("Running prediction...")
            prediction_result = predictor.predict(processed_image, use_tta=use_tta)

            # Combine results
            result = {
                "success": True,
                "predictions": prediction_result.traits,
                "metadata": {
                    "preprocessing": preprocessing_meta,
                    "prediction": prediction_result.metadata
                }
            }

            logger.info(f"Prediction completed: {prediction_result.traits}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Prediction failed: {str(e)}")

    def predict_batch(
        self,
        pil_images: List[Image.Image],
        use_tta: Optional[bool] = None
    ) -> List[Dict]:
        """
        Predict personality for multiple images

        Args:
            pil_images: List of PIL Images
            use_tta: Whether to use Test Time Augmentation

        Returns:
            List of prediction result dictionaries
        """
        use_tta = use_tta if use_tta is not None else settings.USE_TTA
        predictor = self._ensure_predictor()

        try:
            logger.info(f"Running batch prediction for {len(pil_images)} images...")

            # Predict batch
            prediction_results = predictor.predict_batch(pil_images, use_tta=use_tta)

            # Format results
            results = []
            for i, pred_result in enumerate(prediction_results):
                result = {
                    "success": True,
                    "predictions": pred_result.traits,
                    "metadata": {
                        "batch_index": i,
                        "prediction": pred_result.metadata
                    }
                }
                results.append(result)

            logger.info(f"Batch prediction completed for {len(results)} images")
            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Batch prediction failed: {str(e)}")

    def add_interpretations(
        self,
        result: Dict,
        include_summary: bool = True,
        generate_report: bool = False
    ) -> Dict:
        """
        Add T-score interpretations to prediction result

        Args:
            result: Prediction result dictionary
            include_summary: Whether to include personality summary
            generate_report: Whether to generate AI-powered narrative report

        Returns:
            Result with interpretations added
        """
        interpretation_service = get_interpretation_service()

        # Get predictions
        predictions = result.get("predictions", {})

        # Generate interpretations
        interpretations = interpretation_service.interpret_all_traits(predictions)

        # Format as dictionary
        interpretations_dict = interpretation_service.format_interpretation_dict(interpretations)

        # Add to result
        result["interpretations"] = interpretations_dict

        # Add summary if requested
        summary = None
        if include_summary:
            summary = interpretation_service.get_summary(interpretations)
            result["summary"] = summary

        # Add norms metadata for reproducibility
        norms_meta = interpretation_service.get_norms_metadata()
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["norms"] = norms_meta["norms"]
        result["metadata"]["cutoffs"] = norms_meta["cutoffs"]

        # Calculate relationship and empathy metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            relationship_metrics = derived_metrics_service.calculate_all_relationship_metrics(predictions)
            result["relationship_metrics"] = relationship_metrics
            logger.info("Relationship metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate relationship metrics: {e}", exc_info=True)
            result["relationship_metrics"] = None

        # Calculate work DNA and focus metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            work_metrics = derived_metrics_service.calculate_all_work_metrics(predictions)
            result["work_metrics"] = work_metrics
            logger.info("Work metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate work metrics: {e}", exc_info=True)
            result["work_metrics"] = None

        # Calculate creativity pulse metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            creativity_metrics = derived_metrics_service.calculate_all_creativity_metrics(predictions)
            result["creativity_metrics"] = creativity_metrics
            logger.info("Creativity metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate creativity metrics: {e}", exc_info=True)
            result["creativity_metrics"] = None

        # Calculate stress and resilience metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            stress_metrics = derived_metrics_service.calculate_all_stress_metrics(predictions)
            result["stress_metrics"] = stress_metrics
            logger.info("Stress metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate stress metrics: {e}", exc_info=True)
            result["stress_metrics"] = None

        # Calculate openness to experience metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            openness_metrics = derived_metrics_service.calculate_all_openness_metrics(predictions)
            result["openness_metrics"] = openness_metrics
            logger.info("Openness metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate openness metrics: {e}", exc_info=True)
            result["openness_metrics"] = None

        # Calculate learning and growth metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            learning_metrics = derived_metrics_service.calculate_all_learning_metrics(predictions)
            result["learning_metrics"] = learning_metrics
            logger.info("Learning metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate learning metrics: {e}", exc_info=True)
            result["learning_metrics"] = None

        # Generate AI insights if requested (async operation)
        if generate_report:
            # Import here to avoid circular imports
            from src.services.report_service import get_report_service
            import asyncio

            try:
                report_service = get_report_service()

                # Run async report generation
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, create a task
                    # This will be handled by the async route handler
                    result["_generate_report"] = True
                    result["_report_data"] = {
                        "interpretations": interpretations_dict,
                        "summary": summary
                    }
                else:
                    # Run synchronously
                    report_result = loop.run_until_complete(
                        report_service.generate_report(interpretations_dict, summary)
                    )
                    # Store only the overview insights (no coaching content)
                    result["insights"] = report_result.get("insights")
                    logger.info(f"AI insights added to result: {report_result.get('insights', {}).get('title', 'N/A')}")

                    # Route LLM-generated coaching content directly to metrics sections
                    if result.get("relationship_metrics") and report_result.get("relationship_coaching"):
                        coaching = report_result["relationship_coaching"]
                        result["relationship_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                        result["relationship_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                        # Accordion section fields
                        result["relationship_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                        result["relationship_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                        result["relationship_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                        result["relationship_metrics"]["strength"] = coaching.get("strength")
                        result["relationship_metrics"]["tradeoff"] = coaching.get("tradeoff")
                        result["relationship_metrics"]["growth_lever"] = coaching.get("growth_lever")
                        result["relationship_metrics"]["suitable_for"] = coaching.get("suitable_for")
                        logger.info("Routed LLM relationship coaching to metrics")

                    if result.get("work_metrics") and report_result.get("work_coaching"):
                        coaching = report_result["work_coaching"]
                        result["work_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                        result["work_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                        # Accordion section fields
                        result["work_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                        result["work_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                        result["work_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                        result["work_metrics"]["strength"] = coaching.get("strength")
                        result["work_metrics"]["tradeoff"] = coaching.get("tradeoff")
                        result["work_metrics"]["growth_lever"] = coaching.get("growth_lever")
                        result["work_metrics"]["suitable_for"] = coaching.get("suitable_for")
                        logger.info("Routed LLM work coaching to metrics")

                    if result.get("creativity_metrics") and report_result.get("creativity_coaching"):
                        coaching = report_result["creativity_coaching"]
                        result["creativity_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                        result["creativity_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                        # Accordion section fields
                        result["creativity_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                        result["creativity_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                        result["creativity_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                        result["creativity_metrics"]["strength"] = coaching.get("strength")
                        result["creativity_metrics"]["tradeoff"] = coaching.get("tradeoff")
                        result["creativity_metrics"]["growth_lever"] = coaching.get("growth_lever")
                        result["creativity_metrics"]["suitable_for"] = coaching.get("suitable_for")
                        logger.info("Routed LLM creativity coaching to metrics")

                    if result.get("stress_metrics") and report_result.get("stress_coaching"):
                        coaching = report_result["stress_coaching"]
                        result["stress_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                        result["stress_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                        # Accordion section fields
                        result["stress_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                        result["stress_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                        result["stress_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                        result["stress_metrics"]["strength"] = coaching.get("strength")
                        result["stress_metrics"]["tradeoff"] = coaching.get("tradeoff")
                        result["stress_metrics"]["growth_lever"] = coaching.get("growth_lever")
                        result["stress_metrics"]["suitable_for"] = coaching.get("suitable_for")
                        logger.info("Routed LLM stress coaching to metrics")

                    if result.get("openness_metrics") and report_result.get("openness_coaching"):
                        coaching = report_result["openness_coaching"]
                        result["openness_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                        result["openness_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                        # Accordion section fields
                        result["openness_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                        result["openness_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                        result["openness_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                        result["openness_metrics"]["strength"] = coaching.get("strength")
                        result["openness_metrics"]["tradeoff"] = coaching.get("tradeoff")
                        result["openness_metrics"]["growth_lever"] = coaching.get("growth_lever")
                        result["openness_metrics"]["suitable_for"] = coaching.get("suitable_for")
                        logger.info("Routed LLM openness coaching to metrics")

                    if result.get("learning_metrics") and report_result.get("learning_coaching"):
                        coaching = report_result["learning_coaching"]
                        result["learning_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                        result["learning_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                        # Accordion section fields
                        result["learning_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                        result["learning_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                        result["learning_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                        result["learning_metrics"]["strength"] = coaching.get("strength")
                        result["learning_metrics"]["tradeoff"] = coaching.get("tradeoff")
                        result["learning_metrics"]["growth_lever"] = coaching.get("growth_lever")
                        result["learning_metrics"]["suitable_for"] = coaching.get("suitable_for")
                        logger.info("Routed LLM learning coaching to metrics")

                    if report_result.get("voice_coaching"):
                        coaching = report_result["voice_coaching"]
                        # Initialize audio_metrics if it doesn't exist but we have voice coaching
                        if not result.get("audio_metrics"):
                            result["audio_metrics"] = {"indicators": {}}
                            logger.info("Initialized audio_metrics for voice coaching")
                        result["audio_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                        result["audio_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                        # Accordion section fields
                        result["audio_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                        result["audio_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                        result["audio_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                        result["audio_metrics"]["strength"] = coaching.get("strength")
                        result["audio_metrics"]["tradeoff"] = coaching.get("tradeoff")
                        result["audio_metrics"]["growth_lever"] = coaching.get("growth_lever")
                        result["audio_metrics"]["suitable_for"] = coaching.get("suitable_for")
                        logger.info("Routed LLM voice coaching to audio_metrics")

            except Exception as e:
                logger.error(f"Failed to generate AI insights: {e}", exc_info=True)
                result["insights"] = None
                result["report_error"] = str(e)

        return result

    async def add_interpretations_async(
        self,
        result: Dict,
        include_summary: bool = True,
        generate_report: bool = False,
        multimodal_data: Optional[Dict] = None
    ) -> Dict:
        """
        Async version of add_interpretations that supports AI report generation

        Args:
            result: Prediction result dictionary
            include_summary: Whether to include personality summary
            generate_report: Whether to generate AI-powered narrative report
            multimodal_data: Optional dict with frames_base64 and transcript for multimodal insights

        Returns:
            Result with interpretations and optional AI report added
        """
        interpretation_service = get_interpretation_service()

        # Get predictions
        predictions = result.get("predictions", {})

        # Generate interpretations
        interpretations = interpretation_service.interpret_all_traits(predictions)

        # Format as dictionary
        interpretations_dict = interpretation_service.format_interpretation_dict(interpretations)

        # Add to result
        result["interpretations"] = interpretations_dict

        # Add summary if requested
        summary = None
        if include_summary:
            summary = interpretation_service.get_summary(interpretations)
            result["summary"] = summary

        # Add norms metadata for reproducibility
        norms_meta = interpretation_service.get_norms_metadata()
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["norms"] = norms_meta["norms"]
        result["metadata"]["cutoffs"] = norms_meta["cutoffs"]

        # Calculate relationship and empathy metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            relationship_metrics = derived_metrics_service.calculate_all_relationship_metrics(predictions)
            result["relationship_metrics"] = relationship_metrics
            logger.info("Relationship metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate relationship metrics: {e}", exc_info=True)
            result["relationship_metrics"] = None

        # Calculate work DNA and focus metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            work_metrics = derived_metrics_service.calculate_all_work_metrics(predictions)
            result["work_metrics"] = work_metrics
            logger.info("Work metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate work metrics: {e}", exc_info=True)
            result["work_metrics"] = None

        # Calculate creativity pulse metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            creativity_metrics = derived_metrics_service.calculate_all_creativity_metrics(predictions)
            result["creativity_metrics"] = creativity_metrics
            logger.info("Creativity metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate creativity metrics: {e}", exc_info=True)
            result["creativity_metrics"] = None

        # Calculate stress and resilience metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            stress_metrics = derived_metrics_service.calculate_all_stress_metrics(predictions)
            result["stress_metrics"] = stress_metrics
            logger.info("Stress metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate stress metrics: {e}", exc_info=True)
            result["stress_metrics"] = None

        # Calculate openness to experience metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            openness_metrics = derived_metrics_service.calculate_all_openness_metrics(predictions)
            result["openness_metrics"] = openness_metrics
            logger.info("Openness metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate openness metrics: {e}", exc_info=True)
            result["openness_metrics"] = None

        # Calculate learning and growth metrics from OCEAN scores
        try:
            derived_metrics_service = get_derived_metrics_service()
            learning_metrics = derived_metrics_service.calculate_all_learning_metrics(predictions)
            result["learning_metrics"] = learning_metrics
            logger.info("Learning metrics calculated successfully")
        except Exception as e:
            logger.error(f"Failed to calculate learning metrics: {e}", exc_info=True)
            result["learning_metrics"] = None

        # Generate AI insights if requested
        if generate_report:
            from src.services.report_service import get_report_service

            try:
                report_service = get_report_service()

                # Use multimodal analysis if data is available
                if multimodal_data and multimodal_data.get("frames_base64"):
                    assessment_metadata = multimodal_data.get("assessment_metadata")
                    has_assessment = assessment_metadata is not None and len(assessment_metadata.get("question_responses", [])) > 0
                    logger.info(f"Using multimodal report generation with frames and transcript (has_assessment={has_assessment})")
                    report_result = await report_service.generate_multimodal_report(
                        frames_base64=multimodal_data.get("frames_base64", []),
                        transcript=multimodal_data.get("transcript", ""),
                        ocean_scores=predictions,
                        audio_metrics=multimodal_data.get("audio_metrics"),
                        assessment_metadata=assessment_metadata
                    )
                    # Add multimodal metadata
                    result["metadata"]["multimodal"] = {
                        "frames_analyzed": len(multimodal_data.get("frames_base64", [])),
                        "has_transcript": bool(multimodal_data.get("transcript")),
                        "transcript_length": len(multimodal_data.get("transcript", "")),
                        "has_assessment": has_assessment,
                        "questions_answered": assessment_metadata.get("questions_answered", 0) if assessment_metadata else 0
                    }
                else:
                    # Fallback to score-based report
                    logger.info("Using score-based report generation")
                    report_result = await report_service.generate_report(interpretations_dict, summary)

                # Store only the overview insights (no coaching content)
                result["insights"] = report_result.get("insights")
                logger.info(f"AI insights added to result: {report_result.get('insights', {}).get('title', 'N/A')}")

                # Route LLM-generated coaching content directly to metrics sections
                if result.get("relationship_metrics") and report_result.get("relationship_coaching"):
                    coaching = report_result["relationship_coaching"]
                    result["relationship_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                    result["relationship_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                    # Accordion section fields
                    result["relationship_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                    result["relationship_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                    result["relationship_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                    result["relationship_metrics"]["strength"] = coaching.get("strength")
                    result["relationship_metrics"]["tradeoff"] = coaching.get("tradeoff")
                    result["relationship_metrics"]["growth_lever"] = coaching.get("growth_lever")
                    result["relationship_metrics"]["suitable_for"] = coaching.get("suitable_for")
                    logger.info("Routed LLM relationship coaching to metrics")

                if result.get("work_metrics") and report_result.get("work_coaching"):
                    coaching = report_result["work_coaching"]
                    result["work_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                    result["work_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                    # Accordion section fields
                    result["work_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                    result["work_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                    result["work_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                    result["work_metrics"]["strength"] = coaching.get("strength")
                    result["work_metrics"]["tradeoff"] = coaching.get("tradeoff")
                    result["work_metrics"]["growth_lever"] = coaching.get("growth_lever")
                    result["work_metrics"]["suitable_for"] = coaching.get("suitable_for")
                    logger.info("Routed LLM work coaching to metrics")

                if result.get("creativity_metrics") and report_result.get("creativity_coaching"):
                    coaching = report_result["creativity_coaching"]
                    result["creativity_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                    result["creativity_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                    # Accordion section fields
                    result["creativity_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                    result["creativity_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                    result["creativity_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                    result["creativity_metrics"]["strength"] = coaching.get("strength")
                    result["creativity_metrics"]["tradeoff"] = coaching.get("tradeoff")
                    result["creativity_metrics"]["growth_lever"] = coaching.get("growth_lever")
                    result["creativity_metrics"]["suitable_for"] = coaching.get("suitable_for")
                    logger.info("Routed LLM creativity coaching to metrics")

                if result.get("stress_metrics") and report_result.get("stress_coaching"):
                    coaching = report_result["stress_coaching"]
                    result["stress_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                    result["stress_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                    # Accordion section fields
                    result["stress_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                    result["stress_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                    result["stress_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                    result["stress_metrics"]["strength"] = coaching.get("strength")
                    result["stress_metrics"]["tradeoff"] = coaching.get("tradeoff")
                    result["stress_metrics"]["growth_lever"] = coaching.get("growth_lever")
                    result["stress_metrics"]["suitable_for"] = coaching.get("suitable_for")
                    logger.info("Routed LLM stress coaching to metrics")

                if result.get("openness_metrics") and report_result.get("openness_coaching"):
                    coaching = report_result["openness_coaching"]
                    result["openness_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                    result["openness_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                    # Accordion section fields
                    result["openness_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                    result["openness_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                    result["openness_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                    result["openness_metrics"]["strength"] = coaching.get("strength")
                    result["openness_metrics"]["tradeoff"] = coaching.get("tradeoff")
                    result["openness_metrics"]["growth_lever"] = coaching.get("growth_lever")
                    result["openness_metrics"]["suitable_for"] = coaching.get("suitable_for")
                    logger.info("Routed LLM openness coaching to metrics")

                if result.get("learning_metrics") and report_result.get("learning_coaching"):
                    coaching = report_result["learning_coaching"]
                    result["learning_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                    result["learning_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                    # Accordion section fields
                    result["learning_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                    result["learning_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                    result["learning_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                    result["learning_metrics"]["strength"] = coaching.get("strength")
                    result["learning_metrics"]["tradeoff"] = coaching.get("tradeoff")
                    result["learning_metrics"]["growth_lever"] = coaching.get("growth_lever")
                    result["learning_metrics"]["suitable_for"] = coaching.get("suitable_for")
                    logger.info("Routed LLM learning coaching to metrics")

                logger.info(f"[VOICE DEBUG] report_result keys: {report_result.keys() if report_result else 'None'}")
                logger.info(f"[VOICE DEBUG] voice_coaching exists: {bool(report_result.get('voice_coaching'))}")
                if report_result.get("voice_coaching"):
                    coaching = report_result["voice_coaching"]
                    logger.info(f"[VOICE DEBUG] voice_coaching content: {coaching}")
                    # Initialize audio_metrics if it doesn't exist but we have voice coaching
                    if not result.get("audio_metrics"):
                        result["audio_metrics"] = {"indicators": {}}
                        logger.info("Initialized audio_metrics for voice coaching")
                    result["audio_metrics"]["coach_recommendation"] = coaching.get("coach_recommendation")
                    result["audio_metrics"]["actionable_steps"] = coaching.get("actionable_steps")
                    # Accordion section fields
                    result["audio_metrics"]["snapshot_insight"] = coaching.get("snapshot_insight")
                    result["audio_metrics"]["behavioral_patterns"] = coaching.get("behavioral_patterns")
                    result["audio_metrics"]["how_others_experience"] = coaching.get("how_others_experience")
                    result["audio_metrics"]["strength"] = coaching.get("strength")
                    result["audio_metrics"]["tradeoff"] = coaching.get("tradeoff")
                    result["audio_metrics"]["growth_lever"] = coaching.get("growth_lever")
                    result["audio_metrics"]["suitable_for"] = coaching.get("suitable_for")
                    logger.info(f"[VOICE DEBUG] result audio_metrics after routing: {result['audio_metrics']}")
                    logger.info("Routed LLM voice coaching to audio_metrics")
                else:
                    logger.warning("[VOICE DEBUG] No voice_coaching in report_result!")

            except Exception as e:
                logger.error(f"Failed to generate AI insights: {e}", exc_info=True)
                result["insights"] = None
                result["report_error"] = str(e)

        return result

    def get_trait_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all personality traits"""
        predictor = self._ensure_predictor()
        if isinstance(predictor, VideoPersonalityPredictor):
            return {
                trait: PersonalityPredictor.get_trait_description(trait)
                for trait in predictor.trait_labels
            }
        return {
            trait: PersonalityPredictor.get_trait_description(trait)
            for trait in predictor.trait_labels
        }

    def predict_from_video(
        self,
        video_path: str,
        use_tta: Optional[bool] = None
    ) -> Dict:
        """
        Predict personality from video file using VAT model.

        Args:
            video_path: Path to video file
            use_tta: Whether to use Test Time Augmentation (uses config if None)

        Returns:
            Dictionary with prediction results and metadata

        Raises:
            PredictionError: If prediction fails
            ModelNotLoadedException: If model is not loaded or not VAT
        """
        if not self.model_manager.is_loaded:
            raise ModelNotLoadedException("Model must be loaded before prediction")

        if not self.model_manager.is_vat:
            raise PredictionError("VAT model required for video prediction")

        use_tta = use_tta if use_tta is not None else settings.USE_TTA

        try:
            # Preprocess video
            logger.info(f"Preprocessing video: {video_path}")
            frames, preprocessing_meta = self.preprocessing_service.preprocess_video_for_vat(
                video_path
            )

            # Get predictor
            predictor = self._ensure_predictor()
            if not isinstance(predictor, VideoPersonalityPredictor):
                raise PredictionError("Expected VideoPersonalityPredictor for VAT model")

            # Temporarily set TTA if specified differently from default
            original_tta = predictor.use_tta
            predictor.use_tta = use_tta

            # Predict
            logger.info("Running VAT prediction...")
            predictions = predictor.predict(frames)

            # Restore TTA setting
            predictor.use_tta = original_tta

            # Combine results
            result = {
                "success": True,
                "predictions": predictions,
                "metadata": {
                    "file_path": str(video_path),
                    "preprocessing": preprocessing_meta,
                    "prediction": {
                        "model": "vat",
                        "num_frames": settings.MODEL_NUM_FRAMES,
                        "tta_used": use_tta,
                        "device": str(self.model_manager.device)
                    }
                }
            }

            logger.info(f"Video prediction completed: {predictions}")
            return result

        except Exception as e:
            logger.error(f"Video prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Video prediction failed: {str(e)}")

    def predict_from_video_bytes(
        self,
        video_bytes: bytes,
        filename: Optional[str] = None,
        use_tta: Optional[bool] = None
    ) -> Dict:
        """
        Predict personality from video bytes (API uploads).

        Saves video to temp file since OpenCV requires file path.

        Args:
            video_bytes: Video data as bytes
            filename: Original filename (for extension detection)
            use_tta: Whether to use Test Time Augmentation

        Returns:
            Dictionary with prediction results and metadata

        Raises:
            PredictionError: If prediction fails
        """
        # Determine extension from filename
        ext = ".mp4"  # default
        if filename:
            path = Path(filename)
            if path.suffix.lower() in settings.ALLOWED_VIDEO_EXTENSIONS:
                ext = path.suffix.lower()

        # Save to temp file
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(video_bytes)
                temp_file = f.name

            logger.info(f"Saved video to temp file: {temp_file}")

            # Generate hash for metadata
            video_hash = hashlib.md5(video_bytes).hexdigest()

            # Run prediction based on model type
            if self.model_manager.is_vat:
                result = self.predict_from_video(temp_file, use_tta=use_tta)
            else:
                # ResNet: extract middle frame and predict
                result = self.predict_from_video_resnet(temp_file, use_tta=use_tta)

            # Update metadata
            result["metadata"]["filename"] = filename
            result["metadata"]["video_hash"] = video_hash
            result["metadata"]["file_type"] = "video_upload"

            return result

        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

    def predict_from_video_resnet(
        self,
        video_path: str,
        use_tta: Optional[bool] = None,
        preprocessing_method: Optional[str] = None,
        num_frames: Optional[int] = None,
        return_raw_frames: bool = False
    ) -> Dict:
        """
        Predict personality from video using ResNet model with top-K frame selection.

        Extracts N evenly-spaced frames from the video, applies face detection
        to each, runs ResNet prediction on all frames, then selects the top K
        frames with the strongest predictions (highest sum of trait scores)
        and averages only those K frames for the final result.

        Args:
            video_path: Path to video file
            use_tta: Whether to use Test Time Augmentation (uses config if None)
            preprocessing_method: Face detection method (uses config if None)
            num_frames: Number of frames to extract (uses config if None)
            return_raw_frames: If True, include raw frames in result for multimodal reuse

        Returns:
            Dictionary with averaged prediction results and metadata.
            If return_raw_frames=True, includes 'raw_frames' key with List[np.ndarray]

        Raises:
            PredictionError: If prediction fails
        """
        if not self.model_manager.is_loaded:
            raise ModelNotLoadedException("Model must be loaded before prediction")

        if self.model_manager.is_vat:
            raise PredictionError("This method is for ResNet model. Use predict_from_video for VAT.")

        use_tta = use_tta if use_tta is not None else settings.USE_TTA
        preprocessing_method = preprocessing_method or settings.effective_face_detection
        num_frames = num_frames or settings.RESNET_NUM_FRAMES

        try:
            # Preprocess video for ResNet (extract multiple frames + face detection)
            logger.info(f"Preprocessing video for ResNet: {video_path} ({num_frames} frames)")
            processed_images, preprocessing_meta, raw_frames = self.preprocessing_service.preprocess_video_for_resnet(
                video_path,
                method=preprocessing_method,
                num_frames=num_frames,
                return_raw_frames=return_raw_frames
            )

            # Get predictor
            predictor = self._ensure_predictor()

            # Run prediction on each frame and collect scores
            logger.info(f"Running ResNet prediction on {len(processed_images)} frames...")
            all_predictions = []
            trait_labels = self.model_manager.trait_labels

            for i, image in enumerate(processed_images):
                prediction_result = predictor.predict(image, use_tta=use_tta)
                # Store prediction with frame index and strength score
                pred_dict = prediction_result.traits
                strength = sum(pred_dict.values())  # Sum of all trait scores as strength
                all_predictions.append({
                    "frame_idx": i,
                    "traits": pred_dict,
                    "strength": strength
                })
                logger.debug(f"Frame {i+1}/{len(processed_images)}: {pred_dict} (strength: {strength:.3f})")

            # Top-K selection: use only the K frames with highest prediction strength
            top_k = settings.RESNET_TOP_K
            if top_k > 0 and top_k < len(all_predictions):
                # Sort by strength descending and take top K
                sorted_predictions = sorted(all_predictions, key=lambda x: x["strength"], reverse=True)
                selected_predictions = sorted_predictions[:top_k]
                selected_indices = [p["frame_idx"] for p in selected_predictions]
                aggregation_method = f"top-{top_k}"
                logger.info(f"Selected top-{top_k} frames by strength: indices {selected_indices}")
            else:
                # Use all frames (fallback to mean)
                selected_predictions = all_predictions
                selected_indices = list(range(len(all_predictions)))
                aggregation_method = "mean"

            # Average the predictions across selected frames
            averaged_traits = {}
            for trait in trait_labels:
                trait_scores = [p["traits"][trait] for p in selected_predictions]
                averaged_traits[trait] = sum(trait_scores) / len(trait_scores)

            # Calculate per-trait standard deviation across ALL frames (for metadata insight)
            trait_std = {}
            for trait in trait_labels:
                trait_scores = [p["traits"][trait] for p in all_predictions]
                mean = sum(trait_scores) / len(trait_scores)
                variance = sum((x - mean) ** 2 for x in trait_scores) / len(trait_scores)
                trait_std[trait] = variance ** 0.5

            # Combine results
            result = {
                "success": True,
                "predictions": averaged_traits,
                "metadata": {
                    "file_path": str(video_path),
                    "preprocessing": preprocessing_meta,
                    "prediction": {
                        "model": "resnet",
                        "backbone": settings.MODEL_BACKBONE,
                        "image_size": settings.effective_image_size,
                        "tta_used": use_tta,
                        "device": str(self.model_manager.device),
                        "num_frames_extracted": len(processed_images),
                        "num_frames_used": len(selected_predictions),
                        "selected_frame_indices": selected_indices,
                        "aggregation_method": aggregation_method,
                        "per_trait_std_all_frames": trait_std
                    }
                }
            }

            # Include only the selected top-K raw frames for multimodal reuse
            if return_raw_frames and raw_frames:
                # Only pass the top-K frames that were used for prediction to LLM
                result["raw_frames"] = [raw_frames[i] for i in selected_indices if i < len(raw_frames)]
                logger.info(f"Passing {len(result['raw_frames'])} selected frames for multimodal analysis")

            logger.info(f"ResNet video prediction completed ({aggregation_method} over {len(selected_predictions)}/{len(processed_images)} frames): {averaged_traits}")
            return result

        except Exception as e:
            logger.error(f"ResNet video prediction failed: {e}", exc_info=True)
            raise PredictionError(f"ResNet video prediction failed: {str(e)}")

    async def predict_from_video_resnet_async(
        self,
        video_path: str,
        use_tta: Optional[bool] = None,
        preprocessing_method: Optional[str] = None,
        num_frames: Optional[int] = None,
        return_raw_frames: bool = False
    ) -> Dict:
        """
        Async version of predict_from_video_resnet.

        Runs the synchronous prediction in a thread pool to enable
        parallel execution with other async operations.

        Args:
            video_path: Path to video file
            use_tta: Whether to use Test Time Augmentation
            preprocessing_method: Face detection method
            num_frames: Number of frames to extract
            return_raw_frames: If True, include raw frames for multimodal reuse

        Returns:
            Dictionary with prediction results
        """
        import asyncio
        return await asyncio.to_thread(
            self.predict_from_video_resnet,
            video_path,
            use_tta,
            preprocessing_method,
            num_frames,
            return_raw_frames
        )
