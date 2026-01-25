"""
Face detection and bounding box expansion
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.exceptions import FaceDetectionError

logger = get_logger(__name__)


@dataclass
class BoundingBox:
    """Represents a face bounding box"""
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        """Calculate bounding box area"""
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple (x, y, w, h)"""
        return (self.x, self.y, self.width, self.height)


class FaceDetector:
    """Face detection using OpenCV Haar Cascades"""

    def __init__(self):
        """Initialize face detector with Haar cascade"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            raise RuntimeError(f"Failed to load face cascade from {cascade_path}")

        logger.info("Face detector initialized successfully")

    def detect_largest_face(
        self,
        image_bgr: np.ndarray,
        min_size: Tuple[int, int] = (60, 60),
        scale_factor: float = 1.1,
        min_neighbors: int = 5
    ) -> Optional[BoundingBox]:
        """
        Detect the largest face in the image

        Args:
            image_bgr: Input image in BGR format
            min_size: Minimum face size (width, height)
            scale_factor: Scale factor for detection
            min_neighbors: Minimum neighbors for detection

        Returns:
            BoundingBox of largest face or None if no face detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            logger.debug("No faces detected in image")
            return None

        # Find largest face by area
        largest = max(faces, key=lambda b: b[2] * b[3])
        bbox = BoundingBox(x=int(largest[0]), y=int(largest[1]),
                          width=int(largest[2]), height=int(largest[3]))

        logger.debug(f"Detected {len(faces)} face(s), largest: {bbox.width}x{bbox.height}")
        return bbox

    def expand_bbox(
        self,
        bbox: BoundingBox,
        image_width: int,
        image_height: int,
        expand_ratio: float = 0.35
    ) -> BoundingBox:
        """
        Expand bounding box by a given ratio

        Args:
            bbox: Original bounding box
            image_width: Image width for boundary checking
            image_height: Image height for boundary checking
            expand_ratio: Expansion ratio (0.35 = 35% larger)

        Returns:
            Expanded bounding box
        """
        # Calculate center
        cx = bbox.x + bbox.width / 2.0
        cy = bbox.y + bbox.height / 2.0

        # Calculate expanded side length (square bbox)
        side = int(max(bbox.width, bbox.height) * (1 + expand_ratio))

        # Calculate new top-left corner
        nx = int(max(0, cx - side / 2))
        ny = int(max(0, cy - side / 2))

        # Ensure bbox stays within image boundaries
        nx2 = int(min(image_width, nx + side))
        ny2 = int(min(image_height, ny + side))

        expanded = BoundingBox(
            x=nx,
            y=ny,
            width=nx2 - nx,
            height=ny2 - ny
        )

        logger.debug(f"Expanded bbox: {bbox.width}x{bbox.height} -> {expanded.width}x{expanded.height}")
        return expanded

    def crop_face(
        self,
        image_bgr: np.ndarray,
        bbox: BoundingBox,
        expand_ratio: float = 0.35
    ) -> np.ndarray:
        """
        Detect and crop face from image

        Args:
            image_bgr: Input image in BGR format
            bbox: Face bounding box
            expand_ratio: Expansion ratio for cropping

        Returns:
            Cropped face region in BGR format
        """
        img_h, img_w = image_bgr.shape[:2]

        # Expand bounding box
        expanded = self.expand_bbox(bbox, img_w, img_h, expand_ratio)

        # Crop face region
        face_crop = image_bgr[
            expanded.y:expanded.y + expanded.height,
            expanded.x:expanded.x + expanded.width
        ]

        if face_crop.size == 0:
            raise FaceDetectionError("Cropped face region is empty")

        return face_crop
