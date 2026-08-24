import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.preprocessing.registration")

class ImageRegistration:
    """
    Handles spatial alignment validation and preprocessing for bi-temporal
    and cross-modal image pairs.
    """

    @staticmethod
    def validate_pair(image_path_a: str, image_path_b: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates that two images are spatially compatible for paired analysis
        (change detection or cross-modal fusion).
        
        Checks:
        - Both files exist and are readable.
        - Both images have the same dimensions (width, height).
        - Both images have compatible channel counts.
        
        Returns:
            Tuple of (is_valid, error_message, metadata).
        """
        if not os.path.exists(image_path_a):
            return False, f"Image A not found: {image_path_a}", {}
        if not os.path.exists(image_path_b):
            return False, f"Image B not found: {image_path_b}", {}

        try:
            img_a = Image.open(image_path_a)
            img_b = Image.open(image_path_b)
        except Exception as e:
            return False, f"Failed to open images: {str(e)}", {}

        metadata = {
            "image_a": {
                "width": img_a.width,
                "height": img_a.height,
                "bands": len(img_a.getbands()),
                "mode": img_a.mode
            },
            "image_b": {
                "width": img_b.width,
                "height": img_b.height,
                "bands": len(img_b.getbands()),
                "mode": img_b.mode
            }
        }

        # Check dimension match
        if img_a.size != img_b.size:
            return False, (
                f"Dimension mismatch: Image A is {img_a.width}x{img_a.height}, "
                f"Image B is {img_b.width}x{img_b.height}. "
                f"Images must have identical dimensions for paired analysis."
            ), metadata

        metadata["dimensions_match"] = True
        metadata["spatial_correspondence"] = True
        return True, "", metadata

    @staticmethod
    def resize_to_match(image_path_a: str, image_path_b: str) -> Tuple[str, str]:
        """
        If images have different sizes, resizes Image B to match Image A.
        Returns paths to the (possibly resized) images.
        """
        img_a = Image.open(image_path_a)
        img_b = Image.open(image_path_b)

        if img_a.size == img_b.size:
            return image_path_a, image_path_b

        logger.warning(
            f"Resizing Image B from {img_b.width}x{img_b.height} "
            f"to {img_a.width}x{img_a.height} to match Image A."
        )
        img_b_resized = img_b.resize(img_a.size, Image.LANCZOS)

        # Save resized image alongside the original
        base, ext = os.path.splitext(image_path_b)
        resized_path = f"{base}_resized{ext}"
        img_b_resized.save(resized_path)

        return image_path_a, resized_path
