import os
from pathlib import Path
import cv2
import numpy as np


def load_image(image_path):
    resolved_path = Path(image_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Specified path is not a file: {resolved_path}")

    image_array = cv2.imread(str(resolved_path), cv2.IMREAD_UNCHANGED)
    if image_array is None:
        raise ValueError(f"Failed to decode image from path: {resolved_path}")

    if image_array.size == 0:
        raise ValueError("Loaded image contains no pixels (empty array).")

    original_preserved = image_array.copy()

    if len(image_array.shape) == 2:
        if image_array.dtype != np.uint8:
            normalized = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
            grayscale_image = normalized.astype(np.uint8)
        else:
            grayscale_image = image_array.copy()
    elif len(image_array.shape) == 3:
        channels = image_array.shape[2]
        if channels == 3:
            grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            grayscale_image = image_array[:, :, 0]
        else:
            raise ValueError(f"Unsupported number of image channels: {channels}")

        if grayscale_image.dtype != np.uint8:
            normalized = cv2.normalize(grayscale_image, None, 0, 255, cv2.NORM_MINMAX)
            grayscale_image = normalized.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {image_array.shape}")

    if grayscale_image.shape[0] < 10 or grayscale_image.shape[1] < 10:
        raise ValueError(f"Image dimensions too small: {grayscale_image.shape}")

    return original_preserved, grayscale_image


def save_image(output_path, image):
    if image is None or image.size == 0:
        raise ValueError("Cannot save an empty or None image.")

    target_path = Path(output_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(target_path), image)
    if not success:
        raise IOError(f"Failed to write image to {target_path}")

    return str(target_path)

