import cv2
import numpy as np


def apply_histogram_equalization(image):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    return cv2.equalizeHist(image)


def apply_power_law_transformation(image, gamma=1.2):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    if gamma <= 0:
        raise ValueError("Gamma must be strictly positive.")

    lookup_table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, lookup_table)


def apply_local_gamma_correction(image, block_size=32, clip_limit=2.0):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    tile_grid = (max(2, image.shape[0] // block_size), max(2, image.shape[1] // block_size))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def apply_laplacian_sharpening(image, strength=0.5):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    float_image = np.float32(image)
    laplacian_response = cv2.filter2D(float_image, cv2.CV_32F, laplacian_kernel)
    sharpened = float_image + strength * laplacian_response
    clipped = np.clip(sharpened, 0, 255)
    return clipped.astype(np.uint8)


def enhance_fingerprint(image, use_hist_eq=False, gamma=1.1, use_clahe=True, laplacian_strength=0.35):
    current = image.copy()
    intermediates = {"input": current}

    if use_clahe:
        current = apply_local_gamma_correction(current, block_size=32, clip_limit=2.0)
        intermediates["clahe"] = current.copy()

    if use_hist_eq:
        current = apply_histogram_equalization(current)
        intermediates["hist_eq"] = current.copy()

    if abs(gamma - 1.0) > 1e-4:
        current = apply_power_law_transformation(current, gamma=gamma)
        intermediates["gamma"] = current.copy()

    if laplacian_strength > 0:
        current = apply_laplacian_sharpening(current, strength=laplacian_strength)
        intermediates["sharpened"] = current.copy()

    intermediates["final_enhanced"] = current
    return current, intermediates

