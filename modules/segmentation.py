import cv2
import numpy as np


def compute_local_variance_mask(image, window_size=15, variance_threshold=80.0):
    float_img = image.astype(np.float32)
    kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
    mean = cv2.filter2D(float_img, -1, kernel)
    mean_sq = cv2.filter2D(float_img ** 2, -1, kernel)
    variance = mean_sq - (mean ** 2)
    variance = np.maximum(variance, 0)
    mask = (variance > variance_threshold).astype(np.uint8) * 255
    return mask


def segment_fingerprint_region(image, block_size=17, c_offset=8, morph_ksize=9, min_area_ratio=0.04):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    adaptive_mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_offset
    )

    var_mask = compute_local_variance_mask(image, window_size=15, variance_threshold=80.0)
    combined_raw = cv2.bitwise_or(adaptive_mask, var_mask)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    closed = cv2.morphologyEx(combined_raw, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize // 2 * 2 + 1, morph_ksize // 2 * 2 + 1))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image.size * min_area_ratio
    max_area = image.size * 0.92

    refined_mask = np.zeros_like(image, dtype=np.uint8)
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        cv2.drawContours(refined_mask, [hull], -1, 255, thickness=-1)
    else:
        candidates = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        if candidates:
            largest_c = max(candidates, key=cv2.contourArea)
            if cv2.contourArea(largest_c) < max_area:
                hull = cv2.convexHull(largest_c)
                cv2.drawContours(refined_mask, [hull], -1, 255, thickness=-1)
            else:
                cv2.drawContours(refined_mask, [largest_c], -1, 255, thickness=-1)
        else:
            cv2.drawContours(refined_mask, contours, -1, 255, thickness=-1)

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_smooth)

    inv_mask = cv2.bitwise_not(refined_mask)
    contours_holes, _ = cv2.findContours(inv_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_holes:
        area = cv2.contourArea(cnt)
        if area < (image.size * 0.10):
            cv2.drawContours(refined_mask, [cnt], -1, 255, thickness=-1)

    return refined_mask


def whiten_background(image, mask):
    if len(image.shape) != 2 or len(mask.shape) != 2:
        raise ValueError("Image and mask must be single-channel grayscale.")
    if image.shape != mask.shape:
        raise ValueError("Image and mask dimensions must match.")

    output_image = image.copy()
    output_image[(mask > 0) & (output_image >= 254)] = 253
    output_image[mask == 0] = 255
    return output_image


def process_segmentation_and_whitening(image, block_size=17, c_offset=8, morph_ksize=9):
    mask = segment_fingerprint_region(image, block_size=block_size, c_offset=c_offset, morph_ksize=morph_ksize)
    whitened = whiten_background(image, mask)
    return whitened, mask
