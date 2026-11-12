import cv2
import numpy as np


def detect_salt_and_pepper_noise(image, threshold_fraction=0.015, extreme_margin=3):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    total_pixels = image.size
    median_filtered = cv2.medianBlur(image, 3)
    pixel_diff = cv2.absdiff(image, median_filtered)

    is_extreme = (image <= extreme_margin) | (image >= (255 - extreme_margin))
    is_outlier = (pixel_diff > 35) & is_extreme

    salt_pepper_count = np.count_nonzero(is_outlier)
    salt_pepper_ratio = salt_pepper_count / float(total_pixels)

    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    spike_0 = hist[0:extreme_margin + 1].sum() / float(total_pixels)
    spike_255 = hist[255 - extreme_margin:256].sum() / float(total_pixels)
    spike_ratio = spike_0 + spike_255

    is_detected = (salt_pepper_ratio >= threshold_fraction) or (salt_pepper_ratio >= (threshold_fraction * 0.6) and spike_ratio > 0.05)
    confidence = min(1.0, salt_pepper_ratio / (threshold_fraction * 2.0))

    return {
        "detected": bool(is_detected),
        "ratio": float(salt_pepper_ratio),
        "spike_ratio": float(spike_ratio),
        "confidence": float(confidence)
    }


def detect_gaussian_noise(image, threshold_sigma=8.0):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    median_abs_dev = np.median(np.abs(laplacian))
    sigma_mad = median_abs_dev * 1.4826 / np.sqrt(6.0)

    rows, cols = image.shape
    block_size = 16
    local_stds = []

    for r in range(0, rows - block_size, block_size):
        for c in range(0, cols - block_size, block_size):
            patch = image[r:r + block_size, c:c + block_size]
            std = np.std(patch)
            local_stds.append(std)

    if local_stds:
        sorted_stds = np.sort(local_stds)
        lowest_quartile = sorted_stds[:max(1, len(sorted_stds) // 4)]
        flat_region_std = float(np.mean(lowest_quartile))
    else:
        flat_region_std = float(sigma_mad)

    estimated_noise_sigma = 0.5 * (sigma_mad + flat_region_std)
    is_detected = estimated_noise_sigma >= threshold_sigma
    confidence = min(1.0, max(0.0, (estimated_noise_sigma - threshold_sigma * 0.75) / (threshold_sigma * 1.5)))

    return {
        "detected": bool(is_detected),
        "estimated_sigma": float(estimated_noise_sigma),
        "mad_sigma": float(sigma_mad),
        "flat_region_sigma": float(flat_region_std),
        "confidence": float(confidence)
    }


def detect_periodic_noise(image, peak_threshold_ratio=4.0):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    float_image = np.float32(image)
    dft = np.fft.fft2(float_image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(np.abs(dft_shift) + 1.0)

    rows, cols = image.shape
    center_y, center_x = rows // 2, cols // 2

    y_indices, x_indices = np.ogrid[:rows, :cols]
    distances_from_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    low_freq_mask = distances_from_center < 8
    outer_boundary_mask = distances_from_center > (min(rows, cols) * 0.48)

    smoothed_spectrum = cv2.GaussianBlur(magnitude_spectrum, (21, 21), 0)
    spectral_diff = magnitude_spectrum - smoothed_spectrum
    spectral_diff[low_freq_mask] = 0
    spectral_diff[outer_boundary_mask] = 0

    std_diff = np.std(spectral_diff)
    if std_diff <= 1e-6:
        std_diff = 1e-6

    z_scores = spectral_diff / std_diff
    peak_points = np.where(z_scores > peak_threshold_ratio)
    num_peaks = len(peak_points[0])

    symmetric_peak_count = 0
    detected_peaks = []

    for py, px in zip(peak_points[0], peak_points[1]):
        sym_y = 2 * center_y - py
        sym_x = 2 * center_x - px
        if 0 <= sym_y < rows and 0 <= sym_x < cols:
            if z_scores[sym_y, sym_x] > (peak_threshold_ratio * 0.7):
                symmetric_peak_count += 1
                detected_peaks.append((int(py), int(px), float(magnitude_spectrum[py, px])))

    is_detected = (symmetric_peak_count >= 2)
    confidence = min(1.0, symmetric_peak_count / 4.0)

    return {
        "detected": bool(is_detected),
        "symmetric_peaks": int(symmetric_peak_count),
        "total_candidate_peaks": int(num_peaks),
        "peaks": detected_peaks,
        "confidence": float(confidence)
    }


def detect_uneven_illumination(image, cv_threshold=0.22, range_threshold=80.0):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    blurred = cv2.GaussianBlur(image, (61, 61), 0)
    mean_val = float(np.mean(blurred))
    std_val = float(np.std(blurred))
    coefficient_of_variation = (std_val / (mean_val + 1e-6))

    rows, cols = image.shape
    corner_size_r = max(10, rows // 6)
    corner_size_c = max(10, cols // 6)

    tl = float(np.mean(blurred[0:corner_size_r, 0:corner_size_c]))
    tr = float(np.mean(blurred[0:corner_size_r, cols - corner_size_c:cols]))
    bl = float(np.mean(blurred[rows - corner_size_r:rows, 0:corner_size_c]))
    br = float(np.mean(blurred[rows - corner_size_r:rows, cols - corner_size_c:cols]))
    center = float(np.mean(blurred[rows // 3:(2 * rows) // 3, cols // 3:(2 * cols) // 3]))

    corner_and_center_values = [tl, tr, bl, br, center]
    illumination_range = float(max(corner_and_center_values) - min(corner_and_center_values))

    is_detected = (coefficient_of_variation >= cv_threshold) or (illumination_range >= range_threshold)
    confidence = min(1.0, max(coefficient_of_variation / (cv_threshold * 1.5), illumination_range / (range_threshold * 1.5)))

    return {
        "detected": bool(is_detected),
        "coefficient_of_variation": float(coefficient_of_variation),
        "illumination_range": float(illumination_range),
        "confidence": float(confidence)
    }


def analyze_all_noise(image, salt_pepper_th=0.015, gaussian_th=8.0, periodic_th=4.0, uneven_th=0.22):
    sp_res = detect_salt_and_pepper_noise(image, threshold_fraction=salt_pepper_th)
    gauss_res = detect_gaussian_noise(image, threshold_sigma=gaussian_th)
    period_res = detect_periodic_noise(image, peak_threshold_ratio=periodic_th)
    uneven_res = detect_uneven_illumination(image, cv_threshold=uneven_th)

    detected_categories = []
    if sp_res["detected"]:
        detected_categories.append("salt_pepper")
    if gauss_res["detected"]:
        detected_categories.append("gaussian")
    if period_res["detected"]:
        detected_categories.append("periodic")
    if uneven_res["detected"]:
        detected_categories.append("uneven_illumination")

    if not detected_categories:
        primary_noise = "clean_or_low_noise"
    else:
        confidences = {
            "salt_pepper": sp_res["confidence"],
            "gaussian": gauss_res["confidence"],
            "periodic": period_res["confidence"],
            "uneven_illumination": uneven_res["confidence"]
        }
        primary_noise = max(detected_categories, key=lambda c: confidences[c])

    return {
        "primary_noise": primary_noise,
        "detected_categories": detected_categories,
        "salt_pepper": sp_res,
        "gaussian": gauss_res,
        "periodic": period_res,
        "uneven_illumination": uneven_res
    }
