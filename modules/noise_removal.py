import cv2
import numpy as np


def apply_median_filter(image, kernel_size=3):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd integer >= 3.")

    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    filtered = cv2.medianBlur(padded_image, kernel_size)
    result = filtered[pad_width:-pad_width, pad_width:-pad_width]
    return np.ascontiguousarray(result, dtype=np.uint8)


def apply_gaussian_smoothing(image, kernel_size=5, sigma=1.2):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd integer >= 3.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def create_ideal_lowpass_filter(shape, cutoff_frequency):
    rows, cols = shape
    center_y, center_x = rows // 2, cols // 2
    y_indices, x_indices = np.ogrid[:rows, :cols]
    distances = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    filter_mask = np.zeros((rows, cols), dtype=np.float32)
    filter_mask[distances <= cutoff_frequency] = 1.0
    return filter_mask


def create_gaussian_lowpass_filter(shape, cutoff_frequency):
    rows, cols = shape
    center_y, center_x = rows // 2, cols // 2
    y_indices, x_indices = np.ogrid[:rows, :cols]
    distances_squared = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
    d0_squared = 2.0 * (float(cutoff_frequency) ** 2)
    return np.exp(-distances_squared / d0_squared).astype(np.float32)


def create_butterworth_lowpass_filter(shape, cutoff_frequency, order=2):
    rows, cols = shape
    center_y, center_x = rows // 2, cols // 2
    y_indices, x_indices = np.ogrid[:rows, :cols]
    distances = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 1.0 + (distances / float(cutoff_frequency)) ** (2 * order)
    return (1.0 / denom).astype(np.float32)


def apply_frequency_lowpass_filter(image, filter_type='butterworth', cutoff_frequency=50, order=2):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    float_img = np.float32(image)
    dft = np.fft.fft2(float_img)
    dft_shift = np.fft.fftshift(dft)

    if filter_type.lower() == 'ideal':
        mask = create_ideal_lowpass_filter(image.shape, cutoff_frequency)
    elif filter_type.lower() == 'gaussian':
        mask = create_gaussian_lowpass_filter(image.shape, cutoff_frequency)
    elif filter_type.lower() == 'butterworth':
        mask = create_butterworth_lowpass_filter(image.shape, cutoff_frequency, order)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    filtered_shift = dft_shift * mask
    inverse_shift = np.fft.ifftshift(filtered_shift)
    spatial_filtered = np.fft.ifft2(inverse_shift)
    magnitude = np.abs(spatial_filtered)
    clipped = np.clip(magnitude, 0, 255)
    return clipped.astype(np.uint8)


def apply_notch_filter(image, notch_points=None, radius=12, order=2):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    rows, cols = image.shape
    center_y, center_x = rows // 2, cols // 2

    float_img = np.float32(image)
    dft = np.fft.fft2(float_img)
    dft_shift = np.fft.fftshift(dft)

    if notch_points is None or len(notch_points) == 0:
        magnitude_spectrum = np.log(np.abs(dft_shift) + 1.0)
        smoothed = cv2.GaussianBlur(magnitude_spectrum, (21, 21), 0)
        diff = magnitude_spectrum - smoothed

        y_indices, x_indices = np.ogrid[:rows, :cols]
        center_dist = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        diff[center_dist < 15] = 0
        diff[center_dist > min(rows, cols) * 0.48] = 0

        std_d = np.std(diff)
        if std_d > 1e-6:
            candidates = np.where(diff / std_d > 5.5)
            notch_points = list(zip(candidates[0], candidates[1]))
        else:
            notch_points = []

    y_indices, x_indices = np.ogrid[:rows, :cols]
    combined_notch = np.ones((rows, cols), dtype=np.float32)

    for (uk, vk, *_) in notch_points:
        d1 = np.sqrt((y_indices - uk) ** 2 + (x_indices - vk) ** 2)
        sym_uk = 2 * center_y - uk
        sym_vk = 2 * center_x - vk
        d2 = np.sqrt((y_indices - sym_uk) ** 2 + (x_indices - sym_vk) ** 2)

        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = 1.0 / (1.0 + (float(radius) / (d1 + 1e-6)) ** (2 * order))
            term2 = 1.0 / (1.0 + (float(radius) / (d2 + 1e-6)) ** (2 * order))
        pair_filter = term1 * term2
        combined_notch *= pair_filter

    filtered_shift = dft_shift * combined_notch
    inverse_shift = np.fft.ifftshift(filtered_shift)
    spatial_filtered = np.fft.ifft2(inverse_shift)
    magnitude = np.abs(spatial_filtered)
    clipped = np.clip(magnitude, 0, 255)
    return clipped.astype(np.uint8)


def apply_homomorphic_filter(image, cutoff=30, gamma_low=0.4, gamma_high=1.6, order=2):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    float_img = np.float32(image) + 1.0
    log_transformed = np.log(float_img)

    dft = np.fft.fft2(log_transformed)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    center_y, center_x = rows // 2, cols // 2
    y_indices, x_indices = np.ogrid[:rows, :cols]
    distances_squared = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
    d0_squared = float(cutoff) ** 2

    high_emphasis_filter = (gamma_high - gamma_low) * (1.0 - np.exp(-distances_squared / (2.0 * d0_squared))) + gamma_low

    filtered_shift = dft_shift * high_emphasis_filter
    inverse_shift = np.fft.ifftshift(filtered_shift)
    spatial_log = np.real(np.fft.ifft2(inverse_shift))

    restored = np.exp(spatial_log) - 1.0
    normalized = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(normalized, 0, 255).astype(np.uint8)

