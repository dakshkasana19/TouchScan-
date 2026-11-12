import unittest
from pathlib import Path
import cv2
import numpy as np

from modules.image_input import load_image, save_image
from modules.noise_detection import (
    detect_salt_and_pepper_noise,
    detect_gaussian_noise,
    detect_periodic_noise,
    detect_uneven_illumination,
    analyze_all_noise
)
from modules.noise_removal import (
    apply_median_filter,
    apply_gaussian_smoothing,
    apply_frequency_lowpass_filter,
    apply_notch_filter,
    apply_homomorphic_filter
)
from modules.image_enhancement import (
    apply_histogram_equalization,
    apply_power_law_transformation,
    apply_local_gamma_correction,
    apply_laplacian_sharpening,
    enhance_fingerprint
)
from modules.segmentation import (
    segment_fingerprint_region,
    whiten_background,
    process_segmentation_and_whitening
)
from modules.watermark import (
    string_to_bits,
    bits_to_string,
    embed_watermark,
    extract_watermark,
    get_available_capacity
)
from modules.evaluation import calculate_psnr, calculate_ssim, evaluate_pair
from modules.dataset_manager import (
    generate_gaussian_noise,
    generate_salt_and_pepper_noise,
    generate_periodic_noise,
    generate_uneven_illumination,
    run_pipeline_on_image
)


class TestTouchScanPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        sample_path = Path("data/original/101_1.tif")
        if sample_path.exists():
            _, self.clean_reference = load_image(sample_path)
        else:
            rows, cols = 160, 160
            y_indices, x_indices = np.ogrid[:rows, :cols]
            center_y, center_x = rows // 2, cols // 2
            dist_from_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
            fingerprint_pattern = 128 + 60 * np.sin(dist_from_center / 3.0)
            circle_mask = dist_from_center <= 55
            synthetic = np.full((rows, cols), 245, dtype=np.float32)
            synthetic[circle_mask] = fingerprint_pattern[circle_mask]
            self.clean_reference = np.clip(synthetic, 0, 255).astype(np.uint8)

    def test_image_loading_and_invalid_handling(self):
        with self.assertRaises(FileNotFoundError):
            load_image("non_existent_image_path_12345.png")

        test_img_path = "outputs/test_temp.png"
        save_image(test_img_path, self.clean_reference)
        orig, gray = load_image(test_img_path)
        self.assertEqual(gray.shape, self.clean_reference.shape)
        self.assertEqual(gray.dtype, np.uint8)

    def test_noise_detection_salt_and_pepper(self):
        sp_noisy = generate_salt_and_pepper_noise(self.clean_reference, density=0.08)
        det = detect_salt_and_pepper_noise(sp_noisy)
        self.assertTrue(det["detected"])
        self.assertGreater(det["ratio"], 0.02)

    def test_noise_detection_gaussian(self):
        gauss_noisy = generate_gaussian_noise(self.clean_reference, sigma=30.0)
        det = detect_gaussian_noise(gauss_noisy)
        self.assertTrue(det["detected"])
        self.assertGreater(det["estimated_sigma"], 10.0)

    def test_noise_detection_periodic(self):
        periodic_noisy = generate_periodic_noise(self.clean_reference, amplitude=45.0)
        det = detect_periodic_noise(periodic_noisy)
        self.assertTrue(det["detected"])
        self.assertGreaterEqual(det["symmetric_peaks"], 2)

    def test_noise_detection_uneven_illumination(self):
        uneven_img = generate_uneven_illumination(self.clean_reference, intensity_drop=0.85)
        det = detect_uneven_illumination(uneven_img)
        self.assertTrue(det["detected"])

    def test_median_filter(self):
        sp_noisy = generate_salt_and_pepper_noise(self.clean_reference, density=0.05)
        filtered = apply_median_filter(sp_noisy, kernel_size=3)
        self.assertEqual(filtered.shape, sp_noisy.shape)
        psnr_noisy = calculate_psnr(self.clean_reference, sp_noisy)
        psnr_filtered = calculate_psnr(self.clean_reference, filtered)
        self.assertGreater(psnr_filtered, psnr_noisy)

    def test_gaussian_and_frequency_filters(self):
        gauss_noisy = generate_gaussian_noise(self.clean_reference, sigma=20.0)
        filtered_spatial = apply_gaussian_smoothing(gauss_noisy, kernel_size=5, sigma=1.2)
        self.assertEqual(filtered_spatial.shape, gauss_noisy.shape)

        filtered_butterworth = apply_frequency_lowpass_filter(gauss_noisy, filter_type='butterworth', cutoff_frequency=40)
        self.assertEqual(filtered_butterworth.shape, gauss_noisy.shape)

        filtered_gaussian_freq = apply_frequency_lowpass_filter(gauss_noisy, filter_type='gaussian', cutoff_frequency=40)
        self.assertEqual(filtered_gaussian_freq.shape, gauss_noisy.shape)

        filtered_ideal = apply_frequency_lowpass_filter(gauss_noisy, filter_type='ideal', cutoff_frequency=40)
        self.assertEqual(filtered_ideal.shape, gauss_noisy.shape)

    def test_notch_filter(self):
        periodic_noisy = generate_periodic_noise(self.clean_reference, amplitude=40.0)
        det = detect_periodic_noise(periodic_noisy)
        notch_cleaned = apply_notch_filter(periodic_noisy, notch_points=det["peaks"], radius=8, order=2)
        self.assertEqual(notch_cleaned.shape, periodic_noisy.shape)
        psnr_noisy = calculate_psnr(self.clean_reference, periodic_noisy)
        psnr_cleaned = calculate_psnr(self.clean_reference, notch_cleaned)
        self.assertGreater(psnr_cleaned, psnr_noisy)

    def test_homomorphic_filter(self):
        uneven_img = generate_uneven_illumination(self.clean_reference, intensity_drop=0.7)
        homo_filtered = apply_homomorphic_filter(uneven_img, cutoff=30)
        self.assertEqual(homo_filtered.shape, uneven_img.shape)

    def test_image_enhancement(self):
        eq_img = apply_histogram_equalization(self.clean_reference)
        self.assertEqual(eq_img.shape, self.clean_reference.shape)

        gamma_img = apply_power_law_transformation(self.clean_reference, gamma=1.2)
        self.assertEqual(gamma_img.shape, self.clean_reference.shape)

        clahe_img = apply_local_gamma_correction(self.clean_reference)
        self.assertEqual(clahe_img.shape, self.clean_reference.shape)

        sharp_img = apply_laplacian_sharpening(self.clean_reference, strength=0.4)
        self.assertEqual(sharp_img.shape, self.clean_reference.shape)

        enhanced, intermediates = enhance_fingerprint(self.clean_reference)
        self.assertIn("final_enhanced", intermediates)
        self.assertEqual(enhanced.shape, self.clean_reference.shape)

    def test_segmentation_and_whitening(self):
        mask = segment_fingerprint_region(self.clean_reference)
        self.assertEqual(mask.shape, self.clean_reference.shape)
        whitened = whiten_background(self.clean_reference, mask)
        self.assertEqual(whitened.shape, self.clean_reference.shape)
        self.assertEqual(whitened[0, 0], 255)

    def test_watermark_embedding_and_extraction(self):
        test_text = "BU-CSET344-S24CSEU2714"
        canvas = np.full((100, 100), 255, dtype=np.uint8)
        canvas[40:60, 40:60] = 50

        capacity, chars = get_available_capacity(canvas)
        self.assertGreater(chars, len(test_text))

        watermarked, num_bits = embed_watermark(canvas, test_text)
        self.assertEqual(watermarked.shape, canvas.shape)
        self.assertIn(254, np.unique(watermarked))

        extracted_text = extract_watermark(watermarked)
        self.assertEqual(extracted_text, test_text)

    def test_watermark_capacity_exceeded(self):
        tiny_canvas = np.zeros((10, 10), dtype=np.uint8)
        tiny_canvas[0, 0] = 255
        with self.assertRaises(ValueError):
            embed_watermark(tiny_canvas, "Payload too long for single pixel")

    def test_quality_metrics(self):
        identical_psnr = calculate_psnr(self.clean_reference, self.clean_reference)
        self.assertGreater(identical_psnr, 80.0)
        identical_ssim = calculate_ssim(self.clean_reference, self.clean_reference)
        self.assertAlmostEqual(identical_ssim, 1.0, places=4)

        metrics = evaluate_pair(self.clean_reference, self.clean_reference, self.clean_reference)
        self.assertTrue(metrics["has_reference"])
        self.assertAlmostEqual(metrics["input_output_ssim"], 1.0, places=4)

    def test_full_pipeline_execution(self):
        sp_noisy = generate_salt_and_pepper_noise(self.clean_reference, density=0.04)
        result = run_pipeline_on_image(sp_noisy)
        self.assertIn("whitened", result)
        self.assertIn("detection", result)
        self.assertEqual(result["whitened"].shape, self.clean_reference.shape)


if __name__ == "__main__":
    unittest.main()
