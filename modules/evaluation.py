import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(reference_image, target_image):
    if reference_image is None or target_image is None:
        raise ValueError("Reference and target images must not be None.")
    if reference_image.shape != target_image.shape:
        raise ValueError(f"Shape mismatch: reference {reference_image.shape} vs target {target_image.shape}")
    return float(peak_signal_noise_ratio(reference_image, target_image, data_range=255))


def calculate_ssim(reference_image, target_image):
    if reference_image is None or target_image is None:
        raise ValueError("Reference and target images must not be None.")
    if reference_image.shape != target_image.shape:
        raise ValueError(f"Shape mismatch: reference {reference_image.shape} vs target {target_image.shape}")
    return float(structural_similarity(reference_image, target_image, data_range=255))


def evaluate_pair(clean_reference, noisy_input, processed_output):
    results = {}
    if clean_reference is not None:
        results["noisy_psnr"] = calculate_psnr(clean_reference, noisy_input)
        results["noisy_ssim"] = calculate_ssim(clean_reference, noisy_input)
        results["processed_psnr"] = calculate_psnr(clean_reference, processed_output)
        results["processed_ssim"] = calculate_ssim(clean_reference, processed_output)
        results["psnr_improvement"] = results["processed_psnr"] - results["noisy_psnr"]
        results["ssim_improvement"] = results["processed_ssim"] - results["noisy_ssim"]
        results["has_reference"] = True
    else:
        results["noisy_psnr"] = None
        results["noisy_ssim"] = None
        results["processed_psnr"] = None
        results["processed_ssim"] = None
        results["psnr_improvement"] = None
        results["ssim_improvement"] = None
        results["has_reference"] = False

    results["input_output_psnr"] = calculate_psnr(noisy_input, processed_output)
    results["input_output_ssim"] = calculate_ssim(noisy_input, processed_output)
    return results


def save_comparison_plot(original_image, processed_image, output_path, metrics=None, title="TouchScan Comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Input Fingerprint")
    axes[0].axis('off')

    axes[1].imshow(processed_image, cmap='gray')
    axes[1].set_title("Processed & Whitened")
    axes[1].axis('off')

    if metrics and metrics.get("has_reference", False):
        info_text = (
            f"Reference Evaluation:\n"
            f"Input PSNR: {metrics['noisy_psnr']:.2f} dB | Output PSNR: {metrics['processed_psnr']:.2f} dB\n"
            f"Input SSIM: {metrics['noisy_ssim']:.4f} | Output SSIM: {metrics['processed_ssim']:.4f}"
        )
        fig.suptitle(f"{title}\n{info_text}", fontsize=11)
    else:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

