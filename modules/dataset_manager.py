import os
import csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.image_input import load_image, save_image
from modules.noise_detection import analyze_all_noise
from modules.noise_removal import (
    apply_median_filter,
    apply_gaussian_smoothing,
    apply_notch_filter,
    apply_homomorphic_filter,
    apply_frequency_lowpass_filter
)
from modules.image_enhancement import enhance_fingerprint
from modules.segmentation import process_segmentation_and_whitening
from modules.evaluation import calculate_psnr, calculate_ssim, evaluate_pair, save_comparison_plot


def generate_gaussian_noise(image, mean=0.0, sigma=25.0):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_float = image.astype(np.float32) + noise
    return np.clip(noisy_float, 0, 255).astype(np.uint8)


def generate_salt_and_pepper_noise(image, density=0.04):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    noisy = image.copy()
    num_noise = int(density * image.size)
    half_noise = num_noise // 2

    coords_y = np.random.randint(0, image.shape[0], half_noise)
    coords_x = np.random.randint(0, image.shape[1], half_noise)
    noisy[coords_y, coords_x] = 255

    coords_y = np.random.randint(0, image.shape[0], half_noise)
    coords_x = np.random.randint(0, image.shape[1], half_noise)
    noisy[coords_y, coords_x] = 0

    return noisy


def generate_periodic_noise(image, frequency_u=0.08, frequency_v=0.08, amplitude=35.0):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    rows, cols = image.shape
    y_coords, x_coords = np.ogrid[:rows, :cols]
    interference = amplitude * np.sin(2.0 * np.pi * (frequency_u * x_coords + frequency_v * y_coords))
    noisy_float = image.astype(np.float32) + interference
    return np.clip(noisy_float, 0, 255).astype(np.uint8)


def generate_uneven_illumination(image, intensity_drop=0.75):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")
    rows, cols = image.shape
    y_coords, x_coords = np.ogrid[:rows, :cols]
    gradient = 1.0 - (intensity_drop * (x_coords / float(cols) * 0.6 + y_coords / float(rows) * 0.4))
    noisy_float = image.astype(np.float32) * gradient
    return np.clip(noisy_float, 0, 255).astype(np.uint8)


def create_controlled_dataset(original_dir, noisy_base_dir, max_samples=20, random_seed=42):
    np.random.seed(random_seed)
    orig_path = Path(original_dir).resolve()
    base_noisy = Path(noisy_base_dir).resolve()

    categories = ["gaussian", "salt_pepper", "periodic", "uneven_illumination"]
    for cat in categories:
        (base_noisy / cat).mkdir(parents=True, exist_ok=True)

    valid_extensions = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
    image_files = sorted([f for f in orig_path.iterdir() if f.suffix.lower() in valid_extensions and f.stem.endswith('_1') or not f.stem.endswith('_1')][:max_samples])

    generated_map = {cat: [] for cat in categories}

    for img_path in image_files:
        _, gray = load_image(img_path)
        base_name = img_path.stem + ".png"

        gauss_img = generate_gaussian_noise(gray, sigma=25.0)
        gauss_dest = base_noisy / "gaussian" / base_name
        save_image(gauss_dest, gauss_img)
        generated_map["gaussian"].append(str(gauss_dest))

        sp_img = generate_salt_and_pepper_noise(gray, density=0.04)
        sp_dest = base_noisy / "salt_pepper" / base_name
        save_image(sp_dest, sp_img)
        generated_map["salt_pepper"].append(str(sp_dest))

        per_img = generate_periodic_noise(gray, amplitude=35.0)
        per_dest = base_noisy / "periodic" / base_name
        save_image(per_dest, per_img)
        generated_map["periodic"].append(str(per_dest))

        uneven_img = generate_uneven_illumination(gray, intensity_drop=0.65)
        uneven_dest = base_noisy / "uneven_illumination" / base_name
        save_image(uneven_dest, uneven_img)
        generated_map["uneven_illumination"].append(str(uneven_dest))

    return generated_map


def run_pipeline_on_image(image, force_noise_type=None):
    if len(image.shape) != 2:
        raise ValueError("Image must be single-channel grayscale.")

    detection = analyze_all_noise(image)
    noise_type = force_noise_type if force_noise_type is not None else detection["primary_noise"]

    filtered = image.copy()
    if noise_type == "salt_pepper":
        filtered = apply_median_filter(filtered, kernel_size=3)
        if detection["salt_pepper"]["ratio"] > 0.05:
            filtered = apply_median_filter(filtered, kernel_size=3)
    elif noise_type == "gaussian":
        filtered = apply_gaussian_smoothing(filtered, kernel_size=5, sigma=1.2)
        filtered = apply_frequency_lowpass_filter(filtered, filter_type='butterworth', cutoff_frequency=45, order=2)
    elif noise_type == "periodic":
        notch_pts = [(p[0], p[1]) for p in detection["periodic"]["peaks"]]
        filtered = apply_notch_filter(filtered, notch_points=notch_pts, radius=10, order=2)
    elif noise_type == "uneven_illumination":
        filtered = apply_homomorphic_filter(filtered, cutoff=30, gamma_low=0.4, gamma_high=1.6)
    else:
        filtered = apply_median_filter(filtered, kernel_size=3)

    enhanced, _ = enhance_fingerprint(
        filtered,
        use_hist_eq=False,
        gamma=1.1,
        use_clahe=True,
        laplacian_strength=0.3
    )

    whitened, mask = process_segmentation_and_whitening(enhanced)

    return {
        "detection": detection,
        "selected_noise": noise_type,
        "filtered": filtered,
        "enhanced": enhanced,
        "mask": mask,
        "whitened": whitened
    }


def evaluate_dataset_batch(dataset_root, output_dir, run_noisy_categories=True):
    root_path = Path(dataset_root).resolve()
    out_path = Path(output_dir).resolve()
    processed_dir = out_path / "processed"
    comparisons_dir = out_path / "comparisons"
    eval_dir = out_path / "evaluation"

    processed_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / "evaluation_results.csv"
    records = []

    noisy_dir = root_path / "noisy"
    original_dir = root_path / "original"

    tasks = []
    if run_noisy_categories and noisy_dir.exists():
        categories = ["gaussian", "salt_pepper", "periodic", "uneven_illumination"]
        for cat in categories:
            cat_dir = noisy_dir / cat
            if cat_dir.exists():
                for f in sorted(cat_dir.iterdir()):
                    if f.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".jpg"}:
                        clean_ref_path = original_dir / f.name
                        if not clean_ref_path.exists():
                            clean_ref_path = original_dir / (f.stem + ".tif")
                        tasks.append({
                            "category": cat,
                            "noisy_path": f,
                            "reference_path": clean_ref_path if clean_ref_path.exists() else None
                        })

    if original_dir.exists():
        for f in sorted(original_dir.iterdir()):
            if f.suffix.lower() in {".png", ".tif"}:
                tasks.append({
                    "category": "original_natural",
                    "noisy_path": f,
                    "reference_path": None
                })

    for task in tasks:
        noisy_path = task["noisy_path"]
        ref_path = task["reference_path"]
        category = task["category"]

        try:
            _, noisy_img = load_image(noisy_path)
            clean_img = None
            if ref_path is not None and ref_path.exists():
                _, clean_img = load_image(ref_path)

            pipeline_result = run_pipeline_on_image(noisy_img)
            whitened_img = pipeline_result["whitened"]

            proc_filename = f"{category}_{noisy_path.stem}_cleaned.png"
            proc_out_path = processed_dir / proc_filename
            save_image(proc_out_path, whitened_img)

            pair_metrics = evaluate_pair(clean_img, noisy_img, whitened_img)

            comp_filename = f"{category}_{noisy_path.stem}_comparison.png"
            comp_out_path = comparisons_dir / comp_filename
            save_comparison_plot(
                noisy_img, whitened_img, comp_out_path,
                metrics=pair_metrics,
                title=f"Category: {category} | File: {noisy_path.name}"
            )

            records.append({
                "image_name": noisy_path.name,
                "category": category,
                "status": "SUCCESS",
                "detected_primary": pipeline_result["detection"]["primary_noise"],
                "noisy_psnr": f"{pair_metrics['noisy_psnr']:.2f}" if pair_metrics['noisy_psnr'] is not None else "N/A",
                "noisy_ssim": f"{pair_metrics['noisy_ssim']:.4f}" if pair_metrics['noisy_ssim'] is not None else "N/A",
                "processed_psnr": f"{pair_metrics['processed_psnr']:.2f}" if pair_metrics['processed_psnr'] is not None else "N/A",
                "processed_ssim": f"{pair_metrics['processed_ssim']:.4f}" if pair_metrics['processed_ssim'] is not None else "N/A",
                "psnr_improvement": f"{pair_metrics['psnr_improvement']:.2f}" if pair_metrics['psnr_improvement'] is not None else "N/A",
                "ssim_improvement": f"{pair_metrics['ssim_improvement']:.4f}" if pair_metrics['ssim_improvement'] is not None else "N/A",
                "error": ""
            })

        except Exception as exc:
            records.append({
                "image_name": noisy_path.name,
                "category": category,
                "status": "FAILED",
                "detected_primary": "N/A",
                "noisy_psnr": "N/A",
                "noisy_ssim": "N/A",
                "processed_psnr": "N/A",
                "processed_ssim": "N/A",
                "psnr_improvement": "N/A",
                "ssim_improvement": "N/A",
                "error": str(exc)
            })

    fieldnames = [
        "image_name", "category", "status", "detected_primary",
        "noisy_psnr", "noisy_ssim", "processed_psnr", "processed_ssim",
        "psnr_improvement", "ssim_improvement", "error"
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    generate_summary_plots(records, eval_dir / "evaluation_summary_plot.png")

    return {
        "csv_path": str(csv_path),
        "total": len(tasks),
        "success": len([r for r in records if r["status"] == "SUCCESS"]),
        "failed": len([r for r in records if r["status"] == "FAILED"]),
        "records": records
    }


def generate_summary_plots(records, plot_output_path):
    cat_metrics = {}
    for r in records:
        if r["status"] == "SUCCESS" and r["processed_psnr"] != "N/A":
            cat = r["category"]
            if cat not in cat_metrics:
                cat_metrics[cat] = {"noisy_psnr": [], "proc_psnr": [], "noisy_ssim": [], "proc_ssim": []}
            cat_metrics[cat]["noisy_psnr"].append(float(r["noisy_psnr"]))
            cat_metrics[cat]["proc_psnr"].append(float(r["processed_psnr"]))
            cat_metrics[cat]["noisy_ssim"].append(float(r["noisy_ssim"]))
            cat_metrics[cat]["proc_ssim"].append(float(r["processed_ssim"]))

    if not cat_metrics:
        return

    categories = list(cat_metrics.keys())
    avg_noisy_psnr = [np.mean(cat_metrics[c]["noisy_psnr"]) for c in categories]
    avg_proc_psnr = [np.mean(cat_metrics[c]["proc_psnr"]) for c in categories]
    avg_noisy_ssim = [np.mean(cat_metrics[c]["noisy_ssim"]) for c in categories]
    avg_proc_ssim = [np.mean(cat_metrics[c]["proc_ssim"]) for c in categories]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_indices = np.arange(len(categories))
    bar_width = 0.35

    axes[0].bar(x_indices - bar_width / 2, avg_noisy_psnr, bar_width, label='Noisy Input PSNR', color='#d9534f')
    axes[0].bar(x_indices + bar_width / 2, avg_proc_psnr, bar_width, label='Processed Output PSNR', color='#5cb85c')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Average PSNR Comparison by Degradation Category')
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(categories, rotation=15)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].bar(x_indices - bar_width / 2, avg_noisy_ssim, bar_width, label='Noisy Input SSIM', color='#f0ad4e')
    axes[1].bar(x_indices + bar_width / 2, avg_proc_ssim, bar_width, label='Processed Output SSIM', color='#0275d8')
    axes[1].set_ylabel('SSIM (0-1)')
    axes[1].set_title('Average SSIM Comparison by Degradation Category')
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(categories, rotation=15)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

