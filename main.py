import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from modules.image_input import load_image, save_image
from modules.noise_detection import analyze_all_noise
from modules.dataset_manager import (
    run_pipeline_on_image,
    create_controlled_dataset,
    evaluate_dataset_batch
)
from modules.watermark import embed_watermark, extract_watermark, get_available_capacity
from modules.evaluation import evaluate_pair, save_comparison_plot


def parse_arguments():
    parser = argparse.ArgumentParser(description="TouchScan: Fingerprint Cleaning & Watermarking Tool")
    parser.add_argument("--input", type=str, default=None, help="Path to input fingerprint image")
    parser.add_argument("--output", type=str, default="outputs/processed/result.png", help="Path to save processed image")
    parser.add_argument("--reference", type=str, default=None, help="Path to clean reference image for PSNR/SSIM")
    parser.add_argument("--watermark", type=str, default=None, help="Text string to embed as invisible watermark")
    parser.add_argument("--extract-watermark", action="store_true", help="Extract watermark from the specified input image")
    parser.add_argument("--generate-noisy-dataset", action="store_true", help="Generate controlled noisy copies from data/original")
    parser.add_argument("--evaluate-dataset", action="store_true", help="Run batch evaluation over dataset and generate CSV & plots")
    parser.add_argument("--dataset-dir", type=str, default="data", help="Root directory containing original and noisy folders")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per category for dataset operations")
    parser.add_argument("--no-display", action="store_true", help="Do not open interactive Matplotlib visual window")
    return parser.parse_args()


def display_visual_summary(original, whitened, metrics=None, primary_noise=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f"Input Fingerprint\n(Detected Noise: {primary_noise})")
    axes[0].axis('off')

    axes[1].imshow(whitened, cmap='gray')
    axes[1].set_title("TouchScan Output\n(Cleaned & Whitened Background)")
    axes[1].axis('off')

    if metrics and metrics.get("has_reference", False):
        subtitle = (
            f"Reference Evaluation: "
            f"Input PSNR: {metrics['noisy_psnr']:.2f} dB -> Output PSNR: {metrics['processed_psnr']:.2f} dB | "
            f"Input SSIM: {metrics['noisy_ssim']:.4f} -> Output SSIM: {metrics['processed_ssim']:.4f}"
        )
        fig.suptitle(subtitle, fontsize=10)
    else:
        fig.suptitle("TouchScan Cleaned Fingerprint Comparison", fontsize=12)

    plt.tight_layout()
    plt.show()


def main():
    args = parse_arguments()

    if args.generate_noisy_dataset:
        orig_dir = Path(args.dataset_dir) / "original"
        noisy_dir = Path(args.dataset_dir) / "noisy"
        print(f"Generating controlled noisy dataset from {orig_dir} into {noisy_dir}...")
        gen_map = create_controlled_dataset(orig_dir, noisy_dir, max_samples=args.samples)
        for cat, files in gen_map.items():
            print(f"  - Category '{cat}': {len(files)} degraded images generated.")
        print("Controlled dataset generation complete.")
        return 0

    if args.evaluate_dataset:
        print(f"Starting batch evaluation on dataset at: {args.dataset_dir}")
        eval_summary = evaluate_dataset_batch(args.dataset_dir, "outputs")
        print("Dataset Evaluation Summary:")
        print(f"  Total tasks processed: {eval_summary['total']}")
        print(f"  Successful: {eval_summary['success']}")
        print(f"  Failed: {eval_summary['failed']}")
        print(f"  Evaluation CSV saved at: {eval_summary['csv_path']}")
        print("  Comparison images saved in: outputs/comparisons/")
        print("  Summary plots saved in: outputs/evaluation/")
        return 0

    if args.extract_watermark:
        if not args.input:
            print("Error: --input image path is required to extract a watermark.", file=sys.stderr)
            return 1
        _, gray_img = load_image(args.input)
        try:
            extracted_text = extract_watermark(gray_img)
            print("=" * 50)
            print("TOUCHSCAN WATERMARK EXTRACTION SUCCESSFUL")
            print("=" * 50)
            print(f"Source Image   : {args.input}")
            print(f"Extracted Text : {extracted_text}")
            print("=" * 50)
            return 0
        except Exception as exc:
            print(f"Failed to extract watermark: {exc}", file=sys.stderr)
            return 1

    if not args.input:
        print("Error: Please provide --input <image_path> or choose --evaluate-dataset / --generate-noisy-dataset.")
        print("Usage: python main.py --input <path_to_image> [--watermark <text>] [--reference <clean_ref>]")
        return 1

    original_raw, gray_img = load_image(args.input)
    print(f"Loaded image '{args.input}' with dimensions {gray_img.shape[1]}x{gray_img.shape[0]}.")

    pipeline_output = run_pipeline_on_image(gray_img)
    detection = pipeline_output["detection"]
    whitened = pipeline_output["whitened"]
    primary_noise = pipeline_output["selected_noise"]

    print("=" * 50)
    print("NOISE IDENTIFICATION DIAGNOSTICS")
    print("=" * 50)
    print(f"Primary Identified Noise : {primary_noise}")
    print(f"Detected Categories      : {detection['detected_categories']}")
    print(f"Salt-and-Pepper Ratio    : {detection['salt_pepper']['ratio']:.4f} (Detected: {detection['salt_pepper']['detected']})")
    print(f"Gaussian Estimated Sigma : {detection['gaussian']['estimated_sigma']:.2f} (Detected: {detection['gaussian']['detected']})")
    print(f"Periodic Symmetric Peaks : {detection['periodic']['symmetric_peaks']} (Detected: {detection['periodic']['detected']})")
    print(f"Illumination Coeff-Var   : {detection['uneven_illumination']['coefficient_of_variation']:.3f} (Detected: {detection['uneven_illumination']['detected']})")
    print("=" * 50)

    clean_ref = None
    metrics = None
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            _, clean_ref = load_image(ref_path)
            metrics = evaluate_pair(clean_ref, gray_img, whitened)
            print("QUALITY METRICS (GROUND TRUTH COMPARISON)")
            print(f"  Input PSNR      : {metrics['noisy_psnr']:.2f} dB")
            print(f"  Processed PSNR  : {metrics['processed_psnr']:.2f} dB (Change: {metrics['psnr_improvement']:+.2f} dB)")
            print(f"  Input SSIM      : {metrics['noisy_ssim']:.4f}")
            print(f"  Processed SSIM  : {metrics['processed_ssim']:.4f} (Change: {metrics['ssim_improvement']:+.4f})")
            print("=" * 50)
        else:
            print(f"Warning: Reference file {args.reference} not found. Skipping reference evaluation.")

    final_image = whitened
    if args.watermark:
        print(f"Embedding invisible watermark: '{args.watermark}'...")
        eligible_pixels, available_chars = get_available_capacity(final_image)
        print(f"Available white pixels for watermarking: {eligible_pixels} (~{available_chars} characters max).")
        watermarked_img, embedded_bits = embed_watermark(final_image, args.watermark)
        final_image = watermarked_img
        print(f"Watermark embedded successfully ({embedded_bits} bits modified in white background).")

        extracted_verify = extract_watermark(final_image)
        if extracted_verify == args.watermark:
            print("Watermark verification: EXACT MATCH verified through extraction.")
        else:
            print("Watermark verification warning: Extracted text does not match original.", file=sys.stderr)

    saved_path = save_image(args.output, final_image)
    print(f"Saved processed output image to: {saved_path}")

    comp_path = Path("outputs/comparisons") / f"comparison_{Path(args.input).stem}.png"
    save_comparison_plot(gray_img, final_image, comp_path, metrics=metrics, title=f"TouchScan: {Path(args.input).name}")
    print(f"Saved visual comparison plot to: {comp_path}")

    if not args.no_display:
        display_visual_summary(gray_img, final_image, metrics=metrics, primary_noise=primary_noise)

    return 0


if __name__ == "__main__":
    sys.exit(main())

