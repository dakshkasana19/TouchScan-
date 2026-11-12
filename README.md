# TouchScan — Fingerprint Image Cleaning and Invisible Watermarking Tool

A Python-based digital image processing tool designed to clean degraded paper-to-digital and sensor-captured fingerprint images, whiten non-fingerprint background regions, and provide spatial-domain invisible watermarking using traditional image processing techniques.

Developed as a university mini-project for **Bennett University**, School of Computer Science Engineering and Technology, for the course **Image and Video Processing (CSET344)**.

---

## Academic Information

- **Institution**: Bennett University
- **School**: School of Computer Science Engineering and Technology
- **Course**: Image and Video Processing
- **Course Code**: CSET344
- **Project Title**: TouchScan — Fingerprint Image Cleaning and Invisible Watermarking Tool
- **Project Members**:
  - Daksh Kasana (Enrollment: `S24CSEU2714`)
  - Saivats (Enrollment: `S24CSEU1436`)

---

## 1. Project Overview & Problem Statement

Fingerprint impressions transferred onto physical paper via an ink pad and subsequently photographed using mobile devices or low-cost optical scanners frequently suffer from severe visual artifacts:
- **Salt-and-pepper impulse noise** from paper grain, sensor dust, or dry ink flakes.
- **Gaussian sensor noise** from low-light smartphone camera captures or thermal sensor fluctuations.
- **Periodic sinusoidal interference** from electronic raster interference or screen displays.
- **Uneven illumination gradients** caused by directional ambient lighting or flash reflections.
- **Unwanted paper background clutter** obscuring fingerprint boundaries.

**TouchScan** addresses these issues through a deterministic, non-machine-learning digital image processing pipeline. It diagnoses the degradation type, selectively applies noise removal filters, enhances ridge-valley contrast via adaptive equalisation and Laplacian sharpening, segments the fingerprint foreground, converts background clutter to uniform white (`255`), and optionally embeds an imperceptible text watermark into pure white pixels (`255` -> `254`) without altering ridge details.

---

## 2. Technology Stack

Adhering strictly to traditional computer vision without neural networks or deep learning:

- **Programming Language**: Python 3.10+
- **Numerical Computing**: NumPy
- **Image Processing**: OpenCV (`opencv-python`), scikit-image (`scikit-image`)
- **Metrics & Plotting**: scikit-image metrics (`peak_signal_noise_ratio`, `structural_similarity`), Matplotlib
- **Standard Library**: `pathlib`, `csv`, `argparse`, `unittest`, `sys`, `os`

*Note: No machine learning libraries (TensorFlow, PyTorch, Keras, scikit-learn training) or heavy web frameworks (Flask, Django, Streamlit) are used.*

---

## 3. Project Directory Structure

```
TouchScan/
├── README.md                           # Comprehensive documentation
├── requirements.txt                    # Project dependencies
├── main.py                             # CLI application entry point
│
├── modules/
│   ├── __init__.py                     # Package initializer
│   ├── image_input.py                  # Loading, verification & grayscale conversion
│   ├── noise_detection.py              # Statistical & frequency-domain noise diagnosis
│   ├── noise_removal.py                # Spatial & frequency-domain filters (ILPF, GLPF, BLPF, Notch, Homomorphic)
│   ├── image_enhancement.py            # CLAHE, gamma correction, Laplacian sharpening
│   ├── segmentation.py                 # Adaptive thresholding, morphological closing & background whitening
│   ├── watermark.py                    # Spatial LSB-based background watermarking & extraction
│   ├── evaluation.py                   # PSNR & SSIM metric evaluation and plotting
│   └── dataset_manager.py              # Batch execution, controlled noise generation & CSV reporting
│
├── data/
│   ├── README.md                       # Dataset overview and metadata
│   ├── original/                       # 80 FVC2002 DB1_B TIFF and PNG fingerprint images
│   └── noisy/                          # Controlled degraded evaluation sets
│       ├── gaussian/                   # Controlled Gaussian noise degraded copies
│       ├── salt_pepper/                # Controlled Salt-and-Pepper degraded copies
│       ├── periodic/                   # Controlled Sinusoidal interference copies
│       └── uneven_illumination/        # Controlled spatial gradient illumination copies
│
├── outputs/
│   ├── processed/                      # Cleaned and whitened fingerprint outputs
│   ├── comparisons/                    # Visual before-and-after comparison figures
│   └── evaluation/                     # evaluation_results.csv & evaluation_summary_plot.png
│
└── tests/
    └── test_touchscan.py               # Unit and integration test suite
```

---

## 4. Dataset Inspection & Characteristics

The integrated benchmark dataset attached for evaluation is **FVC2002 DB1_B**:
- **Dataset Source**: Biometric Recognition Group, University of Bologna (Fingerprint Verification Competition 2002).
- **Sensor**: Low-cost optical sensor "TouchView II" by Identix.
- **Image Count**: 80 fingerprint impressions.
- **Subjects**: 10 distinct fingers/subjects (`101` to `110`), with 8 impressions per subject (`101_1` to `110_8`).
- **Dimensions**: 640 × 480 pixels (or 480 × 640 depending on orientation).
- **Color Format**: 8-bit grayscale uncompressed TIFF (`.tif`), accompanied by 80 lossless PNG duplicates for universal viewer compatibility.
- **Resolution**: 500 dpi.
- **Clean Reference Status**: The original impressions represent clean baseline physical impressions with low natural noise, enabling their use as ground truth references for controlled quantitative benchmarking.

---

## 5. Image Processing Pipeline & Algorithms

### Phase 1: Image Input & Validation (`modules/image_input.py`)
- Resolves paths using `pathlib.Path`.
- Decodes image arrays and validates dimensions.
- Converts multichannel inputs (RGB, BGR, BGRA) into single-channel 8-bit grayscale (`np.uint8`).

### Phase 2: Automatic Noise Diagnosis (`modules/noise_detection.py`)
- **Salt-and-Pepper**: Evaluates absolute deviation from a 3×3 median blur alongside histogram spikes near intensity `0` and `255`.
- **Gaussian Noise**: Estimates standard deviation using a 3×3 Laplacian mask with Median Absolute Deviation (MAD), reinforced by low-variance flat patch analysis.
- **Periodic Interference**: Computes 2D FFT magnitude spectrum, subtracts broad background illumination via Gaussian spectral smoothing, and identifies symmetric outlier frequency spikes.
- **Uneven Illumination**: Applies a broad 61×61 spatial Gaussian blur to measure the illumination coefficient of variation ($CV = \sigma / \mu$) and regional corner-to-center brightness gradients.

### Phase 3: Noise Removal (`modules/noise_removal.py`)
- **Spatial Median Filtering**: 3×3 median filter with constant border padding.
- **Spatial Gaussian Smoothing**: Configurable kernel size and sigma for standard deviations.
- **FFT Low-Pass Filters**:
  - Ideal Low-Pass Filter (ILPF)
  - Gaussian Low-Pass Filter (GLPF)
  - Butterworth Low-Pass Filter (BLPF)
- **FFT Notch Filtering**: Multi-point Butterworth band-reject notch filters centered on detected frequency peaks to eliminate periodic lines without smearing ridge valleys.
- **Homomorphic Filtering**: High-emphasis frequency filtering in the logarithmic domain ($\ln(I + 1) \to \text{FFT} \to H(u,v) \to \text{IFFT} \to \exp$), balancing high-frequency reflectance while attenuating low-frequency illumination gradients.

### Phase 4: Ridge & Contrast Enhancement (`modules/image_enhancement.py`)
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Local contrast enhancement preventing noise over-amplification.
- **Power-Law (Gamma) Correction**: Contrast adjustment via intensity lookup tables.
- **Laplacian Sharpening**: Second-order derivative kernel enhancing ridge edge gradients:
  $$\text{Sharpened} = \text{Image} + \alpha \cdot \nabla^2(\text{Image})$$

### Phase 5: Segmentation & Background Whitening (`modules/segmentation.py`)
- Dual-cue foreground segmentation combining adaptive Gaussian thresholding and local intensity variance filtering.
- Morphological closing and opening using elliptical structuring elements to seal ridge gaps and eliminate small debris.
- Convex hull and largest-contour extraction to outline the true fingerprint area.
- Unwanted background conversion to pure white (`255`), clamping foreground ridge pixels below `254` to maintain watermark compatibility.

### Phase 6: Spatial-Domain Invisible Watermarking (`modules/watermark.py`)
- Embeds arbitrary UTF-8 text (e.g., student ID, biometric timestamp, device ID) exclusively into pure white background pixels (`255`).
- Encodes a 32-bit big-endian length header followed by byte data into binary bits:
  - Bit `0` $\to$ Pixel value `255`
  - Bit `1` $\to$ Pixel value `254`
- Ridge and valley pixels are completely unperturbed, guaranteeing zero degradation to biometric ridge patterns.
- Performs full extraction and integrity verification.

---

## 6. Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/daksh-kasana/TouchScan-.git
   cd TouchScan-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 7. How to Run TouchScan

### 7.1 Single Fingerprint Image Processing
Process an image with automatic noise diagnosis, enhancement, and background whitening:
```bash
python main.py --input data/original/101_1.tif --output outputs/processed/101_1_cleaned.png
```

### 7.2 Processing with Reference PSNR/SSIM Quality Evaluation
```bash
python main.py --input data/noisy/gaussian/101_1.png --reference data/original/101_1.tif --output outputs/processed/101_1_gaussian_cleaned.png
```

### 7.3 Invisible Watermark Embedding
Embed a student ID or reference text invisibly into the whitened background:
```bash
python main.py --input data/original/101_1.tif --watermark "BENNETT-CSET344-S24CSEU2714" --output outputs/processed/watermarked_101_1.png --no-display
```

### 7.4 Invisible Watermark Extraction
Extract an embedded watermark string from an image:
```bash
python main.py --input outputs/processed/watermarked_101_1.png --extract-watermark
```

### 7.5 Generating Controlled Noisy Benchmarking Copies
Generate controlled noisy datasets across all four degradation models:
```bash
python main.py --generate-noisy-dataset --samples 10
```

### 7.6 Running Full Dataset Batch Evaluation
Execute batch processing across the complete dataset, generating `outputs/evaluation/evaluation_results.csv` and summary plots:
```bash
python main.py --evaluate-dataset
```

---

## 8. Experimental Results & Quantitative Evaluation

Batch evaluation was executed across 180 total image tasks (including controlled degradation sets and original baseline FVC2002 images). All 180 tasks completed successfully with 0 failures.

### Quantitative PSNR & SSIM Summary

| Degradation Category | Input PSNR (dB) | Processed PSNR (dB) | PSNR Change | Input SSIM | Processed SSIM | SSIM Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Salt-and-Pepper Noise** | 17.30 dB | 19.41 dB | **+2.11 dB** | 0.4785 | 0.8251 | **+0.3466** |
| **Gaussian Noise** | 22.54 dB | 21.56 dB | -0.98 dB | 0.3511 | 0.8938 | **+0.5427** |
| **Periodic Noise** | 22.65 dB | 20.53 dB | -2.12 dB | 0.3784 | 0.7907 | **+0.4123** |
| **Uneven Illumination** | 9.36 dB | 6.14 dB | -3.22 dB | 0.8904 | 0.5905 | -0.2999 |
| **Original Baseline (FVC2002)** | *Ref / Clean* | 24.66 dB | N/A | 1.0000 | 0.9767 | -0.0233 |

### Metric Analysis
1. **SSIM Improvements**: Substantial structural similarity gains were achieved for Gaussian noise (+0.5427), periodic noise (+0.4123), and salt-and-pepper noise (+0.3466), demonstrating successful recovery of ridge structures.
2. **PSNR Dynamics**: Background whitening replaces non-uniform sensor borders with pure white (`255`), which naturally introduces numerical pixel differences against non-whitened baseline inputs, while preserving fingerprint ridge topology.
3. **Watermark Extraction**: 100% extraction accuracy across all tested payloads with zero distortion to biometric ridges.

---

## 9. Running Tests

Execute the comprehensive automated test suite covering all modules:
```bash
python -m unittest tests/test_touchscan.py
```

---

## 10. Limitations & Future Scope

### Limitations
- The spatial LSB watermark is fragile by design against lossy compression (JPEG) and geometric resampling.
- Extreme illumination non-uniformity across very faint latents may require manual parameter tuning.

### Future Scope
- Wavelet-domain (DWT) invisible watermarking for enhanced robustness against compression.
- Minutiae extraction integration (bifurcations and ridge endings) to evaluate biometric matching score retention post-cleaning.
