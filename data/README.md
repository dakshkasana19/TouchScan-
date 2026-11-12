# Data Directory

This directory contains the fingerprint dataset used for evaluating TouchScan.

## Directory Structure

```
data/
├── original/             # Unmodified baseline fingerprint images (80 TIFF images and 80 lossless PNG copies)
│   ├── 101_1.tif ... 110_8.tif
│   └── 101_1.png ... 110_8.png
│
└── noisy/                # Controlled degraded copies for reproducible benchmarking
    ├── gaussian/
    ├── salt_pepper/
    ├── periodic/
    └── uneven_illumination/
```

## Dataset Summary

- **Source Dataset**: FVC2002 DB1_B benchmark dataset (Biometric Recognition Group - Università di Bologna).
- **Format**: 8-bit uncompressed grayscale images (`.tif` and `.png`).
- **Resolution**: 480 × 640 pixels at 500 dpi.
- **Sensor**: Low-cost optical sensor "TouchView II" by Identix.
- **Subjects**: 10 distinct subjects (indexed 101 to 110), with 8 impressions per subject (total 80 original impressions).

