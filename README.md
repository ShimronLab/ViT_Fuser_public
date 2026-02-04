# ViT-Fuser: Deep Personalized Priors for Longitudinal MRI

## Official Implementation
**Paper Title:** Deep personalized priors enable high-quality longitudinal MRI with low-cost systems  
**Authors:** Tal Oved, Beatrice Lena, Chloé F. Najac, Sheng Shen, Matthew S. Rosen, Andrew Webb, Efrat Shimron  

---

## Abstract
Magnetic resonance imaging (MRI) produces excellent high-contrast images, but its accessibility is limited by high costs. Low-field MRI provides affordable imaging but is hindered by low signal-to-noise ratio (SNR) and prolonged scan durations. 

This repository contains the official implementation of **ViT-Fuser**, a deep learning framework that extracts personalized features from past high-field MRI scans (priors) to accelerate and enhance follow-up low-field scans. ViT-Fuser enables diagnostic-quality scans on low-cost systems by synergistically integrating prior features with undersampled low-field data.

## Project Structure

The repository is organized as follows:

```text
ViTFuser_longitudinal_MRI
├── core_models
│   ├── models                  # Neural network architectures
│   │   ├── vision_transformer.py  # ViT Backbone
│   │   ├── ViTFuserWrap.py        # Main ViT-Fuser architecture
│   │   ├── ViTWrap.py             # Baseline ViT model
│   │   ├── unet/                  # U-Net architecture
│   │   ├── FSloss_wrap.py         # Feature-Space (Perceptual) Loss
│   │   └── ...
│   ├── runners                 # Scripts for training and evaluation
│   │   ├── Train_ViTFuser.py      # Main training script for ViT-Fuser
│   │   ├── Train_ViT.py           # Training script for baseline ViT
│   │   ├── Test_and_Compare.py    # Benchmarking and metric calculation
│   │   ├── Bias_Variance_Tradeoff.py
│   │   └── ...
│   └── utils                   # Helper functions
├── diffusion_mri               # Diffusion Probabilistic Models
│   ├── EDM
│   ├── models
│   └── ...
├── LICENSE
└── README.md
```

## Requirements

The code is implemented in Python using PyTorch. Key dependencies include:

* Python >= 3.8
* PyTorch
* NumPy
* fastMRI (for subsampling masks)
* SigPy (for Poisson Disc sampling)
* SciPy
* Matplotlib (for visualization)

To install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

To reproduce the results, please download the Lumiere Paired Dataset and organize the files as follows. 

* **Training Data:** Place the Lumiere Paired training data in the `core_models/registered_data` folder.
* **Test Data:** Place the Lumiere Paired test data in the `core_models/test_data` folder.

## Citation

If you use this code or the paper's results, please cite:
