# Explainable AI-Based Image Authenticity Verification and AI Edit Detection using Vision Foundation Models

A final-year research capstone project for single-image authenticity assessment using **DINOv2-Small (ViT-S/14)**, calibrated classification, approximate patch-level attribution, and representation-based out-of-distribution (OOD) analysis.

The system predicts whether an input image is **Authentic** or **AI-Generated / Synthetic** and provides supporting diagnostic visualizations.

> **Research scope:** The primary benchmark uses CIFAKE-style synthetic imagery. The output is an AI-based assessment, not definitive forensic proof or pixel-level tampering localization.

---

## Project Overview

| Component               | Implementation                           |
| ----------------------- | ---------------------------------------- |
| Primary backbone        | DINOv2-Small (ViT-S/14)                  |
| Backbone representation | 384-dimensional CLS embedding            |
| Classifier              | Lightweight MLP classification head      |
| Backbone training       | Frozen feature extractor                 |
| Explainability          | Gradient-based spatial patch attribution |
| Calibration             | Temperature scaling                      |
| Uncertainty indicator   | Predictive Shannon entropy               |
| OOD indicator           | Minimum class-centroid cosine distance   |
| Frequency analysis      | Compact FFT feature ablation             |
| Interface               | Streamlit                                |
| Framework               | PyTorch                                  |
| Runtime target          | CPU-compatible Windows environment       |
| Task                    | Binary image classification              |

## Project Objectives

* Accept one image without requiring an original/reference image.
* Predict Authentic or AI-Generated / Synthetic.
* Display calibrated class probabilities and predictive entropy.
* Generate a 16×16 patch attribution map and visualization overlay.
* Provide a representation-distance OOD indicator.
* Evaluate frequency-feature fusion, robustness, and optional test-time augmentation.
* Preserve reproducible evaluation artifacts and document limitations.

## Dataset

The primary experimental dataset is **CIFAKE**, containing authentic CIFAR-10-derived images and synthetic images.

The project uses the following local class directories:

```text
dataset/
├── authentic/
└── ai_edited/
```

The prepared experimental split contains:

| Split      |  Authentic | AI-generated |      Total |
| ---------- | ---------: | -----------: | ---------: |
| Train      |      8,000 |        8,000 |     16,000 |
| Validation |      1,000 |        1,000 |      2,000 |
| Test       |      1,000 |        1,000 |      2,000 |
| **Total**  | **10,000** |   **10,000** | **20,000** |

The synthetic class is not a benchmark of localized human image splicing or inpainting. The dataset's small CIFAR-10-derived images are standardized to the model input size.

## Main Held-Out Test Results

The following metrics are from the 2,000-image held-out test split for the DINOv2 CLS classifier.

| Metric          | Result |
| --------------- | -----: |
| Accuracy        | 93.85% |
| Macro Precision | 93.87% |
| Macro Recall    | 93.85% |
| Macro F1-score  | 93.85% |
| ROC-AUC         | 98.41% |

### Confusion Matrix

| Actual / Predicted | Authentic | Synthetic |
| ------------------ | --------: | --------: |
| Authentic          |       927 |        73 |
| Synthetic          |        50 |       950 |

* Specificity: 92.70%
* Sensitivity: 95.00%

These results describe the evaluated CIFAKE test split and should not be interpreted as verified performance on every contemporary image generator or real-world forensic scenario.

## Calibration and Uncertainty

Temperature scaling was fitted using the validation split.

| Measure                    |    Raw | Calibrated |
| -------------------------- | -----: | ---------: |
| Expected Calibration Error |  1.00% |      0.73% |
| Negative Log-Likelihood    | 0.1608 |     0.1604 |
| Brier Score                | 0.0936 |     0.0933 |

Calibration adjusts probability confidence without changing the underlying class ranking or predicted class.

On the evaluated test set, incorrect predictions had higher average predictive entropy than correct predictions. Entropy is presented as an operational uncertainty indicator, not a guarantee that every incorrect prediction will be detected.

## Explainability

The system computes gradient-based attribution over DINOv2 spatial patch tokens:

1. Extract spatial patch representations.
2. Compute gradients for the selected class score.
3. Calculate patch attribution values.
4. Arrange the 256 values into a 16×16 grid.
5. Upsample the map for visualization.
6. Overlay the attribution visualization on the input image.

**Important:** This is approximate model attribution, not a ground-truth tampering mask or exact pixel-level manipulation segmentation.

## Frequency Ablation

A compact 32-dimensional FFT descriptor was evaluated as an additional feature branch.

| Model               | Accuracy | Macro F1 | ROC-AUC |
| ------------------- | -------: | -------: | ------: |
| DINOv2 CLS baseline |   93.85% |   93.85% |  98.41% |
| DINOv2 + FFT        |   93.75% |   93.75% |  98.62% |

The FFT branch produced a small ROC-AUC increase while thresholded accuracy and F1 were slightly lower. It is retained as an ablation and diagnostic component rather than claimed as an overall performance improvement.

## Robustness and Additional Experiments

A separate balanced 500-image test subset was used for corruption and TTA experiments.

* Clean subset accuracy: 92.20%.
* Horizontal-flip TTA accuracy: 92.60%.
* Measured TTA latency overhead: approximately 2.01×.
* Blur, downscale-upscale, and strong Gaussian noise caused substantial performance degradation.

OOD thresholding was evaluated using in-distribution data. A genuine external real-world OOD benchmark was not available; random-vector proxy results are not treated as evidence of real-world OOD performance.

## System Pipeline

```text
Single Input Image
        |
        v
Image Preprocessing (224 × 224)
        |
        v
Frozen DINOv2-Small (ViT-S/14)
        |
        v
384-D CLS Representation
        |
        v
Trained MLP Classification Head
        |
        +----> Temperature-Scaled Probabilities
        |
        +----> Predictive Entropy
        |
        +----> Patch Attribution Heatmap
        |
        +----> OOD Representation Distance
        |
        +----> Optional TTA / FFT Diagnostics
        |
        v
Streamlit Research Demonstration
```

## Project Structure

```text
Explainable_AI_Image_Authenticity_Verification/
├── app/
│   ├── streamlit_app.py
│   ├── models_vfm.py
│   ├── explainability.py
│   ├── calibration.py
│   ├── ood_detection.py
│   ├── frequency_features.py
│   ├── cache_features.py
│   ├── train_vfm.py
│   ├── predict_vfm.py
│   ├── run_calibration.py
│   ├── run_explainability.py
│   ├── run_error_analysis.py
│   ├── run_phase6.py
│   └── validate_streamlit_pipeline.py
├── models/
│   ├── dinov2_authenticity_head.pth
│   ├── dinov2_frequency_fusion.pth
│   ├── calibration_params.json
│   └── ood_params.json
├── notebooks/
├── outputs/
├── dataset/                 # Not included
├── requirements.txt
└── README.md
```

The pretrained DINOv2 backbone weights and local repository/cache are not included in this source tree listing. Follow the setup notes below to provide the required local model artifacts.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/95F2A42637/Capstone.git
cd Capstone
```

### 2. Create and activate a virtual environment

Windows CMD:

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Provide DINOv2 pretrained assets

The application uses the local DINOv2-Small repository and pretrained checkpoint. Place the compatible local repository and pretrained weights at the paths expected by `app/models_vfm.py`.

The trained classifier and parameter files should be placed under `models/`.

Do not assume that downloading the source repository alone provides the pretrained checkpoint.

### 5. Prepare the dataset (for retraining or evaluation)

Place the dataset into the documented `dataset/` structure and prepare the required train, validation, and test splits.

The dataset itself is not committed to this repository.

### 6. Run the final application

From the repository root:

```cmd
python -m streamlit run app/streamlit_app.py
```

Open the local Streamlit address shown in the terminal, usually:

```text
http://localhost:8501
```

## Application Features

* Single-image upload and benchmark presets.
* Authentic / AI-generated prediction.
* Calibrated class probabilities.
* Predictive entropy.
* Patch attribution heatmap and overlay.
* OOD representation-distance indicator.
* Optional horizontal-flip TTA.
* Optional FFT spectrum diagnostics.
* Research limitations and technical details.

## Reproducibility

The repository includes training, evaluation, calibration, explainability, robustness, and audit scripts alongside experiment reports.

Reported metrics were obtained using the preserved experiment configuration and model artifacts in the documented CPU environment. Re-running experiments may require the dataset, pretrained backbone assets, and compatible dependencies.

## Limitations

* Evaluation is primarily based on CIFAKE-style synthetic imagery.
* Generalization to contemporary generators such as Midjourney, Flux, DALL·E, SDXL, and unseen editing pipelines has not been established.
* Severe blur, downscaling, and noise can reduce classification performance.
* Attribution is approximate and has no pixel-level ground-truth mask validation.
* External real-world OOD validation is incomplete.
* TTA increases CPU inference latency.
* High confidence does not guarantee correctness.

## Research Disclaimer

This system provides an AI-based image authenticity assessment, not definitive forensic proof. Its heatmap indicates approximate model attribution and must not be interpreted as verified tampering localization.

## License and Academic Use

Developed as a final-year academic capstone project.

## Author

GitHub: [95F2A42637](https://github.com/95F2A42637)
