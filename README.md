# 🔍 Explainable AI Image Authenticity Verification

A Final Year Capstone Project that detects whether an image is **Authentic** or **AI-Generated/Manipulated** using Deep Learning and provides visual explanations using **Grad-CAM**.

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Model** | ResNet18 (ImageNet pretrained, fine-tuned) |
| **Explainability** | Grad-CAM |
| **Dataset** | CIFAKE (Authentic + AI-Edited) |
| **Framework** | PyTorch + Streamlit |
| **Task** | Binary Classification (Authentic vs AI-Generated) |

---

## 🎯 Objective

Build an AI system that:
- Accepts a **single image** (no reference image required)
- Predicts whether it is **Authentic** or **AI-Generated/Manipulated**
- **Explains** the prediction by highlighting suspicious image regions using Grad-CAM

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 96.00% |
| Precision | 96.00% |
| Recall | 96.00% |
| F1-Score | 96.00% |

---

## 🗂️ Project Structure

```
Explainable_AI_Image_Authenticity_Verification/
│
├── app/
│   ├── app.py                # Streamlit web application
│   ├── train_model.py        # Model training pipeline
│   ├── explain.py            # Grad-CAM explanation module
│   └── prepare_dataset.py    # Dataset preparation script
│
├── models/
│   └── resnet18_authenticity.pth   # Trained model weights (not in repo)
│
├── outputs/
│   ├── accuracy_curve.png    # Training/Validation accuracy curve
│   ├── loss_curve.png        # Training/Validation loss curve
│   ├── confusion_matrix.png  # Confusion matrix
│   ├── test_report.txt       # Classification report
│   └── gradcam_result.jpg    # Sample Grad-CAM output
│
├── dataset/                  # Dataset (not included - see below)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/95F2A42637/Capstone.git
cd Capstone
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
Download the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) and place it as:
```
dataset/
├── authentic/    # Real images
└── ai_edited/    # AI-generated images
```

Then run:
```bash
python app/prepare_dataset.py
```

### 5. Train the Model
```bash
python app/train_model.py
```

### 6. Run the Streamlit App
```bash
streamlit run app/app.py
```
Open your browser at: **http://localhost:8501**

---

## 🖼️ How It Works

```
User uploads image
        ↓
Image resized to 224×224 and normalized
        ↓
ResNet18 extracts visual features
        ↓
Classifier predicts Authentic / AI-Generated
        ↓
Confidence score calculated
        ↓
Grad-CAM generates visual explanation heatmap
```

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| PyTorch 2.14 | Deep learning framework |
| Torchvision | ResNet18 model + transforms |
| pytorch-grad-cam | Grad-CAM explainability |
| Streamlit 1.64 | Web application interface |
| scikit-learn | Evaluation metrics |
| Matplotlib | Visualization |
| NumPy / Pillow | Image processing |

---

## 📸 Sample Output

The system provides:
- **Prediction:** Authentic or AI-Generated
- **Confidence Score:** Probability of each class
- **Grad-CAM Heatmap:** Visual explanation of model decision

---

## ⚠️ Limitations

- Trained on CIFAKE dataset (CIFAR-10 based images)
- Grad-CAM provides approximate visual explanations
- May not generalize to all types of AI generation tools
- No reference image required — single image analysis only

---

## 📄 License

This project is developed as a Final Year Capstone Project for academic purposes.

---

## 👨‍💻 Author

**GitHub:** [95F2A42637](https://github.com/95F2A42637)
