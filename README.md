# 🕵️ Deepfake Face Detection Using Custom CNN Architectures

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.style=for-the-badge)](LICENSE)

An end-to-end Deep Learning research project and interactive web application designed to classify facial images as **Real** or **Deepfake (Synthetic Generation)** using custom Convolutional Neural Network (CNN) architectures built from scratch.

---

## 📌 Executive Summary

With the rapid progress of Generative Adversarial Networks (GANs) and diffusion models, deepfakes pose significant challenges to digital media authenticity, cybersecurity, and online trust. While large pre-trained vision models (e.g., ResNet, EfficientNet) achieve high accuracy, they require heavy computational resources and large memory footprints. 

This project explores **lightweight, custom CNN models built de novo** without relying on heavy transfer learning. By systematically comparing three distinct architectural strategies on a standardized dataset of facial images, this research demonstrates that a carefully designed, lightweight baseline CNN can achieve superior classification performance (**74.50% test accuracy, 0.834 ROC-AUC**) with low latency and minimal resource overhead.

---

## 🏗️ System Architecture & Pipeline

```
  ┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────────┐
  │  Input Image    │ ───► │ Preprocessing & EDA │ ───► │ Custom CNN Architecture │
  │  (Upload/Batch) │      │ (Resize, Normalize) │      │ (Model 1 / 2 / 3)       │
  └─────────────────┘      └─────────────────────┘      └────────────┬────────────┘
                                                                     │
  ┌─────────────────┐      ┌─────────────────────┐                   │
  │ Streamlit Web   │ ◄─── │ Inference Engine    │ ◄─────────────────┘
  │ User Interface  │      │ (Real vs Deepfake)  │
  └─────────────────┘      └─────────────────────┘
```

---

## 📊 Dataset & Preprocessing

The model is trained and evaluated using a balanced dataset of facial images sourced from Hugging Face Hub (`JamieWithofs/Deepfake-and-real-images`).

* **Dataset Split**:
  * **Training Set**: 6,000 images (2,983 Fake, 3,017 Real)
  * **Validation Set**: 1,200 images (582 Fake, 618 Real)
  * **Testing Set**: 1,200 images (587 Fake, 613 Real)
* **Preprocessing Steps**:
  1. **Color Conversion**: Converted to RGB format.
  2. **Resolution Standardization**: Resized to $64 \times 64 \times 3$.
  3. **Pixel Normalization**: Scaled pixel values to $[0.0, 1.0]$.
  4. **Data Augmentation**: Applied random rotation ($\pm 15^\circ$), horizontal/vertical shifting ($\pm 10\%$), zoom ($\pm 10\%$), and horizontal flips.

---

## 🤖 Model Architectures Comparison

Three distinct custom CNN architectures were engineered and compared under identical training hyper-parameters (Adam optimizer, learning rate = 0.001, Binary Cross-Entropy loss, 10 epochs):

| Model Architecture | Description | Parameters | Test Accuracy | Test Loss | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Baseline CNN)** 🏆 | 3 Conv Blocks + MaxPool + Dense + Dropout (0.5) | **1,142,081** | **74.50%** | **0.5085** | **76.78%** | **0.834** |
| **Model 2 (Deep CNN + BatchNorm)** | 4 Conv Blocks + Batch Normalization + MaxPool | 1,438,465 | 61.83% | 0.7640 | 70.38% | 0.782 |
| **Model 3 (Wide CNN + GAP)** | Stacked Conv Layers + Global Average Pooling | 303,649 | 48.92% | 0.6936 | 65.70% | 0.500 |

### Key Findings:
- **Model 1 (Baseline CNN)** achieved the best overall performance with **74.50% accuracy** and an **AUC of 0.834**, demonstrating that simpler feature representations generalize better on compact image resolutions without overfitting.
- **Model 2** exhibited signs of overfitting due to excessive depth and parameter volume relative to input resolution.
- **Model 3** experienced underfitting due to aggressive spatial reduction via Global Average Pooling without sufficient dense feature mapping.

---

## 🌐 Interactive Streamlit Web Application

The best-performing model (`deepfake_model.keras`) is integrated into a lightweight **Streamlit** web application for real-time inference.

### Features:
- 📤 **Image Upload**: Accepts JPG, JPEG, and PNG image formats.
- ⚡ **Real-Time Classification**: Instant preprocessing and inference.
- 🎯 **Visual Confidence Gauge**: Displays prediction probabilities, classification verdict badges, and confidence metrics.
- 🔍 **Technical Inspection**: Expandable view showing raw tensor shapes, prediction probabilities, and decision thresholds.

---

## 📁 Repository Structure

```
├── app.py                            # Streamlit web application interface
├── deepfake_model.keras              # Trained Keras model weights (Baseline CNN)
├── Deepfake_Detection_CNN_Webpage.py # Python script version of training pipeline
├── Deepfake_Detection_CNN_Webpage.ipynb # Jupyter notebook with full EDA & experiments
├── requirements.txt                  # Python dependencies for deployment
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore rules for cached & temporary files
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/sharmi2112-hub/DEEPFAKE-face-Detaction-using-Deep-learning.git
cd DEEPFAKE-face-Detaction-using-Deep-learning
pip install -r requirements.txt
```

### 2. Run Streamlit Application Locally

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment

This repository is pre-configured for **Streamlit Community Cloud** deployment:

1. Log in to [share.streamlit.io](https://share.streamlit.io/).
2. Click **New App** and select repository `sharmi2112-hub/DEEPFAKE-face-Detaction-using-Deep-learning`.
3. Set **Main file path**: `app.py`.
4. Click **Deploy!**

---

## ⚖️ Ethical, Legal & Social Considerations

- **Ethical AI**: This system is developed strictly for research, digital forensics, and media authentication. It must not be deployed for unauthorized surveillance.
- **Dataset Transparency**: Utilizes publicly accessible, anonymized secondary datasets with no personally identifiable information (PII).
- **Algorithmic Fairness**: Mitigates class imbalance through equal representation of real and synthetic face distributions.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
