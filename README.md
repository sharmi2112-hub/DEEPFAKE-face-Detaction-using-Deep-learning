# Deepfake Face Detection Using Custom CNN Architectures

This repository contains the source code, experimental notebooks, trained model weights, and web interface for a research project on binary deepfake face detection using custom Convolutional Neural Networks (CNNs).

Live Web Application: https://deepfake-face-detaction-using-deep-learning-kz5ysgjthhxgv2y9hn.streamlit.app/

---

## Overview

Generative Adversarial Networks (GANs) and diffusion models have made synthetic media generation increasingly realistic, posing challenges for digital media forensics, cybersecurity, and content verification. While pre-trained deep architectures (such as ResNet or EfficientNet) achieve strong detection accuracy, they often incur significant computational and memory overhead.

This project investigates custom, lightweight CNN architectures developed from scratch for binary classification of real versus deepfake facial images. Through systematic comparative evaluation across three distinct network designs, this study highlights the trade-offs between model complexity, parameter volume, and classification performance.

---

## Dataset and Preprocessing

The experiments were conducted using a balanced facial image dataset obtained from the Hugging Face Hub (`JamieWithofs/Deepfake-and-real-images`).

### Dataset Division
- Training Set: 6,000 images (2,983 Fake, 3,017 Real)
- Validation Set: 1,200 images (582 Fake, 618 Real)
- Testing Set: 1,200 images (587 Fake, 613 Real)

### Data Preparation Pipeline
1. Color Space Conversion: Converted all images to RGB format.
2. Resizing: Standardized all input images to 64x64 pixels.
3. Pixel Normalization: Rescaled pixel values to the range [0.0, 1.0].
4. Data Augmentation: Applied random rotation (up to 15 degrees), width and height shifts (up to 10%), zoom (up to 10%), and horizontal flipping.

---

## Model Architectures and Empirical Evaluation

Three custom CNN architectures were implemented using TensorFlow and Keras and evaluated under identical training conditions (Adam optimizer, initial learning rate of 0.001, binary cross-entropy loss, 10 training epochs).

### Comparative Performance

| Architecture | Description | Parameters | Test Accuracy | Test Loss | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Model 1 (Baseline CNN) | 3 Convolutional Blocks + Max Pooling + Dense + Dropout | 1,142,081 | 74.50% | 0.5085 | 76.78% | 0.834 |
| Model 2 (Deep CNN) | 4 Convolutional Blocks + Batch Normalization + Max Pooling | 1,438,465 | 61.83% | 0.7640 | 70.38% | 0.782 |
| Model 3 (Wide CNN) | Stacked Convolutional Layers + Global Average Pooling | 303,649 | 48.92% | 0.6936 | 65.70% | 0.500 |

### Key Observations
- Model 1 (Baseline CNN) demonstrated the highest classification performance with 74.50% accuracy and an ROC-AUC of 0.834. It provided the best balance of feature representation without overfitting on 64x64 input resolutions.
- Model 2 experienced validation instability and overfitting due to excessive depth relative to dataset scale.
- Model 3 suffered from underfitting as aggressive spatial compression via Global Average Pooling removed fine-grained feature details necessary for binary classification.

---

## Web Application

The trained Model 1 weights (`deepfake_model.keras`) are integrated into an interactive web application built with Streamlit.

### Features
- Image Upload: Accepts JPG, JPEG, and PNG facial images.
- Real-Time Inference: Resizes and normalizes input images before passing them to the loaded Keras model.
- Classification Metrics: Displays the predicted class (Real vs. Deepfake) along with a confidence percentage.
- Technical Details: Provides raw probability scores, input tensor dimensions, and decision threshold information.

Live Link: https://deepfake-face-detaction-using-deep-learning-kz5ysgjthhxgv2y9hn.streamlit.app/

---

## Repository Structure

```
.
├── app.py                            # Streamlit web application script
├── deepfake_model.keras              # Trained Keras CNN model weights (Model 1)
├── Deepfake_Detection_CNN_Webpage.py # Python script version of training pipeline
├── Deepfake_Detection_CNN_Webpage.ipynb # Jupyter notebook containing EDA and model experiments
├── requirements.txt                  # Python package dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore rules
```

---

## Ethical and Legal Considerations

This research and software implementation are intended solely for academic study, digital forensics, and media authentication. The project utilizes publicly accessible secondary data containing no personally identifiable information.

---

## License

This project is licensed under the MIT License.
