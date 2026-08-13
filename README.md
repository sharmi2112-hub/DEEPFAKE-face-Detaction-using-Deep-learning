# 🕵️ DEEPFAKE Face Detection using Deep Learning (CNN)

An interactive web application built with **Streamlit** and **TensorFlow/Keras** to detect whether a facial image is **Real** or a **Deepfake (Synthetic Generation)**.

---

## 📌 Features

- 🖼️ **Image Upload**: Supports JPG, JPEG, and PNG image formats.
- ⚡ **Real-Time Classification**: Instant Deepfake vs Real face classification powered by a custom Convolutional Neural Network (CNN).
- 📊 **Confidence Metrics**: Displays prediction confidence percentage, raw probability, and visual metric indicators.
- 🚀 **Streamlit Cloud Ready**: Easily deployable to Streamlit Community Cloud.

---

## 🛠️ Project Structure

```
├── app.py                      # Main Streamlit web application
├── deepfake_model.keras        # Trained Keras CNN model weights
├── requirements.txt            # Dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
└── ARYA_Deepfake_Detection... # Training notebook
```

---

## 💻 Local Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sharmi2112-hub/DEEPFAKE-face-Detaction-using-Deep-learning.git
   cd DEEPFAKE-face-Detaction-using-Deep-learning
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**: Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app** and connect your GitHub account.
3. Select Repository: `sharmi2112-hub/DEEPFAKE-face-Detaction-using-Deep-learning`.
4. Set Main file path: `app.py`.
5. Click **Deploy!**
