# Core libraries
import numpy as np
import pandas as pd
import random
import time
import collections

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset loading
from datasets import load_dataset
from PIL import Image

# Deep learning
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)

# Reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Inline plots
%matplotlib inline
sns.set_style("whitegrid")

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))


# ==================================================


# Load dataset directly from Hugging Face Hub
dataset = load_dataset("JamieWithofs/Deepfake-and-real-images")

# Display dataset structure (splits and row counts)
print(dataset)


# ==================================================


# Take a manageable, shuffled subset of each split for training/evaluation
TRAIN_SAMPLES = 6000
VAL_SAMPLES = 1200
TEST_SAMPLES = 1200

train_subset = dataset['train'].shuffle(seed=42).select(range(TRAIN_SAMPLES))
val_subset = dataset['validation'].shuffle(seed=42).select(range(VAL_SAMPLES))
test_subset = dataset['test'].shuffle(seed=42).select(range(TEST_SAMPLES))

print("Train subset size:", len(train_subset))
print("Validation subset size:", len(val_subset))
print("Test subset size:", len(test_subset))


# ==================================================


# Inspect a single sample record
sample = train_subset[0]
print("Sample keys:", sample.keys())
print("Label:", sample['label'], "(0 = Fake, 1 = Real)")
print("Image size:", sample['image'].size)
print("Image mode:", sample['image'].mode)


# ==================================================


# Class distribution across each split
train_labels = train_subset['label']
val_labels = val_subset['label']
test_labels = test_subset['label']

print("Train class distribution:", collections.Counter(train_labels))
print("Validation class distribution:", collections.Counter(val_labels))
print("Test class distribution:", collections.Counter(test_labels))


# ==================================================


# Check for missing / null labels across splits
print("Missing labels in train:", sum(l is None for l in train_labels))
print("Missing labels in val:", sum(l is None for l in val_labels))
print("Missing labels in test:", sum(l is None for l in test_labels))

# Check image size consistency across a small sample
sizes = [train_subset[i]['image'].size for i in range(20)]
print("Sample image sizes (first 20):", set(sizes))


# ==================================================


# Target image dimensions for CNN input
IMG_SIZE = 64

# Convert training images to RGB numpy arrays and resize
train_images_list = []
for img in train_subset['image']:
    img_resized = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    train_images_list.append(np.array(img_resized))
X_train_raw = np.array(train_images_list, dtype='float32')
y_train = np.array(train_subset['label'], dtype='int32')

print("X_train_raw shape:", X_train_raw.shape)
print("y_train shape:", y_train.shape)


# ==================================================


# Convert validation images to RGB numpy arrays and resize
val_images_list = []
for img in val_subset['image']:
    img_resized = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    val_images_list.append(np.array(img_resized))
X_val_raw = np.array(val_images_list, dtype='float32')
y_val = np.array(val_subset['label'], dtype='int32')

print("X_val_raw shape:", X_val_raw.shape)
print("y_val shape:", y_val.shape)


# ==================================================


# Convert test images to RGB numpy arrays and resize
test_images_list = []
for img in test_subset['image']:
    img_resized = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    test_images_list.append(np.array(img_resized))
X_test_raw = np.array(test_images_list, dtype='float32')
y_test = np.array(test_subset['label'], dtype='int32')

print("X_test_raw shape:", X_test_raw.shape)
print("y_test shape:", y_test.shape)


# ==================================================


# Basic pixel value range check before normalization
print("Min pixel value:", X_train_raw.min())
print("Max pixel value:", X_train_raw.max())
print("Mean pixel value:", X_train_raw.mean())


# ==================================================


# Graph 1: Class distribution in the training set
plt.figure(figsize=(6, 4))
labels_names = ['Fake (0)', 'Real (1)']
counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
sns.barplot(x=labels_names, y=counts, palette='viridis')
plt.title('Class Distribution - Training Set')
plt.ylabel('Number of Images')
plt.show()


# ==================================================


# Graph 2: Sample images grid - Fake vs Real
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fake_idx = np.where(y_train == 0)[0][:5]
real_idx = np.where(y_train == 1)[0][:5]

for col, idx in enumerate(fake_idx):
    axes[0, col].imshow(X_train_raw[idx].astype('uint8'))
    axes[0, col].axis('off')
    axes[0, col].set_title('Fake')

for col, idx in enumerate(real_idx):
    axes[1, col].imshow(X_train_raw[idx].astype('uint8'))
    axes[1, col].axis('off')
    axes[1, col].set_title('Real')

plt.suptitle('Sample Images: Fake (top) vs Real (bottom)')
plt.tight_layout()
plt.show()


# ==================================================


# Graph 3: Overall pixel intensity histogram
plt.figure(figsize=(7, 4))
plt.hist(X_train_raw.flatten(), bins=50, color='steelblue')
plt.title('Overall Pixel Intensity Distribution')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')
plt.show()


# ==================================================


# Graph 4: Average ("mean") image per class
mean_fake_img = X_train_raw[y_train == 0].mean(axis=0).astype('uint8')
mean_real_img = X_train_raw[y_train == 1].mean(axis=0).astype('uint8')

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(mean_fake_img)
axes[0].set_title('Average Fake Image')
axes[0].axis('off')
axes[1].imshow(mean_real_img)
axes[1].set_title('Average Real Image')
axes[1].axis('off')
plt.show()


# ==================================================


# Graph 5: RGB channel mean value comparison across classes
fake_channel_means = X_train_raw[y_train == 0].mean(axis=(0, 1, 2))
real_channel_means = X_train_raw[y_train == 1].mean(axis=(0, 1, 2))

channel_df = pd.DataFrame({
    'Channel': ['Red', 'Green', 'Blue'] * 2,
    'Mean Value': list(fake_channel_means) + list(real_channel_means),
    'Class': ['Fake'] * 3 + ['Real'] * 3
})

plt.figure(figsize=(7, 4))
sns.barplot(data=channel_df, x='Channel', y='Mean Value', hue='Class', palette='coolwarm')
plt.title('Mean RGB Channel Values by Class')
plt.show()


# ==================================================


# Graph 6: Image brightness distribution by class
fake_brightness = X_train_raw[y_train == 0].mean(axis=(1, 2, 3))
real_brightness = X_train_raw[y_train == 1].mean(axis=(1, 2, 3))

plt.figure(figsize=(7, 4))
sns.kdeplot(fake_brightness, label='Fake', fill=True)
sns.kdeplot(real_brightness, label='Real', fill=True)
plt.title('Image Brightness Distribution by Class')
plt.xlabel('Average Brightness')
plt.legend()
plt.show()


# ==================================================


# Graph 7: Class distribution across train/validation/test splits
split_df = pd.DataFrame({
    'Split': ['Train', 'Train', 'Validation', 'Validation', 'Test', 'Test'],
    'Class': ['Fake', 'Real'] * 3,
    'Count': [np.sum(y_train == 0), np.sum(y_train == 1),
              np.sum(y_val == 0), np.sum(y_val == 1),
              np.sum(y_test == 0), np.sum(y_test == 1)]
})

plt.figure(figsize=(8, 4))
sns.barplot(data=split_df, x='Split', y='Count', hue='Class', palette='mako')
plt.title('Class Distribution Across Splits')
plt.show()


# ==================================================


# Graph 8: Per-channel pixel value distribution (R, G, B)
plt.figure(figsize=(7, 4))
colors = ['red', 'green', 'blue']
channel_names = ['Red', 'Green', 'Blue']
for i in range(3):
    plt.hist(X_train_raw[:, :, :, i].flatten(), bins=50, color=colors[i],
             alpha=0.5, label=channel_names[i])
plt.title('Per-Channel Pixel Value Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# ==================================================


# Normalize pixel values to the [0, 1] range
X_train = X_train_raw / 255.0
X_val = X_val_raw / 255.0
X_test = X_test_raw / 255.0

print("X_train range:", X_train.min(), "-", X_train.max())
print("X_val range:", X_val.min(), "-", X_val.max())
print("X_test range:", X_test.min(), "-", X_test.max())


# ==================================================


# Reshape labels for binary classification (sigmoid output)
y_train = y_train.reshape(-1, 1).astype('float32')
y_val = y_val.reshape(-1, 1).astype('float32')
y_test = y_test.reshape(-1, 1).astype('float32')

print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)


# ==================================================


# Data augmentation generator (feature engineering step) for the training set
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
train_datagen.fit(X_train)

BATCH_SIZE = 32
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=42)

print("Augmented training generator ready. Steps per epoch:", len(train_generator))


# ==================================================


# Input shape used by all three CNN models
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
print("CNN input shape:", INPUT_SHAPE)


# ==================================================


# Model 1: Build - Simple Baseline CNN (Sequential, no pretrained weights)
model_1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
], name='Model_1_Baseline_CNN')

model_1.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

model_1.summary()


# ==================================================


# Model 1: Train
start_time = time.time()

history_1 = model_1.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=10,
    verbose=1
)

model_1_train_time = time.time() - start_time
print("Model 1 training time (seconds):", round(model_1_train_time, 2))


# ==================================================


# Model 1: Test - run inference on the held-out test set
model_1_probs = model_1.predict(X_test)
model_1_preds = (model_1_probs > 0.5).astype('int32')

print("Model 1 sample predictions:", model_1_preds[:10].flatten())
print("Model 1 sample true labels:", y_test[:10].flatten())


# ==================================================


# Model 1: Evaluate
model_1_test_loss, model_1_test_acc = model_1.evaluate(X_test, y_test, verbose=0)

model_1_accuracy = accuracy_score(y_test, model_1_preds)
model_1_precision = precision_score(y_test, model_1_preds)
model_1_recall = recall_score(y_test, model_1_preds)
model_1_f1 = f1_score(y_test, model_1_preds)

print("Model 1 Test Loss:", round(model_1_test_loss, 4))
print("Model 1 Test Accuracy:", round(model_1_accuracy, 4))
print("Model 1 Precision:", round(model_1_precision, 4))
print("Model 1 Recall:", round(model_1_recall, 4))
print("Model 1 F1-score:", round(model_1_f1, 4))
print("\nClassification Report:\n", classification_report(y_test, model_1_preds, target_names=['Fake', 'Real']))


# ==================================================


# Model 1: Confusion matrix and training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

cm_1 = confusion_matrix(y_test, model_1_preds)
sns.heatmap(cm_1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
axes[0].set_title('Model 1 - Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

axes[1].plot(history_1.history['accuracy'], label='Train Accuracy')
axes[1].plot(history_1.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Model 1 - Accuracy Curve')
axes[1].legend()

axes[2].plot(history_1.history['loss'], label='Train Loss')
axes[2].plot(history_1.history['val_loss'], label='Val Loss')
axes[2].set_title('Model 1 - Loss Curve')
axes[2].legend()

plt.tight_layout()
plt.show()


# ==================================================


# Model 2: Build - Deeper CNN with Batch Normalization
model_2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
], name='Model_2_Deep_BatchNorm_CNN')

model_2.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

model_2.summary()


# ==================================================


# Model 2: Train
start_time = time.time()

history_2 = model_2.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=10,
    verbose=1
)

model_2_train_time = time.time() - start_time
print("Model 2 training time (seconds):", round(model_2_train_time, 2))


# ==================================================


# Model 2: Test - run inference on the held-out test set
model_2_probs = model_2.predict(X_test)
model_2_preds = (model_2_probs > 0.5).astype('int32')

print("Model 2 sample predictions:", model_2_preds[:10].flatten())
print("Model 2 sample true labels:", y_test[:10].flatten())


# ==================================================


# Model 2: Evaluate
model_2_test_loss, model_2_test_acc = model_2.evaluate(X_test, y_test, verbose=0)

model_2_accuracy = accuracy_score(y_test, model_2_preds)
model_2_precision = precision_score(y_test, model_2_preds)
model_2_recall = recall_score(y_test, model_2_preds)
model_2_f1 = f1_score(y_test, model_2_preds)

print("Model 2 Test Loss:", round(model_2_test_loss, 4))
print("Model 2 Test Accuracy:", round(model_2_accuracy, 4))
print("Model 2 Precision:", round(model_2_precision, 4))
print("Model 2 Recall:", round(model_2_recall, 4))
print("Model 2 F1-score:", round(model_2_f1, 4))
print("\nClassification Report:\n", classification_report(y_test, model_2_preds, target_names=['Fake', 'Real']))


# ==================================================


# Model 2: Confusion matrix and training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

cm_2 = confusion_matrix(y_test, model_2_preds)
sns.heatmap(cm_2, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
axes[0].set_title('Model 2 - Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

axes[1].plot(history_2.history['accuracy'], label='Train Accuracy')
axes[1].plot(history_2.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Model 2 - Accuracy Curve')
axes[1].legend()

axes[2].plot(history_2.history['loss'], label='Train Loss')
axes[2].plot(history_2.history['val_loss'], label='Val Loss')
axes[2].set_title('Model 2 - Loss Curve')
axes[2].legend()

plt.tight_layout()
plt.show()


# ==================================================


# Model 3: Build - Wide CNN with Global Average Pooling
model_3 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
], name='Model_3_Wide_GAP_CNN')

model_3.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

model_3.summary()


# ==================================================


# Model 3: Train
start_time = time.time()

history_3 = model_3.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=10,
    verbose=1
)

model_3_train_time = time.time() - start_time
print("Model 3 training time (seconds):", round(model_3_train_time, 2))


# ==================================================


# Model 3: Test - run inference on the held-out test set
model_3_probs = model_3.predict(X_test)
model_3_preds = (model_3_probs > 0.5).astype('int32')

print("Model 3 sample predictions:", model_3_preds[:10].flatten())
print("Model 3 sample true labels:", y_test[:10].flatten())


# ==================================================


# Model 3: Evaluate
model_3_test_loss, model_3_test_acc = model_3.evaluate(X_test, y_test, verbose=0)

model_3_accuracy = accuracy_score(y_test, model_3_preds)
model_3_precision = precision_score(y_test, model_3_preds)
model_3_recall = recall_score(y_test, model_3_preds)
model_3_f1 = f1_score(y_test, model_3_preds)

print("Model 3 Test Loss:", round(model_3_test_loss, 4))
print("Model 3 Test Accuracy:", round(model_3_accuracy, 4))
print("Model 3 Precision:", round(model_3_precision, 4))
print("Model 3 Recall:", round(model_3_recall, 4))
print("Model 3 F1-score:", round(model_3_f1, 4))
print("\nClassification Report:\n", classification_report(y_test, model_3_preds, target_names=['Fake', 'Real']))


# ==================================================


# Model 3: Confusion matrix and training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

cm_3 = confusion_matrix(y_test, model_3_preds)
sns.heatmap(cm_3, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
axes[0].set_title('Model 3 - Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

axes[1].plot(history_3.history['accuracy'], label='Train Accuracy')
axes[1].plot(history_3.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Model 3 - Accuracy Curve')
axes[1].legend()

axes[2].plot(history_3.history['loss'], label='Train Loss')
axes[2].plot(history_3.history['val_loss'], label='Val Loss')
axes[2].set_title('Model 3 - Loss Curve')
axes[2].legend()

plt.tight_layout()
plt.show()


# ==================================================


# Comparison table across all three models
comparison_df = pd.DataFrame({
    'Model': ['Model 1: Baseline CNN', 'Model 2: Deep BatchNorm CNN', 'Model 3: Wide GAP CNN'],
    'Test Accuracy': [model_1_accuracy, model_2_accuracy, model_3_accuracy],
    'Precision': [model_1_precision, model_2_precision, model_3_precision],
    'Recall': [model_1_recall, model_2_recall, model_3_recall],
    'F1-score': [model_1_f1, model_2_f1, model_3_f1],
    'Test Loss': [model_1_test_loss, model_2_test_loss, model_3_test_loss],
    'Params': [model_1.count_params(), model_2.count_params(), model_3.count_params()],
    'Train Time (s)': [round(model_1_train_time, 1), round(model_2_train_time, 1), round(model_3_train_time, 1)]
})

comparison_df


# ==================================================


# Graph: Accuracy / Precision / Recall / F1 comparison across models
metrics_melt = comparison_df.melt(
    id_vars='Model',
    value_vars=['Test Accuracy', 'Precision', 'Recall', 'F1-score'],
    var_name='Metric', value_name='Score'
)

plt.figure(figsize=(10, 5))
sns.barplot(data=metrics_melt, x='Metric', y='Score', hue='Model', palette='Set2')
plt.title('Model Comparison - Classification Metrics')
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ==================================================


# Graph: Validation accuracy curves overlaid for all three models
plt.figure(figsize=(8, 5))
plt.plot(history_1.history['val_accuracy'], label='Model 1: Baseline CNN')
plt.plot(history_2.history['val_accuracy'], label='Model 2: Deep BatchNorm CNN')
plt.plot(history_3.history['val_accuracy'], label='Model 3: Wide GAP CNN')
plt.title('Validation Accuracy Comparison Across Models')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()


# ==================================================


# Graph: ROC curves for all three models
fpr_1, tpr_1, _ = roc_curve(y_test, model_1_probs)
fpr_2, tpr_2, _ = roc_curve(y_test, model_2_probs)
fpr_3, tpr_3, _ = roc_curve(y_test, model_3_probs)

plt.figure(figsize=(6, 6))
plt.plot(fpr_1, tpr_1, label='Model 1 (AUC = %.3f)' % auc(fpr_1, tpr_1))
plt.plot(fpr_2, tpr_2, label='Model 2 (AUC = %.3f)' % auc(fpr_2, tpr_2))
plt.plot(fpr_3, tpr_3, label='Model 3 (AUC = %.3f)' % auc(fpr_3, tpr_3))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ==================================================


# Identify the best-performing model based on test accuracy
best_row = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
print("Best performing model:", best_row['Model'])
print("Test Accuracy:", round(best_row['Test Accuracy'], 4))
print("F1-score:", round(best_row['F1-score'], 4))


# ==================================================


num_samples = 5

# Get random indices from the test set
random_indices = np.random.choice(len(X_test), num_samples, replace=False)

# Retrieve sample images and their true labels
sample_images = X_test_raw[random_indices] # Use raw images for display
sample_true_labels = y_test[random_indices]

# Make predictions using the best model (Model 1)
sample_predictions_probs = model_1.predict(X_test[random_indices])
sample_predictions = (sample_predictions_probs > 0.5).astype('int32')

# Plotting the sample images with predictions
plt.figure(figsize=(15, 6))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(sample_images[i].astype('uint8'))
    true_label_text = 'Real' if sample_true_labels[i] == 1 else 'Fake'
    predicted_label_text = 'Real' if sample_predictions[i] == 1 else 'Fake'

    plt.title(f"True: {true_label_text}\nPred: {predicted_label_text}")
    plt.axis('off')

plt.suptitle('Model 1 Predictions on Sample Test Images (Actual vs. Predicted)', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


# ==================================================


# Save the best-performing model (based on test accuracy) for inference
best_model_name = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
best_model = {'Model 1: Baseline CNN': model_1,
              'Model 2: Deep BatchNorm CNN': model_2,
              'Model 3: Wide GAP CNN': model_3}[best_model_name]

best_model.save('deepfake_model.keras')
print("Saved:", best_model_name, "-> deepfake_model.keras")


# ==================================================


%%writefile app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 64

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('deepfake_model.keras')

model = load_model()

st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️")
st.title("🕵️ Deepfake vs Real Image Classifier")
st.write("Upload an image to check whether it's Real or Fake (deepfake).")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    label = "Real" if prob > 0.5 else "Fake"
    confidence = prob if prob > 0.5 else 1 - prob

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence*100:.2f}%")
    st.progress(float(confidence))


# ==================================================


!pip install -q streamlit pyngrok
!pip install -q streamlit


# ==================================================


from pyngrok import ngrok
import subprocess, time

NGROK_TOKEN = "3Cg3o4xJLJJqj0JVoVnWSruazo0_4PkG5HLgHT5HVhtz7UFmw"
ngrok.set_auth_token(NGROK_TOKEN)

subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', '8501', '--server.headless', 'true'])
time.sleep(5)

public_url = ngrok.connect(8501)
print("Streamlit app link:", public_url)


# ==================================================

