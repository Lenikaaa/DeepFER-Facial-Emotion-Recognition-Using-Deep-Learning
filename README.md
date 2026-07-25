# DeepFER: Facial Emotion Recognition Using Deep Learning

## Overview

DeepFER (Deep Facial Emotion Recognition) is a deep learning-based computer vision project that automatically recognizes human emotions from facial expressions. The system classifies facial images into seven emotion categories using Convolutional Neural Networks (CNNs) and Transfer Learning techniques.

Three different deep learning models were developed and evaluated:

- Custom CNN (4 Convolutional Blocks)
- VGG16 Transfer Learning
- EfficientNetB0 Transfer Learning (Best Performing Model)

After comprehensive evaluation, EfficientNetB0 achieved the best overall performance and was selected as the final model.

---

## Problem Statement

Understanding human emotions through facial expressions plays a vital role in developing intelligent and empathetic AI systems. Manual emotion analysis is time-consuming and subjective, making automated facial emotion recognition increasingly valuable.

This project aims to build an accurate and efficient facial emotion recognition system capable of identifying emotions from facial images for real-world applications.

---

## Objectives

- Develop an end-to-end facial emotion recognition system.
- Build and compare multiple deep learning models.
- Improve model performance using Transfer Learning.
- Evaluate models using standard classification metrics.
- Save the best-performing model for deployment.
- Demonstrate real-time emotion prediction capability.

---

## Dataset

The dataset contains facial images belonging to seven emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

### Data Preprocessing

The preprocessing pipeline includes:

- Image resizing
- Pixel normalization
- Data augmentation
- Training, validation, and test splitting

Data augmentation techniques include:

- Rotation
- Horizontal flipping
- Zooming
- Width shifting
- Height shifting

These techniques improve model generalization and reduce overfitting.

---

## Models Implemented

### 1. Custom CNN

A CNN architecture built from scratch consisting of four convolutional blocks followed by pooling, dropout, and dense layers.

### 2. VGG16 Transfer Learning

A pre-trained VGG16 model was fine-tuned for facial emotion recognition by replacing the classification head.

### 3. EfficientNetB0 Transfer Learning

EfficientNetB0 was implemented using transfer learning and achieved the best overall performance. It provided superior feature extraction, better generalization, and improved classification accuracy.

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

The notebook also includes:

- Confusion Matrix Interpretation
- Model Comparison
- Error Analysis

---

## Best Model

**EfficientNetB0 Transfer Learning**

Reasons for selection:

- Highest classification accuracy
- Better feature extraction
- Strong generalization capability
- Efficient computational performance

The trained model is saved and reloaded successfully for deployment readiness and sanity checking on unseen images.

---

## Real-Time Emotion Recognition Pipeline

The project also demonstrates a real-time inference workflow:

```
Webcam
    │
    ▼
Frame Capture
    │
    ▼
Face Detection
    │
    ▼
Image Preprocessing
    │
    ▼
EfficientNetB0 Prediction
    │
    ▼
Emotion Classification
```

This workflow can be integrated into webcam-based applications using OpenCV.

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## Project Structure

```
DeepFER/
│
├── DeepFER.ipynb
├── README.md
├── images/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Lenikaaa/DeepFER-Facial-Emotion-Recognition-Using-Deep-Learning.git
```

Navigate to the project folder:

```bash
cd DeepFER-Facial-Emotion-Recognition-Using-Deep-Learning
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the notebook using Jupyter Notebook or Google Colab.

---

## Results

- Successfully developed an end-to-end facial emotion recognition system.
- Compared three deep learning architectures.
- EfficientNetB0 achieved the best overall performance.
- The final model was saved, reloaded, and validated on unseen data.
- The project demonstrates strong potential for real-time emotion recognition applications.

---

## Future Improvements

Possible enhancements include:

- Larger and more diverse datasets
- Vision Transformers (ViT)
- EfficientNetV2
- Mobile and edge-device deployment
- Real-time video optimization
- Improved robustness under varying lighting conditions

---

## Applications

DeepFER can be applied in:

- Human-Computer Interaction
- Mental Health Monitoring
- Customer Experience Analytics
- Smart Education
- Driver Monitoring Systems
- Healthcare Assistance
- Security and Surveillance

---

## Author

**Lenika Yogi**

M.Sc. Artificial Intelligence & Machine Learning

GitHub:
https://github.com/Lenikaaa

---

## License

This project is developed for educational and research purposes as part of a Deep Learning Capstone Project.
