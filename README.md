# 😊 DeepFER: Facial Emotion Recognition Using Deep Learning

DeepFER is a Computer Vision and Deep Learning project that recognizes human facial emotions from images and live webcam input. The project leverages Convolutional Neural Networks (CNNs) and Transfer Learning techniques to classify facial expressions into seven different emotion categories.

The project compares multiple deep learning architectures and selects the best-performing model based on evaluation metrics. It also includes real-time emotion recognition using a webcam, making it suitable for practical AI applications.

---

# 📌 Project Overview

Facial emotion recognition is an important computer vision task with applications in:

- Human-Computer Interaction
- Mental Health Monitoring
- Customer Service Analytics
- Smart Education Systems
- Driver Monitoring Systems
- Healthcare Assistance
- AI-powered Interactive Applications

This project develops a complete facial emotion recognition pipeline, starting from data preprocessing and model training to real-time deployment.

---

# 🎯 Objectives

- Build an accurate facial emotion recognition system.
- Compare multiple deep learning architectures.
- Apply Transfer Learning for improved performance.
- Perform image preprocessing and augmentation.
- Evaluate models using multiple performance metrics.
- Save the best-performing model.
- Deploy the model for real-time webcam emotion recognition.

---

# 😀 Emotion Classes

The model predicts the following seven facial emotions:

- Angry 😠
- Disgust 🤢
- Fear 😨
- Happy 😊
- Neutral 😐
- Sad 😢
- Surprise 😲

---

# 🧠 Models Implemented

Three different deep learning models were developed and compared.

| Model | Description |
|--------|-------------|
| Custom CNN | Four Convolutional Blocks built from scratch |
| VGG16 Transfer Learning | Transfer Learning using ImageNet pretrained weights |
| EfficientNetB0 Transfer Learning | Fine-tuned EfficientNetB0 (Best Performing Model) |

After comparison, **EfficientNetB0** was selected as the final deployment model.

---

# ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Google Colab

---

# 📂 Project Structure

```
DeepFER/

│── DeepFER_fixed.ipynb
│── webcam_demo_colab_fixed.ipynb
│── webcam_demo_local.py
│── README.md
│
├── images/
   ├── train/
   ├── validation/
   └── test/

```

---

# 🔄 Project Workflow

```
Dataset
      │
      ▼
Image Preprocessing
      │
      ▼
Data Augmentation
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Model Saving
      │
      ▼
Real-Time Emotion Recognition
```

---

# 🖼 Dataset

The dataset consists of facial images categorized into seven emotion classes.

### Image Processing

- Face Images
- RGB Images
- Image Size: 96 × 96
- Data Augmentation
- Normalization
- Train / Validation Split

Augmentation techniques include:

- Rotation
- Horizontal Flip
- Zoom
- Width Shift
- Height Shift
- Brightness Adjustment

---

# 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Model comparison was performed to identify the best-performing architecture.

---

# 💾 Model Saving

The final EfficientNetB0 model is saved as:

```
deepfer_efficientnetb0_best.keras
```

The saved model can be loaded later for inference without retraining.

---

# 🎯 Prediction

The project supports prediction on:

- Validation Images
- Test Images
- Single Custom Images
- Live Webcam

---

# 📷 Local Webcam Demo

The repository includes a standalone webcam application for real-time emotion recognition.

Run locally:

```bash
pip install tensorflow opencv-python numpy

python webcam_demo_local.py
```

Requirements:

- Place `deepfer_efficientnetb0_best.keras` in the same folder as the script.
- Connect a webcam.
- Press **Q** to quit the application.

The webcam application:

- Detects faces
- Crops the detected face
- Resizes images to **96×96**
- Applies the same preprocessing used during training
- Predicts one of the seven emotions
- Displays confidence scores in real time

---

# ☁️ Google Colab Webcam Demo

The project also includes:

```
webcam_demo_colab_fixed.ipynb
```

This notebook allows users to:

- Capture an image directly from the browser camera
- Load the trained model
- Predict facial emotion inside Google Colab

---

# 🚀 Features

- Facial Emotion Recognition
- Seven Emotion Classification
- Custom CNN
- VGG16 Transfer Learning
- EfficientNetB0 Transfer Learning
- Data Augmentation
- Model Comparison
- Confusion Matrix
- Classification Report
- Model Saving
- Prediction on New Images
- Local Webcam Emotion Recognition
- Google Colab Webcam Inference

---

# 📊 Applications

- Human Computer Interaction
- Mental Health Monitoring
- Customer Experience Analysis
- Smart Learning Platforms
- Healthcare
- Driver Fatigue Monitoring
- Security Systems
- Interactive AI Applications

---

# 🔮 Future Improvements

- Train on larger datasets
- Improve robustness under varying lighting conditions
- Support multiple face detection
- Deploy using Streamlit or Flask
- TensorFlow Lite optimization
- Mobile deployment
- Edge AI deployment

---

# ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/Lenikaaa/DeepFER-Facial-Emotion-Recognition-Using-Deep-Learning
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run Notebook

Open:

```
DeepFER_fixed.ipynb
```

Run all cells sequentially.

---

### Run Local Webcam

```bash
python webcam_demo_local.py
```

---

# 📷 Sample Output

The model predicts one of the following emotions:

```
😊 Happy

😐 Neutral

😢 Sad

😠 Angry

😲 Surprise

😨 Fear

🤢 Disgust
```

along with the prediction confidence.

---

# 📚 Learning Outcomes

This project demonstrates practical implementation of:

- Computer Vision
- Deep Learning
- Convolutional Neural Networks
- Transfer Learning
- EfficientNetB0
- VGG16
- Image Classification
- Model Evaluation
- Real-Time AI Inference
- Deep Learning Deployment

---

# 👩‍💻 Author

**Lenika Yogi**

M.Sc. Artificial Intelligence & Machine Learning

Deep Learning | Computer Vision | Machine Learning | Data Science

---

# ⭐ Acknowledgements

- TensorFlow
- Keras
- OpenCV
- Scikit-learn
- Google Colab
- ImageNet
- EfficientNet
- VGG16
