"""
DeepFER — Live Webcam Emotion Recognition (local, standalone script)
======================================================================

Run this on your own machine (NOT in Colab) with a physical webcam attached:

    pip install opencv-python tensorflow numpy
    python webcam_demo_local.py

Before running, put the saved model file next to this script (or update
MODEL_PATH below). It's the file produced by the "Model Saving" section
of the DeepFER notebook:

    deepfer_efficientnetb0_best.keras

Press 'q' in the video window to quit.

IMPORTANT: preprocessing must match training exactly
------------------------------------------------------
The model was trained on face crops resized to 96x96 and passed through
tf.keras.applications.efficientnet.preprocess_input(). Any mismatch here
(e.g. a plain /255.0 rescale) will make predictions look wrong/random even
though the model itself is fine -- this was the root cause of the
"neutral for a happy face" issue in the original webcam_demo.py.
"""

import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

MODEL_PATH = "deepfer_efficientnetb0_best.keras"
IMG_SIZE = 96
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CAMERA_INDEX = 0


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find '{MODEL_PATH}'. Download it from the Colab notebook "
            "(produced in the 'Model Saving' section) and place it next to this script, "
            "or update MODEL_PATH above."
        )
    print(f"Loading model from {MODEL_PATH} ...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
    return model


def preprocess_face(face_bgr):
    """Convert a detected face crop (BGR, any size) into the model's expected input."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    return effnet_preprocess(face_resized[np.newaxis, ...])


def main():
    model = load_model()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Make sure a camera is connected and not in use by "
            "another application, and try a different CAMERA_INDEX (0, 1, ...)."
        )

    print("Webcam started. Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                face_crop = frame[y:y + h, x:x + w]
                face_input = preprocess_face(face_crop)
                probs = model.predict(face_input, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                label = EMOTION_CLASSES[pred_idx]
                confidence = float(probs[pred_idx])

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                text = f"{label} ({confidence * 100:.0f}%)"
                cv2.putText(
                    frame, text, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA,
                )

            cv2.imshow("DeepFER - Live Webcam Emotion Recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
