#!/usr/bin/env python3
"""
Improved Inference Script for SignSpeak
Performs real-time ASL prediction using Kaggle-trained models
"""

import cv2
import numpy as np
import json
from pathlib import Path
import time
import os

class ASLPredictor:
    def __init__(self, models_dir=None):
        # Auto-detect model directory
        if models_dir is None:
            # Try multiple possible locations
            possible_dirs = [
                "./Models",
                "./public/dataset/models",
                "../Models",
                "../public/dataset/models"
            ]

            for dir_path in possible_dirs:
                if Path(dir_path).exists() and (Path(dir_path) / "asl_model.h5").exists():
                    models_dir = dir_path
                    break

            if models_dir is None:
                raise FileNotFoundError("Could not find model files in any expected directory")

        self.models_dir = Path(models_dir)
        self.model = None
        self.classes = []
        self.input_size = (128, 128)  # Updated for Kaggle model
        self.confidence_threshold = 0.5

        print(f"🔍 Using models from: {self.models_dir}")
        self.load_config()
        self.load_model()

    def load_config(self):
        """Load inference configuration"""
        config_path = self.models_dir / "inference_config.json"

        # Default classes (ASL alphabet + numbers)
        default_classes = [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z"
        ]

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                self.classes = config.get('classes', default_classes)
                input_size_config = config.get('input_size', [128, 128])
                self.input_size = tuple(input_size_config)
                self.confidence_threshold = config.get('confidence_threshold', 0.5)
                print(f"✅ Config loaded: {len(self.classes)} classes, input size: {self.input_size}")
            except Exception as e:
                print(f"⚠️ Error loading config: {e}, using defaults")
                self.classes = default_classes
        else:
            print("⚠️ No config found, using defaults")
            self.classes = default_classes

    def load_model(self):
        """Load the trained model"""
        h5_model_path = self.models_dir / "asl_model.h5"
        tflite_model_path = self.models_dir / "asl_model.tflite"

        # Try loading H5 model first
        if h5_model_path.exists():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(h5_model_path))
                self.model_type = "keras"
                print(f"✅ Keras model loaded successfully")
                print(f"   Input shape: {self.model.input_shape}")
                print(f"   Output shape: {self.model.output_shape}")
                return True
            except ImportError:
                print("❌ TensorFlow not available for H5 model")
            except Exception as e:
                print(f"❌ Error loading H5 model: {e}")

        # Try loading TFLite model as fallback
        if tflite_model_path.exists():
            try:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
                self.interpreter.allocate_tensors()
                self.model_type = "tflite"

                # Get input and output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

                print(f"✅ TFLite model loaded successfully")
                print(f"   Input shape: {self.input_details[0]['shape']}")
                print(f"   Output shape: {self.output_details[0]['shape']}")
                return True
            except Exception as e:
                print(f"❌ Error loading TFLite model: {e}")

        print("❌ No model could be loaded!")
        return False

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to model input size
        image_resized = cv2.resize(image, self.input_size)

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)

        return image_batch

    def predict(self, image):
        """Make prediction on preprocessed image"""
        if self.model is None and not hasattr(self, 'interpreter'):
            return None, 0.0

        # Preprocess
        processed_image = self.preprocess_image(image)

        try:
            if self.model_type == "keras":
                # Keras model prediction
                predictions = self.model.predict(processed_image, verbose=0)
            else:
                # TFLite model prediction
                self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])

            if predicted_class_idx < len(self.classes):
                predicted_class = self.classes[predicted_class_idx]
            else:
                predicted_class = f"unknown_{predicted_class_idx}"

            return predicted_class, confidence

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0.0

    def get_top_k_predictions(self, image, k=3):
        """Get top k predictions with confidence scores"""
        if self.model is None and not hasattr(self, 'interpreter'):
            return []

        processed_image = self.preprocess_image(image)

        try:
            if self.model_type == "keras":
                predictions = self.model.predict(processed_image, verbose=0)
            else:
                self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Get top k predictions
            top_k_indices = np.argsort(predictions[0])[-k:][::-1]
            top_k_predictions = []

            for idx in top_k_indices:
                if idx < len(self.classes):
                    class_name = self.classes[idx]
                    confidence = float(predictions[0][idx])
                    top_k_predictions.append((class_name, confidence))

            return top_k_predictions

        except Exception as e:
            print(f"❌ Top-k prediction error: {e}")
            return []

def test_webcam_inference():
    """Test real-time inference with webcam"""
    try:
        predictor = ASLPredictor()

        if predictor.model is None and not hasattr(predictor, 'interpreter'):
            print("❌ No model loaded, cannot start webcam test")
            return

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return

        print("🎥 Starting webcam inference...")
        print("   Press 'q' to quit")
        print("   Press 's' to save current frame")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Define ROI for hand detection (center square)
            h, w = frame.shape[:2]
            roi_size = min(h, w) // 2
            roi_x = (w - roi_size) // 2
            roi_y = (h - roi_size) // 2

            # Extract ROI
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]

            # Make prediction every few frames to improve performance
            if frame_count % 5 == 0:
                top_predictions = predictor.get_top_k_predictions(roi, k=3)

            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)

            # Display predictions
            y_offset = 30
            if 'top_predictions' in locals() and top_predictions:
                for i, (class_name, confidence) in enumerate(top_predictions):
                    text = f"{class_name}: {confidence:.2f}"
                    color = (0, 255, 0) if confidence > predictor.confidence_threshold else (0, 255, 255)
                    cv2.putText(frame, text, (10, y_offset + i*30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "No prediction", (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display instructions
            cv2.putText(frame, "Place hand in green box", (10, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, h-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('SignSpeak - ASL Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame and ROI
                timestamp = int(time.time())
                cv2.imwrite(f'captured_frame_{timestamp}.jpg', frame)
                cv2.imwrite(f'captured_roi_{timestamp}.jpg', roi)
                print(f"💾 Saved frame and ROI with timestamp {timestamp}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam inference session ended")

    except Exception as e:
        print(f"❌ Webcam test failed: {e}")

def test_single_image(image_path):
    """Test inference on a single image"""
    try:
        predictor = ASLPredictor()

        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return

        # Make prediction
        predicted_class, confidence = predictor.predict(image)
        top_predictions = predictor.get_top_k_predictions(image, k=5)

        print(f"\n🔍 Prediction for {image_path}:")
        print(f"   Top prediction: {predicted_class} (confidence: {confidence:.3f})")
        print(f"   Top 5 predictions:")
        for i, (class_name, conf) in enumerate(top_predictions, 1):
            print(f"     {i}. {class_name}: {conf:.3f}")

    except Exception as e:
        print(f"❌ Single image test failed: {e}")

if __name__ == "__main__":
    print("🚀 SignSpeak ASL Inference Script")
    print("   Using Kaggle-trained models")

    # Test model loading
    try:
        predictor = ASLPredictor()
        print(f"\n📊 Model Summary:")
        print(f"   Classes: {len(predictor.classes)}")
        print(f"   Input size: {predictor.input_size}")
        print(f"   Model type: {getattr(predictor, 'model_type', 'Unknown')}")

        # Start webcam test
        test_webcam_inference()

    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\n💡 Make sure you have:")
        print("   - Model files (asl_model.h5 or asl_model.tflite) in Models/ directory")
        print("   - TensorFlow installed: pip install tensorflow")
        print("   - OpenCV installed: pip install opencv-python")
