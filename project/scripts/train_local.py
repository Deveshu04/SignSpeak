#!/usr/bin/env python3
"""
SignSpeak ASL Model Training Script
Trains a CNN model on the ASL dataset for real-time sign recognition
Can be run locally or adapted for Kaggle
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import json
from pathlib import Path
import zipfile
from datetime import datetime
import sys

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Configuration
IMG_SIZE = 64  # Much smaller to reduce complexity further
BATCH_SIZE = 32  # Increased batch size for more stable gradients
EPOCHS = 100  # More epochs with simpler model
LEARNING_RATE = 0.001  # Higher learning rate to start
VALIDATION_SPLIT = 0.2

# Dataset path configuration - automatically detect environment
if '/kaggle/working' in sys.path or 'kaggle' in os.getcwd().lower():
    # Running on Kaggle
    DATASET_PATH = '/kaggle/input'  # Will auto-detect the actual dataset
    OUTPUT_PATH = '/kaggle/working'
    print("🔧 Detected Kaggle environment")
else:
    # Running locally
    DATASET_PATH = './Data/asl_dataset'
    OUTPUT_PATH = './public/dataset/models'
    print("🔧 Detected local environment")

class ASLModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

        # Auto-detect classes from dataset
        self.classes = self.detect_available_classes()
        self.num_classes = len(self.classes)

        print(f"🎯 ASL Model Trainer Initialized")
        print(f"   Dataset path: {self.dataset_path}")
        print(f"   Classes found: {self.num_classes}")
        print(f"   Classes: {self.classes}")

    def detect_available_classes(self):
        """Auto-detect available classes from the dataset directory"""
        classes = []

        # Special handling for Kaggle environment
        if '/kaggle/input' in str(self.dataset_path):
            print("🔍 Scanning Kaggle input datasets...")

            # List all available datasets in /kaggle/input
            input_dir = Path('/kaggle/input')
            if input_dir.exists():
                available_datasets = [d.name for d in input_dir.iterdir() if d.is_dir()]
                print(f"   Available datasets: {available_datasets}")

                # Try to find the ASL dataset
                for dataset_name in available_datasets:
                    dataset_dir = input_dir / dataset_name

                    # Look for class directories directly in dataset
                    potential_classes = []
                    for item in dataset_dir.iterdir():
                        if item.is_dir():
                            # Check if it's a class directory (has images)
                            image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + \
                                         list(item.glob("*.png")) + list(item.glob("*.webp"))
                            if len(image_files) > 0:
                                potential_classes.append(item.name)

                    if potential_classes:
                        print(f"   Found classes in dataset '{dataset_name}': {len(potential_classes)} classes")
                        self.dataset_path = dataset_dir
                        classes = potential_classes
                        break

                    # Also check for nested asl_dataset folder
                    asl_subdir = dataset_dir / 'asl_dataset'
                    if asl_subdir.exists():
                        for item in asl_subdir.iterdir():
                            if item.is_dir():
                                image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + \
                                             list(item.glob("*.png")) + list(item.glob("*.webp"))
                                if len(image_files) > 0:
                                    potential_classes.append(item.name)

                        if potential_classes:
                            print(f"   Found classes in nested folder '{dataset_name}/asl_dataset': {len(potential_classes)} classes")
                            self.dataset_path = asl_subdir
                            classes = potential_classes
                            break

                if not classes:
                    print("   ❌ No ASL classes found in any dataset!")
                    print("   📋 Make sure your dataset contains directories named with letters/numbers")

            return sorted(classes)

        # Original logic for local datasets
        search_paths = [self.dataset_path]

        # Check if there's an asl_dataset subdirectory
        asl_subdir = self.dataset_path / 'asl_dataset'
        if asl_subdir.exists() and asl_subdir.is_dir():
            search_paths.append(asl_subdir)

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for item in search_path.iterdir():
                if item.is_dir() and item.name != 'asl_dataset':  # Avoid nested asl_dataset
                    # Check if directory has image files
                    image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + \
                                 list(item.glob("*.png")) + list(item.glob("*.webp"))
                    if len(image_files) > 0:
                        classes.append(item.name)

            # If we found classes, use this path
            if classes:
                self.dataset_path = search_path
                break

        return sorted(classes)

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("📂 Loading and preprocessing data...")

        images = []
        labels = []
        class_counts = {}

        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.dataset_path / class_name

            if not class_dir.exists():
                print(f"⚠️ Warning: Class directory '{class_name}' not found")
                continue

            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png")) + list(class_dir.glob("*.webp"))

            class_counts[class_name] = len(image_files)

            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"⚠️ Could not load image: {img_path}")
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img.astype(np.float32) / 255.0

                    images.append(img)
                    labels.append(class_idx)

                except Exception as e:
                    print(f"❌ Error loading {img_path}: {e}")
                    continue

        print(f"✅ Loaded {len(images)} images from {len(class_counts)} classes")

        # Print class distribution
        print("\n📊 Class distribution:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} images")

        if len(images) == 0:
            raise ValueError("No images were loaded! Please check your dataset path and structure.")

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, self.num_classes)

        return X, y_categorical, class_counts

    def create_model(self):
        """Create a simple but effective CNN model for ASL recognition"""
        print("🏗️ Creating simple and effective model architecture...")

        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

            # Simple CNN blocks - much lighter
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Simple optimizer without learning rate scheduling
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("✅ Simple model created successfully")
        print(f"   Total parameters: {model.count_params():,}")

        return model

    def train_model(self):
        """Complete training pipeline"""
        print("🚀 Starting complete training pipeline...")

        # Load data
        X, y, class_counts = self.load_and_preprocess_data()
        print(f"\nDataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y.argmax(axis=1)
        )

        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")

        # Create data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip for sign language
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        # Create model
        model = self.create_model()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
        ]

        # Train model
        print("🚀 Starting training...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        print("📊 Evaluating model...")
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

        print(f"\n🎯 Final Results:")
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")

        # Generate predictions for classification report
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        # Calculate top-3 accuracy manually
        top3_predictions = np.argsort(y_pred, axis=1)[:, -3:]
        val_top3_accuracy = np.mean([y_true_classes[i] in top3_predictions[i] for i in range(len(y_true_classes))])

        print(f"   Top-3 Accuracy: {val_top3_accuracy:.4f}")

        print("\n📋 Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=self.classes))

        # Save models
        self.save_models(model, history, val_accuracy, val_loss, val_top3_accuracy, class_counts)

        return model, history

    def save_models(self, model, history, val_accuracy, val_loss, val_top3_accuracy, class_counts):
        """Save models in multiple formats"""
        output_dir = Path(OUTPUT_PATH)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("💾 Saving models...")

        # 1. Save as H5 format
        model.save(str(output_dir / 'asl_model.h5'))
        print("✅ H5 model saved")

        # 2. Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(output_dir / 'asl_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("✅ TFLite model saved")

        # 3. Save model metadata
        metadata = {
            'model_info': {
                'name': 'SignSpeak ASL Classifier',
                'version': '1.0.0',
                'created_date': datetime.now().isoformat(),
                'input_shape': [IMG_SIZE, IMG_SIZE, 3],
                'num_classes': self.num_classes,
                'classes': self.classes
            },
            'training_info': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'validation_split': VALIDATION_SPLIT,
                'total_images': len(class_counts),
                'class_distribution': class_counts
            },
            'performance': {
                'validation_accuracy': float(val_accuracy),
                'validation_loss': float(val_loss),
                'top3_accuracy': float(val_top3_accuracy)
            }
        }

        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # 4. Create inference config
        config = {
            "model_path": "asl_model.h5",
            "tflite_path": "asl_model.tflite",
            "input_size": [IMG_SIZE, IMG_SIZE],
            "classes": self.classes,
            "confidence_threshold": 0.5,
            "preprocessing": {
                "resize": [IMG_SIZE, IMG_SIZE],
                "normalize": True,
                "color_mode": "RGB"
            }
        }

        with open(output_dir / 'inference_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print("✅ Model metadata and config saved")
        print(f"\n📁 Files saved to: {output_dir}")

        # Plot training history
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Accuracy
            ax1.plot(history.history['accuracy'], label='Training Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)

            # Loss
            ax2.plot(history.history['loss'], label='Training Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
            print("✅ Training history plot saved")

        except Exception as e:
            print(f"⚠️ Could not save training plot: {e}")

def main():
    """Main training function"""
    trainer = ASLModelTrainer(DATASET_PATH)

    if len(trainer.classes) == 0:
        print("❌ No classes found in dataset!")
        print(f"   Check that {DATASET_PATH} contains subdirectories with images")
        return

    try:
        model, history = trainer.train_model()
        print("\n🎉 Training completed successfully!")
        print(f"   Model saved to: {OUTPUT_PATH}")
        print(f"   You can now test inference with: python scripts/inference_simple.py")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
