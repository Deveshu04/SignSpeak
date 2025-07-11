#!/usr/bin/env python3
"""
ML Model Training Script for SignSpeak
Trains a CNN model on the ASL dataset for real-time sign recognition
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from datetime import datetime

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

class ASLModelTrainer:
    def __init__(self, dataset_path="project/Data"):
        self.dataset_path = Path(dataset_path)
        self.asl_path = self.dataset_path / "asl_dataset"
        # Save models to public directory for web accessibility
        self.model_path = Path("./public/dataset/models")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Classes: lowercase letters a-z and numbers 0-9
        self.classes = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [str(i) for i in range(10)]
        self.num_classes = len(self.classes)
        
        print(f"🎯 Initializing ASL Model Trainer")
        print(f"   Dataset path: {self.dataset_path}")
        print(f"   Model save path: {self.model_path}")
        print(f"   Expected classes: {self.num_classes}")
        print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("📂 Loading and preprocessing data...")
        
        images = []
        labels = []
        class_counts = {}
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.asl_path / class_name
            
            if not class_dir.exists():
                print(f"⚠️ Warning: Class directory '{class_name}' not found")
                continue
            
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png")) + list(class_dir.glob("*.webp"))
            
            class_counts[class_name] = len(image_files)
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
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
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        return X, y_categorical, class_counts
    
    def create_model(self):
        """Create the CNN model architecture"""
        print("🏗️ Creating model architecture...")
        
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print("✅ Model created successfully")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, X, y, class_counts):
        """Train the model"""
        print("🚀 Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y.argmax(axis=1)
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Create model
        model = self.create_model()
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.model_path / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        # Save final model as .h5
        model.save(str(self.model_path / 'final_model.h5'))
        print(f"✅ Final model saved to {self.model_path / 'final_model.h5'}")
        
        # Convert and save as TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = self.model_path / 'final_model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite model saved to {tflite_path}")
        
        return model, history, (X_val, y_val)
    
    def save_model_for_web(self, model, metrics, class_counts):
        """Save model in TensorFlow.js format for web deployment"""
        print("💾 Saving model for web deployment...")
        
        # Save in TensorFlow.js format to public directory
        tfjs_path = self.model_path / "tfjs_model"
        tfjs_path.mkdir(exist_ok=True)
        
        # Convert to TensorFlow.js format
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, str(tfjs_path))
        
        # Save model metadata to public directory
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
                'total_images': sum(class_counts.values()),
                'class_distribution': class_counts
            },
            'performance': metrics,
            'preprocessing': {
                'image_size': IMG_SIZE,
                'normalization': 'divide_by_255',
                'data_augmentation': True
            }
        }
        
        with open(self.model_path / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to: {tfjs_path}")
        print(f"✅ Metadata saved to: {self.model_path / 'model_metadata.json'}")
        
        return str(tfjs_path)
    
    def plot_training_history(self, history):
        """Plot training history"""
        print("📈 Generating training plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training plots saved to: {self.model_path / 'training_history.png'}")

def main():
    """Main training function"""
    print("🎯 SignSpeak ASL Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ASLModelTrainer()
    
    # Load and preprocess data
    X, y, class_counts = trainer.load_and_preprocess_data()
    
    if len(X) == 0:
        print("❌ No data found! Please check your dataset.")
        return
    
    # Train model
    model, history, metrics = trainer.train_model(X, y, class_counts)
    
    # Save model for web
    model_path = trainer.save_model_for_web(model, metrics, class_counts)
    
    # Plot training history
    trainer.plot_training_history(history)
    
    print("\n🎉 Training completed successfully!")
    print(f"   Model saved to: {model_path}")
    print(f"   Validation accuracy: {metrics[1][1]:.4f}")
    print("\n📋 Next steps:")
    print("1. Test the model in the SignSpeak app")
    print("2. Collect more data for classes with low accuracy")
    print("3. Fine-tune hyperparameters if needed")

if __name__ == "__main__":
    main()