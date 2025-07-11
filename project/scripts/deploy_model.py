#!/usr/bin/env python3
"""
Model Deployment Script for SignSpeak
Downloads and sets up trained models from Kaggle for local inference
"""

import os
import json
import zipfile
import requests
from pathlib import Path
import shutil

class ModelDeployer:
    def __init__(self):
        self.models_dir = Path("./public/dataset/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        print("🚀 SignSpeak Model Deployer")
        print(f"   Models directory: {self.models_dir}")

    def extract_kaggle_model(self, zip_path):
        """Extract model package downloaded from Kaggle"""
        zip_path = Path(zip_path)

        if not zip_path.exists():
            print(f"❌ Model package not found: {zip_path}")
            return False

        print(f"📦 Extracting model package: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Extract all files to models directory
                zipf.extractall(self.models_dir)

            print("✅ Model package extracted successfully")

            # Verify extracted files
            self.verify_model_files()
            return True

        except Exception as e:
            print(f"❌ Error extracting model package: {e}")
            return False

    def verify_model_files(self):
        """Verify that all required model files are present"""
        print("\n🔍 Verifying model files...")

        required_files = [
            "asl_model.h5",
            "asl_model.tflite",
            "model_metadata.json"
        ]

        missing_files = []
        present_files = []

        for file in required_files:
            file_path = self.models_dir / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / 1024 / 1024
                present_files.append(f"   ✅ {file} ({size_mb:.1f} MB)")
            else:
                missing_files.append(f"   ❌ {file}")

        # Check for TensorFlow.js model
        tfjs_dir = self.models_dir / "tfjs_model"
        if tfjs_dir.exists():
            present_files.append(f"   ✅ tfjs_model/ (directory)")

        print("📁 Model Files Status:")
        for file in present_files:
            print(file)

        if missing_files:
            print("\n⚠️ Missing files:")
            for file in missing_files:
                print(file)
        else:
            print("\n🎉 All required model files are present!")

    def load_model_metadata(self):
        """Load and display model metadata"""
        metadata_path = self.models_dir / "model_metadata.json"

        if not metadata_path.exists():
            print("❌ Model metadata not found")
            return None

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            print("\n📊 Model Information:")
            print(f"   Name: {metadata['model_info']['name']}")
            print(f"   Version: {metadata['model_info']['version']}")
            print(f"   Created: {metadata['model_info']['created_date']}")
            print(f"   Classes: {metadata['model_info']['num_classes']}")
            print(f"   Input Shape: {metadata['model_info']['input_shape']}")

            print("\n🎯 Performance:")
            print(f"   Accuracy: {metadata['performance']['validation_accuracy']:.3f}")
            print(f"   Top-3 Accuracy: {metadata['performance']['top3_accuracy']:.3f}")

            print("\n🏷️ Supported Signs:")
            classes = metadata['model_info']['classes']
            # Print classes in rows of 6 for better readability
            for i in range(0, len(classes), 6):
                row = classes[i:i+6]
                print(f"   {' '.join(f'{c:>2}' for c in row)}")

            return metadata

        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None

    def create_inference_config(self, metadata=None):
        """Create configuration file for inference"""
        if metadata is None:
            metadata = self.load_model_metadata()

        if metadata is None:
            print("❌ Cannot create inference config without metadata")
            return

        config = {
            "model_path": "asl_model.h5",
            "tflite_path": "asl_model.tflite",
            "tfjs_path": "tfjs_model",
            "input_size": metadata['model_info']['input_shape'][:2],  # [224, 224]
            "classes": metadata['model_info']['classes'],
            "confidence_threshold": 0.5,
            "preprocessing": {
                "resize": metadata['model_info']['input_shape'][:2],
                "normalize": True,
                "color_mode": "RGB"
            }
        }

        config_path = self.models_dir / "inference_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Inference config created: {config_path}")

    def test_model_loading(self):
        """Test if the model can be loaded successfully"""
        model_path = self.models_dir / "asl_model.h5"

        if not model_path.exists():
            print("❌ Model file not found for testing")
            return False

        try:
            import tensorflow as tf
            print("\n🧪 Testing model loading...")

            model = tf.keras.models.load_model(str(model_path))
            print(f"✅ Model loaded successfully")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Parameters: {model.count_params():,}")

            return True

        except ImportError:
            print("⚠️ TensorFlow not available for testing")
            return True  # Still considered success
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    deployer = ModelDeployer()

    print("\n" + "="*50)
    print("SignSpeak Model Deployment")
    print("="*50)

    while True:
        print("\nOptions:")
        print("1. Extract model package from Kaggle")
        print("2. Verify existing model files")
        print("3. Show model information")
        print("4. Test model loading")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            zip_path = input("Enter path to model package (.zip file): ").strip()
            if deployer.extract_kaggle_model(zip_path):
                deployer.create_inference_config()

        elif choice == '2':
            deployer.verify_model_files()

        elif choice == '3':
            deployer.load_model_metadata()

        elif choice == '4':
            deployer.test_model_loading()

        elif choice == '5':
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
