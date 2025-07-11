#!/usr/bin/env python3
"""
Simple model test script to verify Kaggle-trained models work
"""

import sys
from pathlib import Path

def test_model_loading():
    """Test if the Kaggle-trained model can be loaded"""
    print("🧪 Testing Kaggle-trained model setup...")

    # Check if model files exist
    models_dir = Path("./Models")
    h5_model = models_dir / "asl_model.h5"
    tflite_model = models_dir / "asl_model.tflite"
    config_file = models_dir / "inference_config.json"

    print(f"📁 Checking model files in {models_dir}:")
    print(f"   H5 model: {'✅' if h5_model.exists() else '❌'} {h5_model}")
    print(f"   TFLite model: {'✅' if tflite_model.exists() else '❌'} {tflite_model}")
    print(f"   Config file: {'✅' if config_file.exists() else '❌'} {config_file}")

    if not h5_model.exists():
        print("❌ H5 model not found!")
        return False

    # Try to load TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")

        # Try to load the model
        print("🔄 Loading Kaggle-trained model...")
        model = tf.keras.models.load_model(str(h5_model))

        print("✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")

        # Test prediction with dummy data
        import numpy as np
        # Use the model's actual input shape instead of hardcoded 128x128
        input_shape = model.input_shape[1:3]  # Get height, width from model
        dummy_input = np.random.random((1, input_shape[0], input_shape[1], 3)).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)

        print(f"✅ Test prediction successful!")
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Top class index: {np.argmax(prediction[0])}")
        print(f"   Confidence: {np.max(prediction[0]):.3f}")

        return True

    except ImportError:
        print("❌ TensorFlow not available")
        print("💡 Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_config_loading():
    """Test loading the inference configuration"""
    import json

    try:
        config_path = Path("./Models/inference_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        print("✅ Configuration loaded:")
        print(f"   Classes: {len(config.get('classes', []))}")
        print(f"   Input size: {config.get('input_size', 'Unknown')}")
        print(f"   Confidence threshold: {config.get('confidence_threshold', 'Unknown')}")

        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

if __name__ == "__main__":
    print("🚀 SignSpeak Model Test Script")
    print("=" * 50)

    # Test model loading
    model_ok = test_model_loading()
    print()

    # Test config loading
    config_ok = test_config_loading()
    print()

    if model_ok and config_ok:
        print("🎉 All tests passed! Your Kaggle-trained model is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Run: python scripts/convert_model_to_tfjs.py (for web deployment)")
        print("   2. Test webcam: python scripts/inference_simple.py")
        print("   3. Your React app can now use the model!")
    else:
        print("❌ Some tests failed. Check the errors above.")
        if not model_ok:
            print("   - Make sure your Kaggle models are in the Models/ directory")
            print("   - Install TensorFlow: pip install tensorflow")
        if not config_ok:
            print("   - Check that inference_config.json exists and is valid")
