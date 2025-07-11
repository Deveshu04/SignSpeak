#!/usr/bin/env python3
"""
Convert Kaggle H5 model to TensorFlow.js format for web deployment
"""

import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path

def convert_model_to_tfjs():
    """Convert the Kaggle-trained H5 model to TensorFlow.js format"""

    # Paths
    h5_model_path = "./Models/asl_model.h5"
    tfjs_output_path = "./public/Models/tfjs_model"

    try:
        print("🔄 Loading H5 model...")
        model = tf.keras.models.load_model(h5_model_path)

        print("📊 Model summary:")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total parameters: {model.count_params():,}")

        print("🔄 Converting to TensorFlow.js format...")

        # Create output directory
        Path(tfjs_output_path).mkdir(parents=True, exist_ok=True)

        # Convert model
        tfjs.converters.save_keras_model(
            model,
            tfjs_output_path,
            quantize_float16=True,  # Reduce model size
            split_weights_by_layer=True  # Better loading performance
        )

        print(f"✅ Model converted successfully!")
        print(f"   Output location: {tfjs_output_path}")
        print(f"   You can now use this model in your React app")

        # Update the model path in TypeScript
        print("\n💡 Update your modelLoader.ts to use:")
        print(f"   modelPath = '/Models/tfjs_model/model.json'")

        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\n💡 Make sure you have tensorflowjs installed:")
        print("   pip install tensorflowjs")
        return False

def install_dependencies():
    """Install required dependencies for conversion"""
    import subprocess
    import sys

    try:
        print("📦 Installing tensorflowjs...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflowjs"])
        print("✅ Dependencies installed")
        return True
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

if __name__ == "__main__":
    print("🚀 SignSpeak Model Converter")
    print("=" * 50)

    success = convert_model_to_tfjs()

    if success:
        print("\n🎉 Conversion complete!")
        print("📝 Next steps:")
        print("   1. The TensorFlow.js model is now in public/Models/tfjs_model/")
        print("   2. Your React app can load it from '/Models/tfjs_model/model.json'")
        print("   3. Test the web app: npm run dev")
    else:
        print("\n❌ Conversion failed. Please check the errors above.")
