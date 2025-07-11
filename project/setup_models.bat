@echo off
echo 🚀 SignSpeak Model Setup Script
echo ================================

cd /d "E:\SignSpeak\project"

echo 📦 Activating virtual environment...
call venv310\Scripts\activate.bat

echo 📦 Installing TensorFlow...
pip install tensorflow

echo 📦 Installing additional dependencies...
pip install opencv-python
pip install scikit-learn

echo 🧪 Testing model setup...
python scripts\test_model.py

echo.
echo ✅ Setup complete!
echo 💡 You can now run:
echo    - python scripts\inference_simple.py (for webcam testing)
echo    - python scripts\convert_model_to_tfjs.py (for web deployment)

pause
