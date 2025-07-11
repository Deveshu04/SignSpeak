# SignSpeak - AI Sign Language Translator 🤟

A real-time AI-powered sign language to speech translation application built with React, TypeScript, and TensorFlow.js. This project uses a Kaggle-trained deep learning model to recognize American Sign Language (ASL) gestures and convert them to speech.

## 🎯 Features

- **Real-time Sign Recognition**: AI-powered detection of ASL letters and numbers (0-9, a-z)
- **Kaggle-Trained Model**: High-accuracy CNN model trained on ASL dataset
- **Multi-language Support**: ASL with full AI recognition, ISL and BSL support planned
- **Speech Synthesis**: Convert recognized signs to speech in multiple languages
- **Interactive Learning**: Tutor mode for learning sign language
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Processing**: Live webcam feed with 2-second prediction intervals

## 🏗️ Architecture

### Frontend (React + TypeScript)
- **React 18** with TypeScript for type safety
- **Vite** for fast development and building
- **Tailwind CSS** for responsive styling
- **TensorFlow.js** for client-side model inference
- **Lucide React** for icons

### AI Model Pipeline
- **Training**: Kaggle environment with TensorFlow/Keras
- **Model**: CNN architecture with 64x64 input resolution
- **Conversion**: Python script to convert H5 model to TensorFlow.js
- **Inference**: Real-time browser-based prediction

### Dataset
- **ASL Dataset**: Kaggle ASL alphabet and numbers dataset
- **Classes**: 36 total (0-9 digits, a-z letters)
- **Input Size**: 64x64 RGB images
- **Preprocessing**: Normalization and resizing

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+**
- **Python 3.8+** (for model training and conversion)
- **Webcam access**
- **Kaggle account** (for dataset access)

### Installation

1. **Clone and install dependencies**:
```bash
git clone <repository-url>
cd SignSpeak
cd project
npm install
```

2. **Set up Python environment**:
```bash
# Create virtual environment (from SignSpeak root)
python -m venv venv310

# Activate virtual environment
# Windows:
venv310\Scripts\activate
# macOS/Linux:
source venv310/bin/activate

# Install Python dependencies
pip install -r project/requirements.txt
```

3. **Set up Kaggle API** (for dataset access):
```bash
# Install Kaggle CLI
pip install kaggle

# Get API credentials from https://www.kaggle.com/settings/account
# Place kaggle.json in ~/.kaggle/ (Linux/macOS) or C:\Users\<username>\.kaggle\ (Windows)

# Download ASL dataset
kaggle datasets download -d grassknoted/asl-alphabet
```

4. **Convert your Kaggle model**:
```bash
# Navigate to project directory
cd project

# If you have a trained model from Kaggle
python convert_kaggle_model.py

# This will create the TensorFlow.js model in public/Models/tfjs_model/
```

5. **Start the development server**:
```bash
# From project directory
npm run dev
```

6. **Open your browser** and navigate to `http://localhost:5173`

## 🧠 Model Training

### Option 1: Use Pre-trained Model (Recommended)
If you have a trained model from Kaggle, simply place it in the `project/Models/` directory and run the conversion script.

### Option 2: Train Locally
```bash
# Navigate to project directory
cd project

# Validate dataset structure
python scripts/validate_dataset.py

# Train the model locally
python scripts/train_local.py

# Test the trained model
python scripts/test_model.py
```

### Option 3: Train on Kaggle (Recommended for best results)
1. Upload the dataset to Kaggle
2. Use the provided Jupyter notebook: `project/scripts/kaggle_training_notebook.ipynb`
3. Train with GPU acceleration
4. Download the trained model to `project/Models/` directory

## 📁 Project Structure

```
SignSpeak/
├── project/                             # Main application directory
│   ├── public/
│   │   └── Models/
│   │       ├── inference_config.json    # Model configuration
│   │       └── tfjs_model/              # TensorFlow.js model files
│   │           ├── model.json
│   │           └── weights.bin
│   ├── src/
│   │   ├── components/
│   │   │   ├── WebcamCapture.tsx        # Main camera component
│   │   │   ├── ControlPanel.tsx         # App controls
│   │   │   ├── TranslationPanel.tsx     # Results display
│   │   │   ├── TutorMode.tsx            # Learning mode
│   │   │   └── Header.tsx               # App header
│   │   ├── utils/
│   │   │   ├── modelLoader.ts           # TensorFlow.js model loader
│   │   │   └── datasetManager.ts        # Dataset utilities
│   │   └── hooks/
│   │       └── useSpeechSynthesis.ts    # Speech synthesis hook
│   ├── Models/                          # Training models and configs
│   │   ├── asl_model.h5                 # Trained Keras model
│   │   ├── asl_model.tflite             # TensorFlow Lite model
│   │   ├── inference_config.json        # Model configuration
│   │   └── model_metadata.json          # Model metadata
│   ├── Data/
│   │   └── asl_dataset/                 # Training dataset
│   ├── scripts/                         # Python utilities
│   │   ├── convert_model_to_tfjs.py     # Model conversion
│   │   ├── train_local.py               # Local training
│   │   ├── test_model.py                # Model testing
│   │   └── kaggle_training_notebook.ipynb
│   ├── convert_kaggle_model.py          # Main conversion script
│   ├── package.json                     # Node.js dependencies
│   ├── requirements.txt                 # Python dependencies
│   └── vite.config.ts                   # Vite configuration
├── venv310/                             # Python virtual environment
├── README.md                            # This file
└── .gitignore                           # Git ignore patterns
```

## 🔧 Configuration

### Model Configuration (`project/public/Models/inference_config.json`)
```json
{
  "input_size": [64, 64],
  "num_classes": 36,
  "classes": ["0", "1", "2", ..., "a", "b", "c", ..., "z"],
  "confidence_threshold": 0.5,
  "preprocessing": {
    "normalize": true,
    "resize_method": "bilinear"
  }
}
```

### Environment Variables
Create a `.env` file in the project directory:
```env
VITE_MODEL_PATH=/Models/tfjs_model/model.json
VITE_CONFIDENCE_THRESHOLD=0.5
VITE_PREDICTION_INTERVAL=2000
```

## 🎮 Usage

### Real-time Recognition
1. **Enable webcam** when prompted
2. **Select ASL** from the language dropdown
3. **Show hand gestures** to the camera
4. **View predictions** in real-time
5. **Hear speech output** for recognized signs

### Tutor Mode
1. Click **"Tutor Mode"** to enter learning mode
2. **Practice specific letters** or numbers
3. **Get feedback** on your gestures
4. **Track your progress** over time

### Confidence Thresholds
- **Display threshold**: 0.4 (shows predictions)
- **Speech threshold**: 0.65 (triggers speech output)
- **Adjustable** in the control panel

## 🧪 Testing

### Model Testing
```bash
# Navigate to project directory
cd project

# Test the converted model
python scripts/test_model.py

# Test specific gestures
python scripts/inference_simple.py
```

### Frontend Testing
```bash
# From project directory
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🔄 Model Conversion Workflow

### From Kaggle to Web App
1. **Train on Kaggle**: Use GPU acceleration for best results
2. **Download model**: Save as `project/Models/asl_model.h5`
3. **Convert to TensorFlow.js**:
   ```bash
   cd project
   python convert_kaggle_model.py
   ```
4. **Verify conversion**: Check `project/public/Models/tfjs_model/`
5. **Test in browser**: Run `npm run dev`

### Conversion Script Features
- **Automatic detection** of model format
- **Placeholder creation** for development
- **Size optimization** for web deployment
- **Validation** of converted model

## 🎯 Performance

### Model Specifications
- **Input Shape**: (64, 64, 3)
- **Output Shape**: (36,) - 36 classes
- **Parameters**: ~2.17M parameters
- **Model Size**: ~8.5MB (TensorFlow.js)
- **Inference Time**: ~50-100ms per prediction

### Optimization Tips
- **Reduce prediction frequency** for better performance
- **Use GPU acceleration** when available
- **Optimize model size** with quantization
- **Implement frame skipping** for smoother UI

## 🚀 Deployment

### Development
```bash
cd project
npm run dev
```

### Production Build
```bash
cd project
npm run build
npm run preview
```

### Static Hosting
The app can be deployed to any static hosting service:
- **Vercel**: `vercel --prod`
- **Netlify**: Deploy the `project/dist/` folder
- **GitHub Pages**: Use GitHub Actions
- **Firebase Hosting**: `firebase deploy`

## 🤝 Contributing

### Development Setup
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Install dependencies**: 
   ```bash
   cd project
   npm install
   ```
4. **Set up Python environment**: `pip install -r project/requirements.txt`
5. **Make your changes**
6. **Test thoroughly**
7. **Submit a pull request**

### Code Style
- **TypeScript**: Strict mode enabled
- **ESLint**: Configured for React and TypeScript
- **Prettier**: Code formatting
- **Tailwind CSS**: Utility-first styling

## 📊 Model Performance

### Training Results
- **Accuracy**: ~95% on validation set
- **Training Time**: ~2-3 hours on Kaggle GPU
- **Dataset Size**: ~87,000 images
- **Classes**: 36 (0-9, a-z)

### Real-world Performance
- **Recognition Rate**: ~85-90% in good lighting
- **False Positive Rate**: <5%
- **Latency**: 2-second prediction intervals
- **Browser Compatibility**: Chrome, Firefox, Safari, Edge

## 🔍 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check if model files exist
ls project/public/Models/tfjs_model/

# Verify model format
cd project
python -c "import tensorflow as tf; print(tf.keras.models.load_model('Models/asl_model.h5').summary())"
```

#### Webcam Access Issues
- **Check permissions**: Allow camera access in browser
- **HTTPS required**: For production deployments
- **Multiple apps**: Close other applications using the camera

#### Poor Recognition Accuracy
- **Lighting**: Ensure good lighting conditions
- **Background**: Use plain background
- **Hand position**: Keep hand centered in frame
- **Distance**: Maintain appropriate distance from camera

#### Performance Issues
- **Reduce prediction frequency**: Increase interval in `WebcamCapture.tsx`
- **Lower resolution**: Adjust canvas size
- **Close other tabs**: Free up browser resources

### Debug Mode
Enable debug logging by adding to your `.env`:
```env
VITE_DEBUG=true
```

- **Project Repository**: https://github.com/Deveshu04/SignSpeak

## 🔮 Future Enhancements for Code for Bharat

- **Additional sign languages**: ISL, BSL, etc.
- **Sentence formation**: Word-to-sentence translation
- **Real-time conversation**: Two-way communication
- **Mobile app**: React Native version
- **API integration**: Cloud-based inference
- **Social features**: Community learning platform

---

**Made with ❤️ for the deaf and hard-of-hearing community**
