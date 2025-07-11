/**
 * Model Loader for SignSpeak ASL Detection
 * Handles loading and inference with Kaggle-trained models
 */

import * as tf from '@tensorflow/tfjs';

export interface ModelConfig {
  modelPath: string;
  input_size: [number, number];  // Changed from inputSize to match config
  classes: string[];
  confidence_threshold: number;  // Changed from confidenceThreshold to match config
}

export interface Prediction {
  class: string;
  confidence: number;
}

export interface PredictionResult {
  class: string;
  confidence: number;
  timestamp: number;
  allPredictions: Prediction[];
}

export class ASLModelLoader {
  private model: tf.LayersModel | null = null;
  private config: ModelConfig | null = null;
  private isLoading = false;

  constructor() {
    console.log('🎯 ASL Model Loader initialized');
  }

  /**
   * Load the model and configuration
   */
  async load(modelPath = '/Models/tfjs_model/model.json'): Promise<boolean> {
    if (this.isLoading) {
      console.log('⏳ Model loading already in progress...');
      return false;
    }

    this.isLoading = true;

    try {
      console.log('📥 Loading model configuration...');

      // Load configuration
      const configResponse = await fetch('/Models/inference_config.json');
      if (!configResponse.ok) {
        throw new Error('Could not load model configuration');
      }

      this.config = await configResponse.json();
      console.log(`✅ Config loaded: ${this.config.classes.length} classes`);
      console.log(`📏 Input size: ${this.config.input_size[0]}x${this.config.input_size[1]}`);

      // Load the TensorFlow.js model
      console.log('🔄 Loading TensorFlow.js model...');

      try {
        this.model = await tf.loadLayersModel(modelPath);
        console.log('✅ Model loaded successfully from TensorFlow.js format');

        // Verify model input shape matches config
        const modelInputShape = this.model.inputs[0].shape;
        console.log(`📊 Model input shape: ${modelInputShape}`);
        console.log(`📊 Config input size: ${this.config.input_size}`);

      } catch (modelError) {
        console.log('⚠️ TensorFlow.js model not found, using placeholder');
        console.log('💡 Run: python convert_kaggle_model.py to convert your Kaggle model');
        this.createPlaceholderModel();
      }

      if (this.model) {
        console.log('📊 Model summary:');
        console.log(`   Input shape: ${this.model.inputs[0].shape}`);
        console.log(`   Output shape: ${this.model.outputs[0].shape}`);
        console.log(`   Classes: ${this.config.classes.length}`);
      }

      this.isLoading = false;
      return true;

    } catch (error) {
      console.error('❌ Failed to load model:', error);
      this.isLoading = false;
      return false;
    }
  }

  /**
   * Create a placeholder model for development/testing
   */
  private createPlaceholderModel(): void {
    console.log('🔧 Creating placeholder model for development');

    // Create placeholder that matches your Kaggle model structure (64x64 input)
    const inputSize = this.config?.input_size || [64, 64];
    const model = tf.sequential({
      layers: [
        tf.layers.inputLayer({ inputShape: [inputSize[0], inputSize[1], 3] }),
        tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        tf.layers.flatten(),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.5 }),
        tf.layers.dense({ units: 36, activation: 'softmax' })
      ]
    });

    // Initialize with random weights (placeholder)
    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    this.model = model;
    console.log('⚠️ Using placeholder model - convert your Kaggle model for production use');
  }

  /**
   * Preprocess image for model input
   */
  private preprocessImage(imageData: ImageData): tf.Tensor4D {
    return tf.tidy(() => {
      // Convert ImageData to tensor
      let tensor = tf.browser.fromPixels(imageData);

      // Resize to model input size (64x64 for your Kaggle model)
      const inputSize = this.config?.input_size || [64, 64];
      tensor = tf.image.resizeBilinear(tensor, inputSize);

      // Normalize to [0, 1]
      tensor = tensor.div(255.0);

      // Add batch dimension
      return tensor.expandDims(0) as tf.Tensor4D;
    });
  }

  /**
   * Make prediction on canvas element
   */
  async predict(canvas: HTMLCanvasElement): Promise<PredictionResult | null> {
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Get image data from canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Get predictions from the model
    const predictions = await this.predictFromImageData(imageData);

    if (predictions.length === 0) return null;

    const topPrediction = predictions[0];
    const threshold = this.config?.confidence_threshold || 0.5;

    if (topPrediction.confidence >= threshold) {
      return {
        class: topPrediction.class,
        confidence: topPrediction.confidence,
        timestamp: Date.now(),
        allPredictions: predictions
      };
    }

    return null;
  }

  /**
   * Make prediction on image data (renamed from predict to avoid conflict)
   */
  async predictFromImageData(imageData: ImageData): Promise<Prediction[]> {
    if (!this.model || !this.config) {
      console.error('❌ Model not loaded');
      return [];
    }

    try {
      const preprocessed = this.preprocessImage(imageData);

      // Make prediction
      const prediction = this.model.predict(preprocessed) as tf.Tensor;
      const scores = await prediction.data();

      // Clean up tensors
      preprocessed.dispose();
      prediction.dispose();

      // Get top predictions
      const predictions: Prediction[] = [];
      for (let i = 0; i < scores.length; i++) {
        if (i < this.config.classes.length) {
          predictions.push({
            class: this.config.classes[i],
            confidence: scores[i]
          });
        }
      }

      // Sort by confidence
      predictions.sort((a, b) => b.confidence - a.confidence);

      return predictions.slice(0, 5); // Return top 5

    } catch (error) {
      console.error('❌ Prediction failed:', error);
      return [];
    }
  }

  /**
   * Get the top prediction above threshold
   */
  async getTopPrediction(imageData: ImageData): Promise<Prediction | null> {
    const predictions = await this.predictFromImageData(imageData);

    if (predictions.length === 0) {
      return null;
    }

    const topPrediction = predictions[0];
    const threshold = this.config?.confidence_threshold || 0.5;

    if (topPrediction.confidence >= threshold) {
      return topPrediction;
    }

    return null;
  }

  /**
   * Check if model is loaded and ready
   */
  isReady(): boolean {
    return this.model !== null && this.config !== null && !this.isLoading;
  }

  /**
   * Get model information
   */
  getModelInfo() {
    if (!this.config) {
      return null;
    }

    return {
      classes: this.config.classes,
      inputSize: this.config.input_size,
      confidenceThreshold: this.config.confidence_threshold,
      numClasses: this.config.classes.length,
      isReady: this.isReady()
    };
  }

  /**
   * Dispose of the model to free memory
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    console.log('🗑️ Model disposed');
  }
}

// Create singleton instance
export const modelLoader = new ASLModelLoader();

/**
 * Utility function to convert your Kaggle H5 model to TensorFlow.js format
 * Run this in Python to prepare your model for web deployment:
 *
 * ```python
 * import tensorflowjs as tfjs
 *
 * # Convert H5 model to TensorFlow.js format
 * tfjs.converters.save_keras_model(
 *     keras_model,  # Your loaded Keras model
 *     'public/Models/tfjs_model'  # Output directory
 * )
 * ```
 */
