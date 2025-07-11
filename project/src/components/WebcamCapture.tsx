import React, { useRef, useEffect, useState } from 'react';
import { Camera, CameraOff, AlertCircle, Hand, Brain, Zap, Info, RotateCcw, TrendingUp } from 'lucide-react';
import { modelLoader, Prediction, PredictionResult } from '../utils/modelLoader';

interface WebcamCaptureProps {
  onGestureDetected: (gesture: string, confidence: number) => void;
  isActive: boolean;
  selectedSignLanguage: string;
}


export const WebcamCapture: React.FC<WebcamCaptureProps> = ({
  onGestureDetected, 
  isActive, 
  selectedSignLanguage 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isVideoActive, setIsVideoActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastPrediction, setLastPrediction] = useState<PredictionResult | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictionResult[]>([]);
  const [gestureStats, setGestureStats] = useState<any>(null);
  const [detectionCount, setDetectionCount] = useState(0);

  // Load the ASL model
  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsProcessing(true);
        await modelLoader.load();
        console.log('✅ ASL model loaded');
      } catch (error) {
        console.error('❌ Model loading error:', error);
        setError('Failed to load the gesture recognition model. Please try again later.');
      } finally {
        setIsProcessing(false);
      }
    };

    loadModel();
  }, []);

  // Enhanced prediction function with landmark extraction
  const performPrediction = async (): Promise<void> => {
    if (!videoRef.current || !canvasRef.current) {
      return;
    }
    try {
      setIsProcessing(true);
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Get the prediction from the ASL model
      const prediction = await modelLoader.predict(canvas);

      if (prediction && prediction.confidence > 0.4) {
        setLastPrediction(prediction);
        setDetectionCount(prev => prev + 1);
        setPredictionHistory(prev => [prediction, ...prev.slice(0, 9)]);
        if (prediction.confidence > 0.65) {
          onGestureDetected(prediction.class, prediction.confidence);
        }
      }
    } catch (error) {
      console.error('❌ Prediction error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Start webcam and prediction loop
  useEffect(() => {
    let stream: MediaStream | null = null;
    let predictionInterval: NodeJS.Timeout | null = null;

    const startWebcam = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 }, 
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
          } 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsVideoActive(true);
          setError(null);
        }
      } catch (err) {
        setError('Unable to access webcam. Please check permissions and ensure your camera is not being used by another application.');
        console.error('Webcam error:', err);
      }
    };

    const startPredictionLoop = () => {
      // Start predictions for ASL when model is ready
      if (selectedSignLanguage === 'ASL') {
        predictionInterval = setInterval(() => {
          performPrediction();
        }, 2000); // Predict every 2 seconds for better demo experience
      }
    };

    if (isActive) {
      startWebcam();
      startPredictionLoop();
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (predictionInterval) {
        clearInterval(predictionInterval);
      }
      setIsVideoActive(false);
      setIsProcessing(false);
      setLastPrediction(null);
    };
  }, [isActive, selectedSignLanguage, onGestureDetected]);

  const resetGestureTracking = () => {
    setPredictionHistory([]);
    setLastPrediction(null);
    setGestureStats(null);
    setDetectionCount(0);
    console.log('🔄 Gesture tracking reset');
  };

  return (
    <div className="relative bg-gray-800 rounded-2xl overflow-hidden shadow-2xl">
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />
      
      <div className="aspect-video relative">
        {error ? (
          <div className="absolute inset-0 flex items-center justify-center bg-red-900/20 backdrop-blur-sm">
            <div className="text-center p-6 max-w-sm">
              <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
              <p className="text-red-300 font-medium text-sm leading-relaxed">{error}</p>
              <button 
                onClick={() => window.location.reload()} 
                className="mt-4 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg text-sm transition-colors"
              >
                Retry Camera Access
              </button>
            </div>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-full object-cover"
              style={{ transform: 'scaleX(-1)' }}
            />
            
            {/* Enhanced AI Processing Overlay */}
            {isProcessing && selectedSignLanguage === 'ASL' && (
              <div className="absolute inset-0 bg-blue-500/10 backdrop-blur-sm flex items-center justify-center">
                <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 text-center">
                  <div className="w-8 h-8 border-3 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                  <p className="text-white/90 font-medium">
                    Advanced Demo Processing...
                  </p>
                  <p className="text-white/70 text-sm mt-1">
                    Analyzing gesture patterns
                  </p>
                </div>
              </div>
            )}

            {/* Enhanced Model Status Indicator */}
            <div className="absolute top-4 left-4">
              <div className={`flex items-center gap-2 px-3 py-2 rounded-full backdrop-blur-md ${
                isVideoActive ? 'bg-green-500/20 text-green-300' : 'bg-gray-500/20 text-gray-300'
              }`}>
                {isVideoActive ? (
                  <Camera className="w-4 h-4" />
                ) : (
                  <CameraOff className="w-4 h-4" />
                )}
                <span className="text-sm font-medium">
                  {isVideoActive ? 'Camera Active' : 'Camera Inactive'}
                </span>
              </div>
            </div>

            {/* Enhanced Real-time Predictions Display */}
            {isActive && isVideoActive && selectedSignLanguage === 'ASL' && (
              <div className="absolute bottom-4 left-4 right-4">
                <div className="bg-white/10 backdrop-blur-md rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white/90 text-sm font-medium">
                      {lastPrediction ? 'Latest Detection' : 'Waiting for gesture...'}
                    </span>
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        {[...Array(3)].map((_, i) => (
                          <div
                            key={i}
                            className={`w-2 h-2 rounded-full ${
                              isProcessing 
                                ? 'bg-blue-400 animate-pulse'
                                : 'bg-gray-400'
                            }`}
                            style={{ animationDelay: `${i * 0.2}s` }}
                          />
                        ))}
                      </div>
                      {gestureStats && (
                        <button
                          onClick={resetGestureTracking}
                          className="p-1 bg-white/10 hover:bg-white/20 rounded text-white/60 hover:text-white/80 transition-colors"
                          title="Reset tracking"
                        >
                          <RotateCcw className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </div>
                  
                  {lastPrediction && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-white font-bold text-lg">
                          {lastPrediction.class}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-white/80 text-sm">
                            {Math.round(lastPrediction.confidence * 100)}%
                          </span>
                          {lastPrediction.confidence > 0.8 && (
                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                          )}
                        </div>
                      </div>
                      
                      {/* Enhanced Top 3 predictions */}
                      <div className="text-white/60 text-xs space-y-1">
                        <div className="text-white/80 font-medium mb-1">Top Predictions:</div>
                        {lastPrediction.allPredictions.slice(0, 3).map((pred, idx) => (
                          <div key={idx} className="flex justify-between items-center">
                            <span className={idx === 0 ? 'text-white/90 font-medium' : 'text-white/60'}>
                              {idx + 1}. {pred.class}
                            </span>
                            <div className="flex items-center gap-2">
                              <span>{Math.round(pred.confidence * 100)}%</span>
                              <div 
                                className="w-8 h-1 bg-white/20 rounded-full overflow-hidden"
                              >
                                <div 
                                  className={`h-full rounded-full ${
                                    idx === 0 ? 'bg-blue-400' : 'bg-white/40'
                                  }`}
                                  style={{ width: `${pred.confidence * 100}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Enhanced Prediction History */}
                  {predictionHistory.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-white/20">
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-white/60 text-xs">Recent Detections:</div>
                        <div className="text-white/50 text-xs">
                          Total: {detectionCount}
                        </div>
                      </div>
                      <div className="flex gap-2 text-xs flex-wrap">
                        {predictionHistory.slice(0, 6).map((pred, idx) => (
                          <span 
                            key={idx} 
                            className={`px-2 py-1 rounded ${
                              idx === 0 
                                ? 'bg-blue-500/30 text-blue-200' 
                                : 'bg-white/10 text-white/70'
                            }`}
                          >
                            {pred.class}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Gesture Statistics */}
                  {gestureStats && gestureStats.totalGestures > 0 && (
                    <div className="mt-3 pt-3 border-t border-white/20">
                      <div className="grid grid-cols-3 gap-3 text-center text-xs">
                        <div>
                          <div className="text-white font-medium">{gestureStats.totalGestures}</div>
                          <div className="text-white/60">Total</div>
                        </div>
                        <div>
                          <div className="text-white font-medium">{gestureStats.uniqueGestures}</div>
                          <div className="text-white/60">Unique</div>
                        </div>
                        <div>
                          <div className="text-white font-medium">{gestureStats.mostFrequent}</div>
                          <div className="text-white/60">Most Used</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Non-ASL Language Notice */}
            {selectedSignLanguage !== 'ASL' && isActive && (
              <div className="absolute bottom-4 left-4 right-4">
                <div className="bg-yellow-500/20 backdrop-blur-md rounded-lg p-4 border border-yellow-400/30">
                  <div className="flex items-center gap-3">
                    <Hand className="w-5 h-5 text-yellow-400" />
                    <div>
                      <div className="text-yellow-300 font-medium text-sm">
                        {selectedSignLanguage} Advanced Demo Coming Soon
                      </div>
                      <div className="text-yellow-200/80 text-xs mt-1">
                        Currently only ASL has advanced gesture detection. Switch to ASL for the full experience.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};