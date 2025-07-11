import React, { useState, useCallback } from 'react';
import { Header } from './components/Header';
import { WebcamCapture } from './components/WebcamCapture';
import { TranslationPanel } from './components/TranslationPanel';
import { ControlPanel } from './components/ControlPanel';
import { TutorMode } from './components/TutorMode';
import { useSpeechSynthesis } from './hooks/useSpeechSynthesis';

interface Translation {
  id: string;
  text: string;
  confidence: number;
  timestamp: Date;
  signLanguage: string;
}

function App() {
  const [isActive, setIsActive] = useState(false);
  const [translations, setTranslations] = useState<Translation[]>([]);
  const [currentTranslation, setCurrentTranslation] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('en-US');
  const [selectedSignLanguage, setSelectedSignLanguage] = useState('ASL');
  const [volume, setVolume] = useState(80);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeMode, setActiveMode] = useState<'translate' | 'tutor'>('translate');

  const { speak, isSpeaking, isSupported } = useSpeechSynthesis();

  const handleGestureDetected = useCallback((gesture: string, confidence: number) => {
    const newTranslation: Translation = {
      id: Date.now().toString(),
      text: gesture,
      confidence,
      timestamp: new Date(),
      signLanguage: selectedSignLanguage,
    };

    setTranslations(prev => [newTranslation, ...prev.slice(0, 49)]); // Keep last 50
    setCurrentTranslation(gesture);
    
    // Auto-speak based on confidence threshold
    const confidenceThreshold = 0.82; // Higher threshold for sign language accuracy
    if (confidence >= confidenceThreshold && isSupported) {
      setTimeout(() => {
        speak(gesture, { language: selectedLanguage, volume });
      }, 150);
    }
  }, [speak, selectedLanguage, selectedSignLanguage, volume, isSupported]);

  const handleSpeak = useCallback((text: string) => {
    if (isSupported) {
      speak(text, { language: selectedLanguage, volume });
    }
  }, [speak, selectedLanguage, volume, isSupported]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-blue-900">
      {/* Enhanced animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-purple-600/20 to-transparent rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-blue-600/20 to-transparent rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-full blur-2xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12">
        <Header />

        {/* Mode Toggle */}
        <div className="flex justify-center mb-8">
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-2 border border-white/20">
            <div className="flex gap-2">
              <button
                onClick={() => setActiveMode('translate')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  activeMode === 'translate'
                    ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white shadow-lg'
                    : 'text-white/70 hover:text-white hover:bg-white/10'
                }`}
              >
                Sign to Speech
              </button>
              <button
                onClick={() => setActiveMode('tutor')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  activeMode === 'tutor'
                    ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white shadow-lg'
                    : 'text-white/70 hover:text-white hover:bg-white/10'
                }`}
              >
                Learn Signs
              </button>
            </div>
          </div>
        </div>

        {activeMode === 'translate' ? (
          <div className="grid lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
            {/* Left Column - Camera */}
            <div className="lg:col-span-1 space-y-6">
              <WebcamCapture
                onGestureDetected={handleGestureDetected}
                isActive={isActive}
                selectedSignLanguage={selectedSignLanguage}
              />
              
              <ControlPanel
                isActive={isActive}
                onToggleActive={() => setIsActive(!isActive)}
                selectedLanguage={selectedLanguage}
                onLanguageChange={setSelectedLanguage}
                volume={volume}
                onVolumeChange={setVolume}
                selectedSignLanguage={selectedSignLanguage}
                onSignLanguageChange={setSelectedSignLanguage}
              />
            </div>

            {/* Right Column - Translations */}
            <div className="lg:col-span-2">
              <TranslationPanel
                translations={translations}
                currentTranslation={currentTranslation}
                onSpeak={handleSpeak}
                isLoading={isProcessing}
                selectedSignLanguage={selectedSignLanguage}
              />
            </div>
          </div>
        ) : (
          <div className="max-w-7xl mx-auto">
            <TutorMode
              selectedSignLanguage={selectedSignLanguage}
              onSpeak={handleSpeak}
            />
          </div>
        )}

        {/* Enhanced Footer */}
        <footer className="mt-16 text-center text-white/40 text-sm">
          <div className="max-w-2xl mx-auto">
            <p className="mb-2">
              SignSpeak v2.0 - Supporting ASL, ISL & BSL • 
              Built with ❤️ for global accessibility
            </p>
            <p className="text-xs text-white/30">
              {activeMode === 'translate' 
                ? 'English speech synthesis with regional sign language support'
                : 'Interactive sign language learning with step-by-step guidance'
              }
            </p>
          </div>
        </footer>
      </div>

      {/* Enhanced custom scrollbar and slider styles */}
      <style jsx global>{`
        .custom-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: rgba(139, 92, 246, 0.3) transparent;
        }
        
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(139, 92, 246, 0.3);
          border-radius: 3px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(139, 92, 246, 0.5);
        }

        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(45deg, #8B5CF6, #06B6D4);
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }

        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(45deg, #8B5CF6, #06B6D4);
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }

        /* Enhanced animations */
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fadeInUp {
          animation: fadeInUp 0.5s ease-out;
        }
      `}</style>
    </div>
  );
}

export default App;