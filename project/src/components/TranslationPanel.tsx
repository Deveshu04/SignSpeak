import React from 'react';
import { MessageSquare, Volume2, Download, Clock, Award, TrendingUp, Globe } from 'lucide-react';

interface Translation {
  id: string;
  text: string;
  confidence: number;
  timestamp: Date;
  signLanguage: string;
}

interface TranslationPanelProps {
  translations: Translation[];
  currentTranslation: string;
  onSpeak: (text: string) => void;
  isLoading: boolean;
  selectedSignLanguage: string;
}

export const TranslationPanel: React.FC<TranslationPanelProps> = ({ 
  translations, 
  currentTranslation, 
  onSpeak,
  isLoading,
  selectedSignLanguage
}) => {
  const handleDownloadTranscript = () => {
    const transcript = translations
      .map(t => `${t.timestamp.toLocaleTimeString()}: [${t.signLanguage}] ${t.text} (Confidence: ${Math.round(t.confidence * 100)}%)`)
      .join('\n');
    
    const blob = new Blob([transcript], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `signspeak-${selectedSignLanguage.toLowerCase()}-transcript-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-400 bg-green-400/20';
    if (confidence >= 0.8) return 'text-yellow-400 bg-yellow-400/20';
    if (confidence >= 0.7) return 'text-orange-400 bg-orange-400/20';
    return 'text-red-400 bg-red-400/20';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.9) return 'Excellent';
    if (confidence >= 0.8) return 'Good';
    if (confidence >= 0.7) return 'Fair';
    return 'Low';
  };

  const getSignLanguageColor = (signLang: string) => {
    switch (signLang) {
      case 'ASL': return 'bg-blue-500/20 text-blue-300';
      case 'ISL': return 'bg-orange-500/20 text-orange-300';
      case 'BSL': return 'bg-red-500/20 text-red-300';
      default: return 'bg-gray-500/20 text-gray-300';
    }
  };

  const getSignLanguageName = (code: string) => {
    switch (code) {
      case 'ASL': return 'American Sign Language';
      case 'ISL': return 'Indian Sign Language';
      case 'BSL': return 'British Sign Language';
      default: return code;
    }
  };

  const averageConfidence = translations.length > 0 
    ? translations.reduce((sum, t) => sum + t.confidence, 0) / translations.length 
    : 0;

  // Group translations by sign language for stats
  const signLanguageStats = translations.reduce((acc, t) => {
    acc[t.signLanguage] = (acc[t.signLanguage] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 shadow-2xl">
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <MessageSquare className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Sign Language Translation</h2>
              <p className="text-white/60 text-sm">
                {getSignLanguageName(selectedSignLanguage)} to Speech
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Current Sign Language Indicator */}
            <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${getSignLanguageColor(selectedSignLanguage)}`}>
              <Globe className="w-4 h-4" />
              <span className="text-sm font-medium">{selectedSignLanguage}</span>
            </div>
            
            {/* Confidence Stats */}
            {translations.length > 0 && (
              <div className="flex items-center gap-2 px-3 py-2 bg-white/10 rounded-lg">
                <TrendingUp className="w-4 h-4 text-blue-400" />
                <span className="text-white/80 text-sm">
                  Avg: {Math.round(averageConfidence * 100)}%
                </span>
              </div>
            )}
            
            {translations.length > 0 && (
              <button
                onClick={handleDownloadTranscript}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors duration-200"
              >
                <Download className="w-4 h-4" />
                <span className="text-sm font-medium">Export</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Current Translation */}
      <div className="p-6 border-b border-white/10">
        <div className="min-h-[140px] bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-xl p-6 border border-white/10">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-white/80">Analyzing {selectedSignLanguage} gesture...</span>
              </div>
            </div>
          ) : currentTranslation ? (
            <div className="space-y-4">
              <div className="flex items-start justify-between gap-4">
                <p className="text-2xl font-medium text-white leading-relaxed flex-1">
                  {currentTranslation}
                </p>
                
                {/* Confidence and Sign Language indicators for current translation */}
                <div className="flex flex-col gap-2">
                  {translations.length > 0 && (
                    <>
                      <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
                        getConfidenceColor(translations[0].confidence)
                      }`}>
                        <Award className="w-3 h-3" />
                        {getConfidenceLabel(translations[0].confidence)}
                      </div>
                      <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
                        getSignLanguageColor(translations[0].signLanguage)
                      }`}>
                        {translations[0].signLanguage}
                      </div>
                    </>
                  )}
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <button
                  onClick={() => onSpeak(currentTranslation)}
                  className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white rounded-lg transition-all duration-200 transform hover:scale-105"
                >
                  <Volume2 className="w-4 h-4" />
                  <span className="font-medium">Speak</span>
                </button>
                
                {translations.length > 0 && (
                  <div className="text-white/60 text-sm">
                    Confidence: {Math.round(translations[0].confidence * 100)}%
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-white/60">
              <div className="text-center">
                <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-lg">Start signing to see {selectedSignLanguage} translations</p>
                <p className="text-sm mt-1">Position your hands clearly in front of the camera</p>
                <div className="mt-4 text-xs text-white/40">
                  <p>Current mode: {getSignLanguageName(selectedSignLanguage)}</p>
                  <p>Supported: Letters, numbers, common phrases</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Translation History */}
      <div className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Clock className="w-4 h-4 text-white/60" />
          <h3 className="text-lg font-semibold text-white">Translation History</h3>
          <span className="px-2 py-1 bg-white/10 text-white/80 text-xs rounded-full">
            {translations.length}
          </span>
        </div>
        
        <div className="space-y-3 max-h-80 overflow-y-auto custom-scrollbar">
          {translations.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-white/40">No sign language translations yet</p>
              <p className="text-white/30 text-sm mt-1">Your translation history will appear here</p>
            </div>
          ) : (
            translations.map((translation) => (
              <div
                key={translation.id}
                className="p-4 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 transition-colors duration-200 cursor-pointer group"
                onClick={() => onSpeak(translation.text)}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1">
                    <p className="text-white font-medium mb-2">{translation.text}</p>
                    <div className="flex items-center gap-4 text-white/60 text-xs">
                      <span>{translation.timestamp.toLocaleTimeString()}</span>
                      <div className={`flex items-center gap-1 px-2 py-1 rounded-full ${
                        getSignLanguageColor(translation.signLanguage)
                      }`}>
                        {translation.signLanguage}
                      </div>
                      <div className={`flex items-center gap-1 px-2 py-1 rounded-full ${
                        getConfidenceColor(translation.confidence)
                      }`}>
                        <Award className="w-3 h-3" />
                        <span>{Math.round(translation.confidence * 100)}%</span>
                      </div>
                    </div>
                  </div>
                  <Volume2 className="w-4 h-4 text-white/40 group-hover:text-purple-400 transition-colors duration-200" />
                </div>
              </div>
            ))
          )}
        </div>
        
        {/* Enhanced Stats */}
        {translations.length > 0 && (
          <div className="mt-6 pt-4 border-t border-white/10">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-center">
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-lg font-bold text-white">{translations.length}</div>
                <div className="text-white/60 text-xs">Total Signs</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-lg font-bold text-green-400">
                  {Math.round(averageConfidence * 100)}%
                </div>
                <div className="text-white/60 text-xs">Avg Accuracy</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-lg font-bold text-blue-400">
                  {translations.filter(t => t.confidence >= 0.9).length}
                </div>
                <div className="text-white/60 text-xs">High Quality</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-lg font-bold text-purple-400">
                  {Object.keys(signLanguageStats).length}
                </div>
                <div className="text-white/60 text-xs">Sign Languages</div>
              </div>
            </div>
            
            {/* Sign Language Distribution */}
            {Object.keys(signLanguageStats).length > 1 && (
              <div className="mt-4 p-3 bg-white/5 rounded-lg">
                <div className="text-white/80 text-sm font-medium mb-2">Session Distribution</div>
                <div className="flex gap-2 text-xs">
                  {Object.entries(signLanguageStats).map(([lang, count]) => (
                    <div key={lang} className={`px-2 py-1 rounded ${getSignLanguageColor(lang)}`}>
                      {lang}: {count}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};