import React from 'react';
import { Play, Pause, Settings, Languages, Mic, MicOff, Hand } from 'lucide-react';

interface ControlPanelProps {
  isActive: boolean;
  onToggleActive: () => void;
  selectedLanguage: string;
  onLanguageChange: (language: string) => void;
  volume: number;
  onVolumeChange: (volume: number) => void;
  selectedSignLanguage: string;
  onSignLanguageChange: (signLanguage: string) => void;
}

const speechLanguages = [
  { code: 'en-US', name: 'English (US)', flag: '🇺🇸', category: 'English' },
  { code: 'en-GB', name: 'English (UK)', flag: '🇬🇧', category: 'English' },
];

const signLanguages = [
  { 
    code: 'ASL', 
    name: 'American Sign Language', 
    flag: '🇺🇸', 
    description: 'Used primarily in the US and Canada',
    color: 'from-blue-500 to-blue-600'
  },
  { 
    code: 'ISL', 
    name: 'Indian Sign Language', 
    flag: '🇮🇳', 
    description: 'Used across India with regional variations',
    color: 'from-orange-500 to-orange-600'
  },
  { 
    code: 'BSL', 
    name: 'British Sign Language', 
    flag: '🇬🇧', 
    description: 'Used in the UK and Northern Ireland',
    color: 'from-red-500 to-red-600'
  },
];

export const ControlPanel: React.FC<ControlPanelProps> = ({
  isActive,
  onToggleActive,
  selectedLanguage,
  onLanguageChange,
  volume,
  onVolumeChange,
  selectedSignLanguage,
  onSignLanguageChange,
}) => {
  const currentSignLang = signLanguages.find(sl => sl.code === selectedSignLanguage);

  return (
    <div className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 shadow-2xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-500/20 rounded-lg">
          <Settings className="w-5 h-5 text-blue-400" />
        </div>
        <h2 className="text-xl font-bold text-white">Controls</h2>
      </div>

      <div className="space-y-6">
        {/* Main Control */}
        <div className="space-y-3">
          <label className="text-white/80 text-sm font-medium">Recognition Status</label>
          <button
            onClick={onToggleActive}
            className={`w-full flex items-center justify-center gap-3 px-6 py-4 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 ${
              isActive
                ? 'bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white shadow-lg shadow-red-500/25'
                : 'bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white shadow-lg shadow-green-500/25'
            }`}
          >
            {isActive ? (
              <>
                <Pause className="w-5 h-5" />
                Stop Recognition
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Start Recognition
              </>
            )}
          </button>
        </div>

        {/* Sign Language Selection */}
        <div className="space-y-3">
          <label className="text-white/80 text-sm font-medium flex items-center gap-2">
            <Hand className="w-4 h-4" />
            Sign Language System
          </label>
          <div className="space-y-2">
            {signLanguages.map((signLang) => (
              <button
                key={signLang.code}
                onClick={() => onSignLanguageChange(signLang.code)}
                className={`w-full p-4 rounded-lg border transition-all duration-200 text-left ${
                  selectedSignLanguage === signLang.code
                    ? `bg-gradient-to-r ${signLang.color} border-white/30 text-white shadow-lg`
                    : 'bg-white/5 border-white/10 text-white/80 hover:bg-white/10 hover:border-white/20'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{signLang.flag}</span>
                  <div className="flex-1">
                    <div className="font-medium">{signLang.name}</div>
                    <div className={`text-xs mt-1 ${
                      selectedSignLanguage === signLang.code ? 'text-white/90' : 'text-white/60'
                    }`}>
                      {signLang.description}
                    </div>
                  </div>
                  {selectedSignLanguage === signLang.code && (
                    <div className="w-3 h-3 bg-white rounded-full"></div>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Speech Language Selection */}
        <div className="space-y-3">
          <label className="text-white/80 text-sm font-medium flex items-center gap-2">
            <Languages className="w-4 h-4" />
            Speech Output Language
          </label>
          <select
            value={selectedLanguage}
            onChange={(e) => onLanguageChange(e.target.value)}
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
            style={{
              backgroundImage: 'none',
              appearance: 'none'
            }}
          >
            {speechLanguages.map((lang) => (
              <option 
                key={lang.code} 
                value={lang.code} 
                className="bg-gray-800 text-white py-2"
              >
                {lang.flag} {lang.name}
              </option>
            ))}
          </select>
        </div>

        {/* Volume Control */}
        <div className="space-y-3">
          <label className="text-white/80 text-sm font-medium flex items-center gap-2">
            <Mic className="w-4 h-4" />
            Speech Volume
          </label>
          <div className="flex items-center gap-3">
            <MicOff className="w-4 h-4 text-white/40" />
            <input
              type="range"
              min="0"
              max="100"
              value={volume}
              onChange={(e) => onVolumeChange(Number(e.target.value))}
              className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer slider"
            />
            <Mic className="w-4 h-4 text-white/60" />
            <span className="text-white/80 text-sm font-medium w-8">
              {volume}%
            </span>
          </div>
        </div>

        {/* Status Indicators */}
        <div className="pt-4 border-t border-white/10 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-white/60 text-sm">Camera Access</span>
            <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-400' : 'bg-gray-400'}`}></div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-white/60 text-sm">{currentSignLang?.name} Processing</span>
            <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-blue-400 animate-pulse' : 'bg-gray-400'}`}></div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-white/60 text-sm">Speech Synthesis</span>
            <div className="w-3 h-3 rounded-full bg-purple-400"></div>
          </div>
        </div>

        {/* Current Configuration Summary */}
        <div className="pt-4 border-t border-white/10">
          <div className="bg-white/5 rounded-lg p-3">
            <div className="text-white/80 text-sm font-medium mb-2">Current Setup</div>
            <div className="space-y-1 text-xs text-white/60">
              <div>Sign Language: {currentSignLang?.name}</div>
              <div>Speech: {speechLanguages.find(l => l.code === selectedLanguage)?.name}</div>
              <div>Volume: {volume}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};