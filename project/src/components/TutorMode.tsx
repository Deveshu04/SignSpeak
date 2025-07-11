import React, { useState, useCallback, useRef, useEffect } from 'react';
import { 
  BookOpen, 
  Search, 
  Play, 
  Pause, 
  RotateCcw, 
  Volume2, 
  Star, 
  Clock, 
  Award,
  Lightbulb,
  Hand,
  Eye,
  ChevronRight,
  Shuffle,
  Target,
  Brain
} from 'lucide-react';

interface SignData {
  word: string;
  signLanguage: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  category: 'greeting' | 'emotion' | 'question' | 'action' | 'letter' | 'number' | 'family' | 'daily' | 'nature' | 'food';
  description: string;
  steps: string[];
  tips: string[];
  videoUrl?: string;
  animationUrl?: string;
  relatedWords: string[];
  handShape: string;
  movement: string;
  location: string;
  confidence: number;
}

interface TutorModeProps {
  selectedSignLanguage: string;
  onSpeak: (text: string) => void;
}

export const TutorMode: React.FC<TutorModeProps> = ({ selectedSignLanguage, onSpeak }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [currentSign, setCurrentSign] = useState<SignData | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [searchResults, setSearchResults] = useState<SignData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [favorites, setFavorites] = useState<string[]>([]);
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [wordOfTheDay, setWordOfTheDay] = useState<SignData | null>(null);
  const [showQuiz, setShowQuiz] = useState(false);
  const [quizScore, setQuizScore] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);

  // Comprehensive sign database
  const signDatabase: SignData[] = [
    // ASL Signs
    {
      word: 'hello',
      signLanguage: 'ASL',
      difficulty: 'beginner',
      category: 'greeting',
      description: 'A friendly greeting gesture',
      steps: [
        'Raise your dominant hand to forehead level',
        'Keep fingers together and straight',
        'Move hand forward and slightly down',
        'Maintain eye contact and smile'
      ],
      tips: [
        'Keep the movement smooth and natural',
        'Facial expression is important - smile warmly',
        'Can be done with either hand'
      ],
      handShape: 'Open hand, fingers together',
      movement: 'Forward and down from forehead',
      location: 'Forehead area',
      relatedWords: ['hi', 'greetings', 'welcome'],
      confidence: 0.95
    },
    {
      word: 'thank you',
      signLanguage: 'ASL',
      difficulty: 'beginner',
      category: 'action',
      description: 'Express gratitude and appreciation',
      steps: [
        'Place fingertips near your chin/mouth',
        'Keep hand flat with fingers together',
        'Move hand forward toward the person',
        'End with palm facing up'
      ],
      tips: [
        'Start close to your face',
        'The movement represents giving thanks from your heart',
        'Maintain sincere facial expression'
      ],
      handShape: 'Flat hand, fingers together',
      movement: 'Forward from chin/mouth',
      location: 'Chin to forward space',
      relatedWords: ['thanks', 'grateful', 'appreciation'],
      confidence: 0.94
    },
    {
      word: 'love',
      signLanguage: 'ASL',
      difficulty: 'beginner',
      category: 'emotion',
      description: 'Express deep affection and care',
      steps: [
        'Cross both arms over your chest',
        'Place hands flat against your chest',
        'Hold the position with gentle pressure',
        'Show warm, caring expression'
      ],
      tips: [
        'This sign comes from the heart',
        'Keep arms relaxed, not tense',
        'Facial expression conveys the emotion'
      ],
      handShape: 'Flat hands',
      movement: 'Static position on chest',
      location: 'Chest area',
      relatedWords: ['care', 'affection', 'adore'],
      confidence: 0.92
    },
    {
      word: 'water',
      signLanguage: 'ASL',
      difficulty: 'beginner',
      category: 'daily',
      description: 'The essential liquid for life',
      steps: [
        'Make a "W" handshape with three fingers',
        'Tap the side of your mouth twice',
        'Keep the movement small and precise',
        'Maintain the W shape throughout'
      ],
      tips: [
        'The W represents the first letter of "water"',
        'Tap gently, don\'t hit hard',
        'Keep other fingers folded down'
      ],
      handShape: 'W handshape (three fingers up)',
      movement: 'Tapping motion at mouth',
      location: 'Side of mouth',
      relatedWords: ['drink', 'liquid', 'beverage'],
      confidence: 0.96
    },
    {
      word: 'family',
      signLanguage: 'ASL',
      difficulty: 'intermediate',
      category: 'family',
      description: 'People related by blood or choice',
      steps: [
        'Make F handshapes with both hands',
        'Start with hands together, thumbs touching',
        'Move hands apart in a circular motion',
        'End with hands facing each other'
      ],
      tips: [
        'The F represents the first letter of "family"',
        'The circular motion shows the family unit',
        'Keep movements smooth and connected'
      ],
      handShape: 'F handshape (thumb and index finger touching)',
      movement: 'Circular motion outward',
      location: 'Neutral space in front of body',
      relatedWords: ['relatives', 'clan', 'household'],
      confidence: 0.89
    },

    // ISL Signs
    {
      word: 'namaste',
      signLanguage: 'ISL',
      difficulty: 'beginner',
      category: 'greeting',
      description: 'Traditional Indian greeting showing respect',
      steps: [
        'Bring both palms together at chest level',
        'Keep fingers pointing upward',
        'Bow head slightly forward',
        'Hold position for a moment'
      ],
      tips: [
        'This is similar to the prayer position',
        'The bow shows respect and humility',
        'Keep palms pressed together firmly'
      ],
      handShape: 'Prayer position, palms together',
      movement: 'Static with slight bow',
      location: 'Chest level',
      relatedWords: ['greeting', 'respect', 'prayer'],
      confidence: 0.97
    },
    {
      word: 'dhanyawad',
      signLanguage: 'ISL',
      difficulty: 'intermediate',
      category: 'action',
      description: 'Thank you in Hindi/ISL',
      steps: [
        'Place right hand on heart',
        'Move hand forward toward person',
        'Keep palm open and facing up',
        'Bow head slightly'
      ],
      tips: [
        'Shows gratitude coming from the heart',
        'The forward movement gives thanks to others',
        'Facial expression should be sincere'
      ],
      handShape: 'Open palm',
      movement: 'From heart forward',
      location: 'Heart to forward space',
      relatedWords: ['thanks', 'gratitude', 'appreciation'],
      confidence: 0.91
    },
    {
      word: 'paani',
      signLanguage: 'ISL',
      difficulty: 'beginner',
      category: 'daily',
      description: 'Water in Hindi/ISL',
      steps: [
        'Cup your hand as if holding water',
        'Bring cupped hand to mouth',
        'Tilt hand as if drinking',
        'Repeat motion 2-3 times'
      ],
      tips: [
        'Mimic the action of drinking water',
        'Keep the cupping shape consistent',
        'Movement should be natural and flowing'
      ],
      handShape: 'Cupped hand',
      movement: 'Drinking motion to mouth',
      location: 'Mouth area',
      relatedWords: ['water', 'drink', 'liquid'],
      confidence: 0.94
    },

    // BSL Signs
    {
      word: 'hello',
      signLanguage: 'BSL',
      difficulty: 'beginner',
      category: 'greeting',
      description: 'British Sign Language greeting',
      steps: [
        'Raise your hand with palm facing forward',
        'Keep fingers together and straight',
        'Wave gently side to side',
        'Maintain friendly facial expression'
      ],
      tips: [
        'Similar to a regular wave but more formal',
        'Keep the wave small and controlled',
        'Eye contact is important in BSL'
      ],
      handShape: 'Open hand, palm forward',
      movement: 'Gentle waving motion',
      location: 'Shoulder height',
      relatedWords: ['hi', 'greetings', 'salutation'],
      confidence: 0.93
    },
    {
      word: 'please',
      signLanguage: 'BSL',
      difficulty: 'beginner',
      category: 'action',
      description: 'Polite request in BSL',
      steps: [
        'Place flat hand on chest',
        'Make small circular rubbing motion',
        'Keep movement gentle and smooth',
        'Show polite facial expression'
      ],
      tips: [
        'The circular motion is key to this sign',
        'Keep hand flat against chest',
        'Facial expression shows politeness'
      ],
      handShape: 'Flat hand',
      movement: 'Circular rubbing on chest',
      location: 'Chest area',
      relatedWords: ['request', 'ask', 'kindly'],
      confidence: 0.90
    }
  ];

  const categories = [
    { id: 'all', name: 'All Categories', icon: BookOpen },
    { id: 'greeting', name: 'Greetings', icon: Hand },
    { id: 'emotion', name: 'Emotions', icon: Star },
    { id: 'action', name: 'Actions', icon: Play },
    { id: 'daily', name: 'Daily Life', icon: Clock },
    { id: 'family', name: 'Family', icon: Award },
    { id: 'question', name: 'Questions', icon: Brain },
    { id: 'number', name: 'Numbers', icon: Target }
  ];

  const difficulties = [
    { id: 'beginner', name: 'Beginner', color: 'text-green-400', bg: 'bg-green-400/20' },
    { id: 'intermediate', name: 'Intermediate', color: 'text-yellow-400', bg: 'bg-yellow-400/20' },
    { id: 'advanced', name: 'Advanced', color: 'text-red-400', bg: 'bg-red-400/20' }
  ];

  // Filter signs based on selected sign language
  const getFilteredSigns = useCallback(() => {
    return signDatabase.filter(sign => 
      sign.signLanguage === selectedSignLanguage &&
      (selectedCategory === 'all' || sign.category === selectedCategory) &&
      (searchQuery === '' || 
       sign.word.toLowerCase().includes(searchQuery.toLowerCase()) ||
       sign.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
       sign.relatedWords.some(word => word.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    );
  }, [selectedSignLanguage, selectedCategory, searchQuery]);

  // Search functionality
  const handleSearch = useCallback(async (query: string) => {
    setSearchQuery(query);
    if (query.trim() === '') {
      setSearchResults([]);
      return;
    }

    setIsLoading(true);
    
    // Simulate API delay
    setTimeout(() => {
      const results = getFilteredSigns();
      setSearchResults(results);
      setIsLoading(false);
      
      // Add to recent searches
      if (query.trim() && !recentSearches.includes(query.trim())) {
        setRecentSearches(prev => [query.trim(), ...prev.slice(0, 4)]);
      }
    }, 300);
  }, [getFilteredSigns, recentSearches]);

  // Select a sign to learn
  const selectSign = useCallback((sign: SignData) => {
    setCurrentSign(sign);
    setIsPlaying(false);
  }, []);

  // Toggle favorite
  const toggleFavorite = useCallback((word: string) => {
    setFavorites(prev => 
      prev.includes(word) 
        ? prev.filter(w => w !== word)
        : [...prev, word]
    );
  }, []);

  // Play animation/video
  const togglePlayback = useCallback(() => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  }, [isPlaying]);

  // Set word of the day
  useEffect(() => {
    const availableSigns = signDatabase.filter(sign => sign.signLanguage === selectedSignLanguage);
    if (availableSigns.length > 0) {
      const randomSign = availableSigns[Math.floor(Math.random() * availableSigns.length)];
      setWordOfTheDay(randomSign);
    }
  }, [selectedSignLanguage]);

  // Get random sign for practice
  const getRandomSign = useCallback(() => {
    const availableSigns = getFilteredSigns();
    if (availableSigns.length > 0) {
      const randomSign = availableSigns[Math.floor(Math.random() * availableSigns.length)];
      selectSign(randomSign);
    }
  }, [getFilteredSigns, selectSign]);

  const getDifficultyStyle = (difficulty: string) => {
    const diff = difficulties.find(d => d.id === difficulty);
    return diff ? { color: diff.color, bg: diff.bg } : { color: 'text-gray-400', bg: 'bg-gray-400/20' };
  };

  return (
    <div className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 shadow-2xl">
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Sign Language Tutor</h2>
              <p className="text-white/60 text-sm">Learn {selectedSignLanguage} signs step by step</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={getRandomSign}
              className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors duration-200"
            >
              <Shuffle className="w-4 h-4" />
              <span className="text-sm">Random</span>
            </button>
          </div>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-white/40" />
          <input
            type="text"
            placeholder={`Search ${selectedSignLanguage} signs...`}
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
          />
        </div>

        {/* Categories */}
        <div className="flex flex-wrap gap-2 mt-4">
          {categories.map((category) => {
            const Icon = category.icon;
            return (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors duration-200 ${
                  selectedCategory === category.id
                    ? 'bg-purple-500/30 text-purple-300 border border-purple-400/30'
                    : 'bg-white/5 text-white/70 hover:bg-white/10 border border-white/10'
                }`}
              >
                <Icon className="w-3 h-3" />
                {category.name}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 p-6">
        {/* Left Column - Sign Learning */}
        <div className="space-y-6">
          {/* Word of the Day */}
          {wordOfTheDay && !currentSign && (
            <div className="bg-gradient-to-br from-yellow-500/10 to-orange-500/10 rounded-xl p-6 border border-yellow-400/20">
              <div className="flex items-center gap-3 mb-4">
                <Lightbulb className="w-5 h-5 text-yellow-400" />
                <h3 className="text-lg font-semibold text-white">Word of the Day</h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold text-white">{wordOfTheDay.word}</span>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${getDifficultyStyle(wordOfTheDay.difficulty).bg} ${getDifficultyStyle(wordOfTheDay.difficulty).color}`}>
                    {wordOfTheDay.difficulty}
                  </div>
                </div>
                <p className="text-white/80">{wordOfTheDay.description}</p>
                <button
                  onClick={() => selectSign(wordOfTheDay)}
                  className="flex items-center gap-2 px-4 py-2 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-300 rounded-lg transition-colors duration-200"
                >
                  <Eye className="w-4 h-4" />
                  Learn This Sign
                </button>
              </div>
            </div>
          )}

          {/* Current Sign Display */}
          {currentSign ? (
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <h3 className="text-2xl font-bold text-white">{currentSign.word}</h3>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${getDifficultyStyle(currentSign.difficulty).bg} ${getDifficultyStyle(currentSign.difficulty).color}`}>
                    {currentSign.difficulty}
                  </div>
                </div>
                <button
                  onClick={() => toggleFavorite(currentSign.word)}
                  className={`p-2 rounded-lg transition-colors duration-200 ${
                    favorites.includes(currentSign.word)
                      ? 'bg-yellow-500/20 text-yellow-400'
                      : 'bg-white/10 text-white/60 hover:bg-white/20'
                  }`}
                >
                  <Star className="w-4 h-4" />
                </button>
              </div>

              <p className="text-white/80 mb-6">{currentSign.description}</p>

              {/* Video/Animation Area */}
              <div className="bg-gray-800 rounded-lg aspect-video mb-6 flex items-center justify-center relative overflow-hidden">
                <div className="text-center">
                  <Hand className="w-16 h-16 text-white/40 mx-auto mb-4" />
                  <p className="text-white/60 mb-4">Sign demonstration for "{currentSign.word}"</p>
                  <div className="flex items-center justify-center gap-3">
                    <button
                      onClick={togglePlayback}
                      className="flex items-center gap-2 px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded-lg transition-colors duration-200"
                    >
                      {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button
                      onClick={() => setIsPlaying(false)}
                      className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white/70 rounded-lg transition-colors duration-200"
                    >
                      <RotateCcw className="w-4 h-4" />
                      Restart
                    </button>
                  </div>
                </div>
              </div>

              {/* Sign Details */}
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white/5 rounded-lg p-4">
                  <h4 className="text-white/80 text-sm font-medium mb-2">Hand Shape</h4>
                  <p className="text-white text-sm">{currentSign.handShape}</p>
                </div>
                <div className="bg-white/5 rounded-lg p-4">
                  <h4 className="text-white/80 text-sm font-medium mb-2">Movement</h4>
                  <p className="text-white text-sm">{currentSign.movement}</p>
                </div>
                <div className="bg-white/5 rounded-lg p-4">
                  <h4 className="text-white/80 text-sm font-medium mb-2">Location</h4>
                  <p className="text-white text-sm">{currentSign.location}</p>
                </div>
              </div>

              {/* Steps */}
              <div className="mb-6">
                <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Step-by-Step Instructions
                </h4>
                <div className="space-y-3">
                  {currentSign.steps.map((step, index) => (
                    <div key={index} className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-purple-500/20 text-purple-300 rounded-full flex items-center justify-center text-sm font-medium flex-shrink-0">
                        {index + 1}
                      </div>
                      <p className="text-white/80 text-sm">{step}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Tips */}
              <div className="mb-6">
                <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                  <Lightbulb className="w-4 h-4" />
                  Pro Tips
                </h4>
                <div className="space-y-2">
                  {currentSign.tips.map((tip, index) => (
                    <div key={index} className="flex items-start gap-3">
                      <ChevronRight className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                      <p className="text-white/70 text-sm">{tip}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Related Words */}
              {currentSign.relatedWords.length > 0 && (
                <div>
                  <h4 className="text-white font-medium mb-3">Related Words</h4>
                  <div className="flex flex-wrap gap-2">
                    {currentSign.relatedWords.map((word, index) => (
                      <button
                        key={index}
                        onClick={() => handleSearch(word)}
                        className="px-3 py-1 bg-white/10 hover:bg-white/20 text-white/70 text-sm rounded-lg transition-colors duration-200"
                      >
                        {word}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex items-center gap-3 mt-6 pt-6 border-t border-white/10">
                <button
                  onClick={() => onSpeak(currentSign.word)}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 rounded-lg transition-colors duration-200"
                >
                  <Volume2 className="w-4 h-4" />
                  Hear Word
                </button>
                <button
                  onClick={getRandomSign}
                  className="flex items-center gap-2 px-4 py-2 bg-green-500/20 hover:bg-green-500/30 text-green-300 rounded-lg transition-colors duration-200"
                >
                  <Shuffle className="w-4 h-4" />
                  Next Sign
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-white/5 rounded-xl p-12 border border-white/10 text-center">
              <BookOpen className="w-16 h-16 text-white/40 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">Ready to Learn?</h3>
              <p className="text-white/60 mb-6">Search for a word or select from the results below to start learning {selectedSignLanguage} signs</p>
              <button
                onClick={getRandomSign}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white rounded-lg transition-all duration-200 transform hover:scale-105 mx-auto"
              >
                <Shuffle className="w-4 h-4" />
                Start with Random Sign
              </button>
            </div>
          )}
        </div>

        {/* Right Column - Search Results & Recent */}
        <div className="space-y-6">
          {/* Search Results */}
          {searchQuery && (
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4">
                Search Results {isLoading && <span className="text-sm text-white/60">(searching...)</span>}
              </h3>
              
              {isLoading ? (
                <div className="space-y-3">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="bg-white/5 rounded-lg p-4 animate-pulse">
                      <div className="h-4 bg-white/10 rounded w-3/4 mb-2"></div>
                      <div className="h-3 bg-white/10 rounded w-1/2"></div>
                    </div>
                  ))}
                </div>
              ) : searchResults.length > 0 ? (
                <div className="space-y-3 max-h-80 overflow-y-auto custom-scrollbar">
                  {searchResults.map((sign, index) => (
                    <div
                      key={index}
                      onClick={() => selectSign(sign)}
                      className="bg-white/5 hover:bg-white/10 rounded-lg p-4 cursor-pointer transition-colors duration-200 border border-white/10"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-white font-medium">{sign.word}</h4>
                        <div className="flex items-center gap-2">
                          <div className={`px-2 py-1 rounded-full text-xs ${getDifficultyStyle(sign.difficulty).bg} ${getDifficultyStyle(sign.difficulty).color}`}>
                            {sign.difficulty}
                          </div>
                          {favorites.includes(sign.word) && (
                            <Star className="w-3 h-3 text-yellow-400 fill-current" />
                          )}
                        </div>
                      </div>
                      <p className="text-white/70 text-sm mb-2">{sign.description}</p>
                      <div className="flex items-center gap-2 text-xs text-white/50">
                        <span className="capitalize">{sign.category}</span>
                        <span>•</span>
                        <span>{sign.signLanguage}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Search className="w-12 h-12 text-white/40 mx-auto mb-3" />
                  <p className="text-white/60">No signs found for "{searchQuery}"</p>
                  <p className="text-white/40 text-sm mt-1">Try a different word or browse categories</p>
                </div>
              )}
            </div>
          )}

          {/* Recent Searches */}
          {recentSearches.length > 0 && !searchQuery && (
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Recent Searches
              </h3>
              <div className="flex flex-wrap gap-2">
                {recentSearches.map((search, index) => (
                  <button
                    key={index}
                    onClick={() => handleSearch(search)}
                    className="px-3 py-2 bg-white/10 hover:bg-white/20 text-white/70 text-sm rounded-lg transition-colors duration-200"
                  >
                    {search}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Popular Signs */}
          {!searchQuery && (
            <div className="bg-white/5 rounded-xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Star className="w-4 h-4" />
                Popular {selectedSignLanguage} Signs
              </h3>
              <div className="space-y-3">
                {getFilteredSigns().slice(0, 5).map((sign, index) => (
                  <div
                    key={index}
                    onClick={() => selectSign(sign)}
                    className="flex items-center justify-between p-3 bg-white/5 hover:bg-white/10 rounded-lg cursor-pointer transition-colors duration-200"
                  >
                    <div>
                      <h4 className="text-white font-medium">{sign.word}</h4>
                      <p className="text-white/60 text-sm">{sign.category}</p>
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs ${getDifficultyStyle(sign.difficulty).bg} ${getDifficultyStyle(sign.difficulty).color}`}>
                      {sign.difficulty}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Learning Stats */}
          <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-xl p-6 border border-purple-400/20">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Award className="w-4 h-4" />
              Your Progress
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">{favorites.length}</div>
                <div className="text-white/60 text-sm">Favorites</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">{recentSearches.length}</div>
                <div className="text-white/60 text-sm">Searched</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};