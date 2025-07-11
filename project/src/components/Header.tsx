import React from 'react';
import { Hand, Zap, Github, Heart } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="text-center mb-12">
      <div className="flex items-center justify-center gap-3 mb-6">
        <div className="p-3 bg-gradient-to-br from-purple-500 to-blue-500 rounded-2xl shadow-lg">
          <Hand className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
          SignSpeak
        </h1>
        <div className="p-2 bg-yellow-500/20 rounded-lg">
          <Zap className="w-6 h-6 text-yellow-400" />
        </div>
      </div>
      
      <p className="text-xl text-white/80 max-w-3xl mx-auto leading-relaxed mb-8">
        AI-powered real-time sign language to speech translation. 
        <span className="text-purple-300 font-medium"> Bridging communication barriers</span> with 
        cutting-edge computer vision and natural language processing.
      </p>

      <div className="flex items-center justify-center gap-6 text-white/60">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-sm">Real-time Processing</span>
        </div>
        <div className="flex items-center gap-2">
          <Heart className="w-4 h-4 text-red-400" />
          <span className="text-sm">Accessibility First</span>
        </div>
        <div className="flex items-center gap-2">
          <Github className="w-4 h-4" />
          <span className="text-sm">Open Source Ready</span>
        </div>
      </div>
    </header>
  );
};