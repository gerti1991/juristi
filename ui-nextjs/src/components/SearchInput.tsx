'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader } from 'lucide-react';

interface SearchInputProps {
  onSubmit: (query: string, mode: 'precise' | 'analyzed') => void;
  isLoading: boolean;
  placeholder?: string;
}

export const SearchInput: React.FC<SearchInputProps> = ({
  onSubmit,
  isLoading,
  placeholder = "Shkruani pyetjen tuaj ligjore këtu..."
}) => {
  const [query, setQuery] = useState('');
  const [showButtons, setShowButtons] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [query]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setQuery(value);
    setShowButtons(value.trim().length > 0);
  };

  const handleSubmit = (mode: 'precise' | 'analyzed') => {
    if (!query.trim() || isLoading) return;
    
    onSubmit(query.trim(), mode);
    setQuery('');
    setShowButtons(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // Default to precise mode on Enter
      handleSubmit('precise');
    }
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-dark-bg via-dark-bg/90 to-transparent p-6">
      <div className="max-w-4xl mx-auto">
        {/* Action Buttons */}
        {showButtons && !isLoading && (
          <div className="flex justify-center gap-4 mb-4 animate-slide-up">
            <button
              onClick={() => handleSubmit('precise')}
              className="group relative bg-glow-green/10 border border-glow-green/30 text-glow-green px-6 py-3 rounded-xl font-medium transition-all duration-300 hover:bg-glow-green/20 hover:scale-105 glow-green"
            >
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-glow-green rounded-full animate-pulse"></div>
                📍 Precize
              </div>
              <div className="text-xs text-text-secondary mt-1">Direct legal answers</div>
            </button>
            
            <button
              onClick={() => handleSubmit('analyzed')}
              className="group relative bg-glow-purple/10 border border-glow-purple/30 text-glow-purple px-6 py-3 rounded-xl font-medium transition-all duration-300 hover:bg-glow-purple/20 hover:scale-105 glow-purple"
            >
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-glow-purple rounded-full animate-pulse"></div>
                🧠 Analizuar
              </div>
              <div className="text-xs text-text-secondary mt-1">Comprehensive analysis</div>
            </button>
          </div>
        )}

        {/* Search Input */}
        <div className="relative bg-dark-surface/80 backdrop-blur-sm rounded-2xl border border-dark-border shadow-2xl">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            disabled={isLoading}
            rows={1}
            className="w-full bg-transparent text-text-primary placeholder-text-muted px-6 py-4 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-glow-blue/50 disabled:opacity-50"
            style={{ maxHeight: '200px', minHeight: '56px' }}
          />
          
          {/* Send Button */}
          <button
            onClick={() => handleSubmit('precise')}
            disabled={!query.trim() || isLoading}
            className="absolute right-3 bottom-3 p-2 bg-glow-blue/20 text-glow-blue rounded-lg transition-all duration-300 hover:bg-glow-blue/30 hover:scale-110 disabled:opacity-50 disabled:hover:scale-100 glow-blue"
          >
            {isLoading ? (
              <Loader className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="text-center mt-4 text-text-secondary animate-fade-in">
            <div className="inline-flex items-center gap-2">
              <Loader className="w-4 h-4 animate-spin" />
              <span className="loading-dots">Analyzing legal documents</span>
            </div>
          </div>
        )}

        {/* Example Queries */}
        {!showButtons && !isLoading && (
          <div className="mt-6 text-center">
            <div className="text-text-muted text-sm mb-3">Try these examples:</div>
            <div className="flex flex-wrap gap-2 justify-center">
              {[
                "Sa është dënimi për vrasje në Kodin Penal?",
                "Kushtet për divorc sipas ligjit shqiptar",
                "Të drejtat e punëtorit në Kodin e Punës"
              ].map((example, i) => (
                <button
                  key={i}
                  onClick={() => setQuery(example)}
                  className="text-xs px-3 py-1 bg-dark-elevated/50 text-text-secondary rounded-lg border border-dark-border hover:border-glow-blue/50 hover:text-glow-blue transition-all duration-300"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
