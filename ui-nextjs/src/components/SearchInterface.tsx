'use client';

import React, { useState } from 'react';
import { Send, Loader, Brain, Target } from 'lucide-react';

interface SearchResponse {
  answer: string;
  sources: Array<{
    source: string;
    content: string;
    page?: number;
  }>;
  mode: 'precise' | 'analyzed';
}

interface SearchInterfaceProps {
  language: 'al' | 'en';
}

export default function SearchInterface({ language }: SearchInterfaceProps) {
  const [query, setQuery] = useState('');
  const [selectedMode, setSelectedMode] = useState<'precise' | 'analyzed'>('precise');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const translations = {
    al: {
      preciseMode: "Kërkimi i Saktë",
      preciseDescription: "Përgjigje të shpejta dhe të drejtpërdrejta nga dokumentet ligjore",
      analyzedMode: "Analiza e Thelluar", 
      analyzedDescription: "Analizë gjithëpërfshirëse me kontekst dhe shpjegime të detajuara",
      placeholder: "Shkruani pyetjen tuaj ligjore këtu në shqip ose anglisht...",
      searchButton: "Kërko",
      searching: "Duke kërkuar...",
      answer: "Përgjigja",
      sources: "Burimet",
      errorTitle: "Ka ndodhur një gabim",
      tryAgain: "Provoni përsëri",
      page: "Faqja"
    },
    en: {
      preciseMode: "Precise Search",
      preciseDescription: "Quick, direct answers from legal documents",
      analyzedMode: "Deep Analysis",
      analyzedDescription: "Comprehensive analysis with context and detailed explanations", 
      placeholder: "Type your legal question here in Albanian or English...",
      searchButton: "Search",
      searching: "Searching...",
      answer: "Answer",
      sources: "Sources",
      errorTitle: "An error occurred",
      tryAgain: "Try again",
      page: "Page"
    }
  };

  const t = translations[language];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const endpoint = selectedMode === 'precise' ? 'http://localhost:8000/search' : 'http://localhost:8000/analyse';
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      setResponse({
        ...data,
        mode: selectedMode,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Search error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Mode Selection */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={() => setSelectedMode('precise')}
          className={`flex items-center gap-2 px-6 py-3 rounded-lg border transition-all ${
            selectedMode === 'precise'
              ? 'bg-green-900/30 border-green-500 text-green-300'
              : 'bg-gray-800 border-gray-600 text-gray-300 hover:border-gray-500'
          }`}
        >
          <Target size={20} />
          <div className="text-left">
            <div className="font-medium">📍 {t.preciseMode}</div>
            <div className="text-xs text-gray-400">{t.preciseDescription}</div>
          </div>
        </button>
        <button
          onClick={() => setSelectedMode('analyzed')}
          className={`flex items-center gap-2 px-6 py-3 rounded-lg border transition-all ${
            selectedMode === 'analyzed'
              ? 'bg-purple-900/30 border-purple-500 text-purple-300'
              : 'bg-gray-800 border-gray-600 text-gray-300 hover:border-gray-500'
          }`}
        >
          <Brain size={20} />
          <div className="text-left">
            <div className="font-medium">🧠 {t.analyzedMode}</div>
            <div className="text-xs text-gray-400">{t.analyzedDescription}</div>
          </div>
        </button>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSubmit} className="mb-8">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t.placeholder}
            className="w-full p-4 pr-12 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none resize-none"
            rows={3}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="absolute bottom-4 right-4 p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors"
            title={isLoading ? t.searching : t.searchButton}
          >
            {isLoading ? (
              <Loader size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>
      </form>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg text-red-300">
          <p>❌ {t.errorTitle}: {error}</p>
        </div>
      )}

      {/* Response Display */}
      {response && (
        <div className="space-y-6">
          {/* Answer */}
          <div className={`p-6 rounded-lg border ${
            response.mode === 'precise'
              ? 'bg-green-900/20 border-green-700'
              : 'bg-purple-900/20 border-purple-700'
          }`}>
            <div className="flex items-center gap-2 mb-4">
              {response.mode === 'precise' ? (
                <>
                  <Target size={20} className="text-green-400" />
                  <span className="text-green-400 font-semibold">📍 {t.preciseMode}</span>
                </>
              ) : (
                <>
                  <Brain size={20} className="text-purple-400" />
                  <span className="text-purple-400 font-semibold">🧠 {t.analyzedMode}</span>
                </>
              )}
            </div>
            <div className="prose prose-invert max-w-none">
              <p className="whitespace-pre-wrap text-gray-200 leading-relaxed">
                {response.answer}
              </p>
            </div>
          </div>

          {/* Sources */}
          {response.sources && response.sources.length > 0 && (
            <div className="bg-gray-900 rounded-lg border border-gray-700 p-6">
              <h3 className="text-lg font-semibold mb-4 text-gray-200">
                📚 {t.sources} ({response.sources.length})
              </h3>
              <div className="space-y-3">
                {response.sources.map((source: any, index: number) => (
                  <div key={index} className="bg-gray-800 rounded-lg p-4 border border-gray-600">
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-blue-400">
                        {source.source}
                      </span>
                      {source.page && (
                        <span className="text-sm text-gray-400">
                          {t.page} {source.page}
                        </span>
                      )}
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      {source.content.substring(0, 200)}...
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
