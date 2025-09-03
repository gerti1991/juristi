'use client';

import SearchInterface from '@/components/SearchInterface';
import { useState } from 'react';

export default function Home() {
  const [language, setLanguage] = useState<'al' | 'en'>('al');

  const translations = {
    al: {
      title: "⚖️ Juristi AI",
      subtitle: "Asistenti Ligjor Shqiptar - I fuqizuar nga AI-ja e Avancuar",
      chooseMode: "Zgjidhni Modalitetin e Kërkimit",
      description: "Merrni përgjigje të sakta ligjore ose analizë të thelluar nga dokumentet ligjore shqiptare",
      footer: "© 2025 Juristi AI. Fuqizimi i profesionistëve ligjorë me kapacitete kërkimi të drejtuara nga AI-ja.",
      languageToggle: "English"
    },
    en: {
      title: "⚖️ Juristi AI",
      subtitle: "Albanian Legal Assistant - Powered by Advanced AI",
      chooseMode: "Choose Your Query Mode",
      description: "Get precise legal answers or comprehensive analysis from Albanian legal documents",
      footer: "© 2025 Juristi AI. Empowering legal professionals with AI-driven research capabilities.",
      languageToggle: "Shqip"
    }
  };

  const t = translations[language];

  return (
    <main className="min-h-screen bg-[#0A0A0A] text-white">
      {/* Header */}
      <header className="border-b border-gray-800 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-between items-start mb-4">
            <div className="flex-1">
              <h1 className="text-3xl font-bold text-center bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                {t.title}
              </h1>
              <p className="text-gray-400 text-center mt-2">
                {t.subtitle}
              </p>
            </div>
            
            {/* Language Toggle Button */}
            <button
              onClick={() => setLanguage(language === 'al' ? 'en' : 'al')}
              className="ml-4 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white text-sm rounded-lg border border-gray-600 transition-colors duration-200 flex items-center gap-2"
              aria-label="Toggle language"
            >
              <span className="text-lg">
                {language === 'al' ? '🇺🇸' : '🇦🇱'}
              </span>
              {t.languageToggle}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto p-6">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-semibold mb-4">
            {t.chooseMode}
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            {t.description}
          </p>
        </div>

        <SearchInterface language={language} />
      </div>

      {/* Footer */}
      <footer className="mt-16 border-t border-gray-800 p-6">
        <div className="max-w-6xl mx-auto text-center text-gray-500 text-sm">
          <p>{t.footer}</p>
        </div>
      </footer>
    </main>
  );
}
