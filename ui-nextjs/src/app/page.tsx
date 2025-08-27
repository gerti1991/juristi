import SearchInterface from '@/components/SearchInterface';

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0A0A0A] text-white">
      {/* Header */}
      <header className="border-b border-gray-800 p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-center bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            ⚖️ Juristi AI
          </h1>
          <p className="text-gray-400 text-center mt-2">
            Albanian Legal Assistant - Powered by Advanced AI
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto p-6">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-semibold mb-4">
            Choose Your Query Mode
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Get precise legal answers or comprehensive analysis from Albanian legal documents
          </p>
        </div>

        <SearchInterface />
      </div>

      {/* Footer */}
      <footer className="mt-16 border-t border-gray-800 p-6">
        <div className="max-w-6xl mx-auto text-center text-gray-500 text-sm">
          <p>© 2025 Juristi AI. Empowering legal professionals with AI-driven research capabilities.</p>
        </div>
      </footer>
    </main>
  );
}
