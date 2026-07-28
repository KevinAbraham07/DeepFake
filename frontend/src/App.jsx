import { useState } from 'react';
import UploadZone from './components/UploadZone';
import ResultsCard from './components/ResultsCard';
import { Loader2 } from 'lucide-react';

function App() {
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const handleImageSelected = async (file) => {
    setIsUploading(true);
    setResult(null);
    
    // Create local preview
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);

    // Send to API
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict/image', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error('API request failed');
      
      const data = await response.json();
      
      // Artificial delay so the user can enjoy the loading animation
      setTimeout(() => {
        setResult(data);
        setIsUploading(false);
      }, 800);
      
    } catch (err) {
      console.error(err);
      alert('Failed to analyze image. Make sure the FastAPI server is running!');
      setImagePreview(null);
      setIsUploading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setImagePreview(null);
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-50 font-sans p-4 md:p-8 selection:bg-purple-500/30 overflow-hidden relative">
      {/* Background glowing orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-blue-600/20 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-purple-600/20 rounded-full blur-[120px] pointer-events-none"></div>
      
      <div className="max-w-4xl mx-auto pt-8 relative z-10">
        <header className="mb-14 text-center animate-in fade-in slide-in-from-top-4 duration-700">
          <div className="inline-block px-4 py-1.5 mb-6 rounded-full bg-slate-800/80 border border-slate-700/50 text-slate-300 text-xs font-bold tracking-widest uppercase shadow-sm">
            Detection Framework <span className="text-blue-400">v1.0</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold bg-gradient-to-br from-white via-slate-200 to-slate-500 bg-clip-text text-transparent mb-6 tracking-tight drop-shadow-sm">
            Verify Authenticity
          </h1>
          <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
            Upload media to our neural network engine to instantly detect AI-generated or manipulated content.
          </p>
        </header>
        
        <main className="relative min-h-[400px]">
          {!result && !isUploading && (
            <div className="animate-in fade-in zoom-in-95 duration-500 delay-150 fill-mode-both">
              <UploadZone onImageSelected={handleImageSelected} isUploading={false} />
            </div>
          )}

          {isUploading && (
            <div className="glass rounded-3xl p-16 text-center animate-in fade-in zoom-in-95 duration-300 flex flex-col items-center justify-center absolute inset-0">
               <Loader2 className="w-12 h-12 text-blue-400 animate-spin mb-6 drop-shadow-[0_0_15px_rgba(96,165,250,0.5)]" />
               <h3 className="text-2xl font-bold text-slate-200 mb-2">Analyzing Media...</h3>
               <p className="text-slate-400 animate-pulse">Extracting features via dummy_cnn engine</p>
            </div>
          )}

          {result && !isUploading && (
            <ResultsCard 
              result={result} 
              imagePreview={imagePreview} 
              onReset={handleReset} 
            />
          )}
        </main>
      </div>
    </div>
  )
}

export default App;
