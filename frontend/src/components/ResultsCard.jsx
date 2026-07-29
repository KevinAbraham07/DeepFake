import { ShieldCheck, ShieldAlert, Activity, Clock, Layers } from 'lucide-react';

export default function ResultsCard({ result, imagePreview, onReset }) {
  if (!result) return null;

  const isFake = result.is_fake;
  const confidencePercent = (result.confidence * 100).toFixed(1);
  const breakdown = result.model_breakdown || [];
  
  return (
    <div className="glass rounded-3xl p-6 md:p-8 animate-in fade-in slide-in-from-bottom-4 duration-500 shadow-2xl">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Left: Image Preview */}
        <div className="relative rounded-2xl overflow-hidden bg-slate-900 border border-slate-800 aspect-square flex items-center justify-center group shadow-inner">
          {imagePreview ? (
            <img src={imagePreview} alt="Uploaded preview" className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" />
          ) : (
            <div className="text-slate-600 flex flex-col items-center">
               <Activity className="w-12 h-12 mb-2 animate-pulse" />
               <p>Processing...</p>
            </div>
          )}
          
          {/* Overlay gradient */}
          <div className="absolute inset-0 bg-gradient-to-t from-slate-950/90 via-transparent to-transparent opacity-80 pointer-events-none"></div>
        </div>

        {/* Right: Metrics & Combined Verdict */}
        <div className="flex flex-col justify-center space-y-6 md:pl-4">
          <div>
            <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full border ${isFake ? 'bg-red-500/10 border-red-500/30 text-red-400' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'} shadow-sm`}>
              {isFake ? <ShieldAlert className="w-5 h-5" /> : <ShieldCheck className="w-5 h-5" />}
              <span className="font-bold tracking-wide text-sm">{isFake ? 'MANIPULATION DETECTED (COMBINED)' : 'AUTHENTIC MEDIA (COMBINED)'}</span>
            </div>
            
            <h2 className="text-5xl font-black mt-4 text-slate-100 tracking-tight">
              {confidencePercent}% <span className="text-xl text-slate-500 font-medium tracking-normal block mt-1">Combined Ensemble Score</span>
            </h2>
          </div>

          {/* Model Breakdown Section */}
          {breakdown.length > 0 && (
            <div className="space-y-3 pt-4 border-t border-slate-800/50">
              <div className="flex items-center space-x-2 text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
                <Layers className="w-4 h-4 text-purple-400" />
                <span>Model Analysis Breakdown ({breakdown.length})</span>
              </div>
              
              {breakdown.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <div className="flex items-center space-x-2">
                    <span className={`w-2 h-2 rounded-full ${item.is_fake ? 'bg-red-500' : 'bg-emerald-500'}`}></span>
                    <span className="text-sm font-medium text-slate-200">{item.model_name}</span>
                  </div>
                  <span className={`text-xs font-bold px-2 py-1 rounded ${item.is_fake ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'}`}>
                    {item.is_fake ? 'FAKE' : 'REAL'} ({(item.confidence * 100).toFixed(1)}%)
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="space-y-3 pt-2 border-t border-slate-800/50">
            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/50 border border-slate-800/50">
              <div className="flex items-center text-slate-400">
                <Clock className="w-4 h-4 mr-3" />
                <span className="text-sm font-medium">Processing Time</span>
              </div>
              <span className="text-slate-200 font-mono">{(result.processing_time * 1000).toFixed(0)} ms</span>
            </div>
          </div>

          <button 
            onClick={onReset}
            className="w-full py-4 mt-2 rounded-xl bg-gradient-to-r from-slate-800 to-slate-900 hover:from-slate-700 hover:to-slate-800 text-slate-200 font-bold transition-all border border-slate-700 hover:border-slate-500 hover:shadow-lg focus:ring-2 focus:ring-slate-500 focus:outline-none flex items-center justify-center space-x-2"
          >
            <span>Analyze Another Image</span>
          </button>
        </div>
        
      </div>
    </div>
  );
}
