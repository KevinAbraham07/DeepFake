import { useState, useRef } from 'react';
import { UploadCloud, AlertCircle } from 'lucide-react';

export default function UploadZone({ onImageSelected, isUploading }) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const validateAndProcessFile = (file) => {
    setError(null);
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPEG, PNG, etc).');
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB.');
      return;
    }

    onImageSelected(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    validateAndProcessFile(file);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    validateAndProcessFile(file);
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div 
        className={`glass rounded-3xl p-12 text-center border-2 border-dashed transition-all duration-300 cursor-pointer relative overflow-hidden group
          ${isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-slate-600 hover:border-purple-500 hover:bg-slate-800/50'}
          ${isUploading ? 'opacity-50 pointer-events-none' : ''}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          className="hidden" 
          ref={fileInputRef} 
          onChange={handleFileChange}
          accept="image/*"
        />
        
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className={`p-4 rounded-full ${isDragging ? 'bg-blue-500/20' : 'bg-slate-800 group-hover:bg-purple-500/20'} transition-colors duration-300`}>
            <UploadCloud className={`w-12 h-12 ${isDragging ? 'text-blue-400' : 'text-slate-400 group-hover:text-purple-400'}`} />
          </div>
          
          <div>
            <h3 className="text-xl font-semibold text-slate-200 mb-1">
              {isDragging ? 'Drop it here!' : 'Click or Drag & Drop'}
            </h3>
            <p className="text-slate-400 text-sm">
              Supports JPG, PNG, WEBP (Max 10MB)
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center space-x-3 text-red-400 max-w-md mx-auto">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <p className="text-sm">{error}</p>
        </div>
      )}
    </div>
  );
}
