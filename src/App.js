// App.js
import React, { useState, useCallback } from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import ProcessingPanel from './ProcessingPanel';
import './styles.css';

const initialPipeline = {
  preprocessing: 'pending',
  ocrRecognition: 'pending',
  latexConversion: 'pending',
  validationOutput: 'pending',
};

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [latexOutput, setLatexOutput] = useState('');
  const [pipelineStatus, setPipelineStatus] = useState(initialPipeline);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileUpload = (uploadedFile) => {
    if (uploadedFile) {
      setFile(uploadedFile);
      setPreviewUrl(URL.createObjectURL(uploadedFile));
      setLatexOutput('');
      setPipelineStatus(initialPipeline);
    }
  };

  const resetPipeline = useCallback(() => {
    setPipelineStatus(initialPipeline);
    setLatexOutput('');
    setIsProcessing(false);
  }, []);

  const updateStepStatus = (step, status) => {
    setPipelineStatus(prev => ({ ...prev, [step]: status }));
  };

  const handleProcess = async () => {
    if (!file) return;

    setIsProcessing(true);
    setLatexOutput('Processing started...');
    resetPipeline(); // Reset state at start

    const formData = new FormData();
    formData.append('image', file);
    
    // --- Conceptual Backend API Call ---
    try {
        // --- STEP 1: Image Preprocessing ---
        updateStepStatus('preprocessing', 'processing');
        
        // This is where you call your backend API endpoint (e.g., /api/process)
        const response = await fetch('http://localhost:5000/api/process', { 
            method: 'POST',
            body: formData 
        });

        if (!response.ok) {
            throw new Error(`Server returned status: ${response.status}`);
        }

        const result = await response.json();
        
        // NOTE: In a real app, you would update the steps sequentially based on backend progress.
        // For simplicity here, we update all steps on final success/failure.
        
        if (result.success) {
            updateStepStatus('preprocessing', 'success');
            updateStepStatus('ocrRecognition', 'success');
            updateStepStatus('latexConversion', 'success');
            updateStepStatus('validationOutput', 'success');
            setLatexOutput(result.full_latex_document);
        } else {
            // Handle specific backend errors
            setLatexOutput(`ERROR: ${result.error_message}`);
            updateStepStatus('validationOutput', 'failed');
        }

    } catch (error) {
        console.error('Processing failed:', error);
        setLatexOutput(`\\text{Fatal Error: Could not connect to the processing pipeline.}`);
        updateStepStatus('validationOutput', 'failed');
    } finally {
        setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      <main className="dashboard-layout">
        <Sidebar />
        <ProcessingPanel
          file={file}
          previewUrl={previewUrl}
          latexOutput={latexOutput}
          pipelineStatus={pipelineStatus}
          isProcessing={isProcessing}
          onFileUpload={handleFileUpload}
          onProcess={handleProcess}
        />
      </main>
    </div>
  );
}