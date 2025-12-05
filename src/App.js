import React, { useState, useCallback } from 'react';
import Header from './Header';
import ProcessingPanel from './ProcessingPanel';
import './styles.css';
import ImageModal from './ImageModal';

const initialPipeline = {
  preprocessing: { status: 'pending', images: [] }, // Changed from image: null
  ocrRecognition: { status: 'pending', images: [] },
  segmentation: { status: 'pending', images: [] },
  modelInference: { status: 'pending', images: [] },
  reassembly: { status: 'pending', images: [] },
  validationOutput: { status: 'pending', images: [] },
};

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [latexOutput, setLatexOutput] = useState('');
  const [pipelineStatus, setPipelineStatus] = useState(initialPipeline);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);

  // Changed: store an array of images for the modal
  const [modalImages, setModalImages] = useState(null);
  const [modalTitle, setModalTitle] = useState('');

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
    setPipelineStatus(prev => ({ ...prev, [step]: { ...prev[step], status } }));
  };

  // Function to update all steps at once from the API response
  const updatePipelineFromBackend = (steps) => {
    const newStatus = {};
    Object.keys(steps).forEach(key => {
      newStatus[key] = {
        status: steps[key].status,
        // Map the backend images list to the frontend
        images: steps[key].images ? steps[key].images.map(img => ({
          label: img.label,
          src: `data:image/png;base64,${img.src}`
        })) : []
      };
    });
    setPipelineStatus(newStatus);
  };

  // REVISED: Accepts an array of images
  const showIntermediateImages = (title, imagesArray) => {
    setModalTitle(title);
    setModalImages(imagesArray);
  };

  const closeModal = () => {
    setModalImages(null);
    setModalTitle('');
  }

  const handleProcess = async () => {
    if (!file) return;

    setIsProcessing(true);
    setLatexOutput('Uploading...');
    resetPipeline();
    setJobId(null);

    const formData = new FormData();
    formData.append('image', file);

    let currentJobId = null;

    try {
      // 1. UPLOAD
      const uploadResponse = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData
      });
      const uploadResult = await uploadResponse.json();
      if (!uploadResult.success) throw new Error(uploadResult.error_message);

      currentJobId = uploadResult.job_id;
      setJobId(currentJobId);
      setLatexOutput(`Upload successful. Starting processing...`);

      // 2. TRIGGER PROCESSING (Now returns immediately because of threading)
      const processResponse = await fetch(`http://localhost:5000/api/process/${currentJobId}`, {
        method: 'POST',
      });
      if (!processResponse.ok) throw new Error(`Process start failed`);

      // 3. POLL FOR STATUS
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await fetch(`http://localhost:5000/api/status/${currentJobId}`);
          const statusResult = await statusResponse.json();

          // Update the UI with whatever step we are currently on
          if (statusResult.steps) {
            updatePipelineFromBackend(statusResult.steps);
          }

          if (statusResult.status === 'complete') {
            clearInterval(pollInterval);
            setLatexOutput(statusResult.result);
            setIsProcessing(false);
          } else if (statusResult.status === 'failed') {
            clearInterval(pollInterval);
            setLatexOutput(`\\text{Processing Failed: ${statusResult.error || 'Unknown error'}}`);
            updateStepStatus('validationOutput', 'failed');
            setIsProcessing(false);
          }
        } catch (e) {
          console.error("Polling error", e);
          // Don't stop polling on a single network blip, but maybe log it
        }
      }, 500); // Check every 0.5 seconds for snappier updates

    } catch (error) {
      console.error('Processing failed:', error);
      setLatexOutput(`\\text{Error: ${error.message}}`);
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      <main className="dashboard-layout">
        <ProcessingPanel
          file={file}
          previewUrl={previewUrl}
          latexOutput={latexOutput}
          pipelineStatus={pipelineStatus}
          isProcessing={isProcessing}
          onFileUpload={handleFileUpload}
          onProcess={handleProcess}
          // Pass the new modal handler
          onStepClick={showIntermediateImages}
        />

        {/* Render the Modal if images are set */}
        {modalImages && (
          <ImageModal
            title={modalTitle}
            images={modalImages}
            onClose={closeModal}
          />
        )}

      </main>
    </div>
  );
}