// App.js
import React, { useState, useCallback } from 'react';
import Header from './Header';
import ProcessingPanel from './ProcessingPanel';
import './styles.css';
import ImageModal from './ImageModal';

const initialPipeline = {
  preprocessing: { status: 'pending', image: null }, // Skew Correction
  ocrRecognition: { status: 'pending', image: null }, // Cleaning/Binarizing
  segmentation: { status: 'pending', image: null }, // Segmentation
  modelInference: { status: 'pending', image: null }, // Model Inference
  reassembly: { status: 'pending', image: null }, // Document Reassembly
  validationOutput: { status: 'pending', image: null }, // Final Output Status
};

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [latexOutput, setLatexOutput] = useState('');
  const [pipelineStatus, setPipelineStatus] = useState(initialPipeline);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [modalImage, setModalImage] = useState(null);
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
    setPipelineStatus(prev => ({ ...prev, [step]: status }));
  };

  // Function to update all steps at once from the API response
  const updatePipelineFromBackend = (steps) => {
    setPipelineStatus({
      preprocessing: {
        status: steps.preprocessing.status,
        image: steps.preprocessing.image ? `data:image/png;base64,${steps.preprocessing.image}` : null
      },
      ocrRecognition: {
        status: steps.ocrRecognition.status,
        image: steps.ocrRecognition.image ? `data:image/png;base64,${steps.ocrRecognition.image}` : null
      },
      segmentation: {
        status: steps.segmentation.status,
        image: steps.segmentation.image ? `data:image/png;base64,${steps.segmentation.image}` : null
      },
      modelInference: { status: steps.modelInference.status, image: null },
      reassembly: { status: steps.reassembly.status, image: null },
      validationOutput: { status: steps.validationOutput.status, image: null },
    });
  };

  const showIntermediateImage = (title, base64Image) => {
    setModalTitle(title);
    setModalImage(base64Image);
  };

  const closeModal = () => {
    setModalImage(null);
    setModalTitle('');
  }

  // --- REVISED handleProcess ---
  const handleProcess = async () => {
    if (!file) return;

    setIsProcessing(true);
    setLatexOutput('Uploading...');
    resetPipeline();
    setJobId(null); // Reset job ID

    const formData = new FormData();
    formData.append('image', file);

    let currentJobId = null;

    try {
      // --- STEP 1: UPLOAD & GET JOB ID ---
      const uploadResponse = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }
      const uploadResult = await uploadResponse.json();
      if (!uploadResult.success) {
        throw new Error(`Upload failed: ${uploadResult.error_message}`);
      }
      currentJobId = uploadResult.job_id;
      setJobId(currentJobId);
      setLatexOutput(`Upload successful. Starting job ${currentJobId}...`);

      // --- STEP 2: START PROCESSING ---
      const processResponse = await fetch(`http://localhost:5000/api/process/${currentJobId}`, {
        method: 'POST',
      });

      if (!processResponse.ok) {
        throw new Error(`Process start failed: ${processResponse.statusText}`);
      }

      // --- STEP 3: POLLING FOR STATUS ---
      setLatexOutput('Processing started. Waiting for results...');

      // Start Polling
      await new Promise((resolve) => {
        const pollInterval = setInterval(async () => {
          const statusResponse = await fetch(`http://localhost:5000/api/status/${currentJobId}`);
          const statusResult = await statusResponse.json();

          // Update status and images on the frontend
          updatePipelineFromBackend(statusResult.steps);

          if (statusResult.status === 'complete') {
            clearInterval(pollInterval);
            setLatexOutput(statusResult.result);
            resolve();
          } else if (statusResult.status === 'failed') {
            clearInterval(pollInterval);
            setLatexOutput(`\\text{Processing Failed: Check server logs.}`);
            updateStepStatus('validationOutput', 'failed');
            resolve();
          }
        }, 1000); // Poll every 1 second
      });

    } catch (error) {
      console.error('Processing failed:', error);
      setLatexOutput(`\\text{Fatal Error: Could not connect or pipeline failed.}`);
      updateStepStatus('validationOutput', 'failed');
    } finally {
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
          onStepClick={showIntermediateImage}
        />

        {/* Render the Modal if an image is set */}
        {modalImage && (
          <ImageModal
            title={modalTitle}
            imageSrc={modalImage}
            onClose={closeModal}
          />
        )}

      </main>
    </div>
  );
}