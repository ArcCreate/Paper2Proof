import React, { useState, useCallback } from 'react';
import ProcessingPanel from './ProcessingPanel';
import ImageModal from './ImageModal';
import './styles.css';

// ... (Header Component)
function Header() {
  return (
    <header className="app-header">
      <div className="logo">MathScanner <span>Equation OCR Pipeline</span></div>
      <nav className="user-nav">
        {/* Placeholder links */}
      </nav>
    </header>
  );
}

const initialPipeline = {
  preprocessing: { status: 'pending', images: [] },
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

  const [modalImages, setModalImages] = useState(null);
  const [modalTitle, setModalTitle] = useState('');

  const [segmentResults, setSegmentResults] = useState([]);

  const [rerunLoadingIndex, setRerunLoadingIndex] = useState(-1);

  const handleFileUpload = (uploadedFile) => {
    if (uploadedFile) {
      setFile(uploadedFile);
      setPreviewUrl(URL.createObjectURL(uploadedFile));
      setLatexOutput('');
      setPipelineStatus(initialPipeline);
      setJobId(null);
    }
  };

  const resetPipeline = useCallback(() => {
    setPipelineStatus(initialPipeline);
    setLatexOutput('');
    setIsProcessing(false);
  }, []);

  const updatePipelineFromBackend = (steps) => {
    const newStatus = {};
    Object.keys(steps).forEach(key => {
      newStatus[key] = {
        status: steps[key].status,
        images: steps[key].images ? steps[key].images.map(img => ({
          label: img.label,
          src: `data:image/png;base64,${img.src}`
        })) : []
      };
    });
    setPipelineStatus(newStatus);
  };

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
      setLatexOutput(`\\text{Upload successful. Processing...}`);

      // 2. TRIGGER PROCESSING
      const processResponse = await fetch(`http://localhost:5000/api/process/${currentJobId}`, {
        method: 'POST',
      });
      if (!processResponse.ok) throw new Error(`Process start failed`);

      // 3. POLL FOR STATUS
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await fetch(`http://localhost:5000/api/status/${currentJobId}`);
          const statusResult = await statusResponse.json();

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
            setIsProcessing(false);
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 500);

    } catch (error) {
      console.error('Processing failed:', error);
      setLatexOutput(`\\text{Error: ${error.message}}`);
      setIsProcessing(false);
    }
  };

  const handleRerunEquation = async (jobId, index) => {
    if (!jobId) {
      console.error("Cannot rerun: Missing job ID.");
      return;
    }
    setRerunLoadingIndex(index);

    try {
      const response = await fetch(`http://localhost:5000/api/rerun_equation/${jobId}/${index}`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Rerun request failed');
      }

      const result = await response.json();
      setLatexOutput(result.new_latex_output);

    } catch (error) {
      console.error(`Rerunning equation ${index + 1} failed:`, error);
      setLatexOutput(prev => `\\text{Rerun Failed for Eq. } ${index + 1} \\text{: ${error.message}} \n ${prev}`);

    } finally {
      setRerunLoadingIndex(-1);
    }
  };

  return (
    <div className="app-container">
      <Header />
      <main>
        <ProcessingPanel
          file={file}
          previewUrl={previewUrl}
          latexOutput={latexOutput}
          pipelineStatus={pipelineStatus}
          isProcessing={isProcessing}
          onFileUpload={handleFileUpload}
          onProcess={handleProcess}
          onStepClick={showIntermediateImages}
          jobId={jobId}
          onRerunEquation={handleRerunEquation}
          rerunLoadingIndex={rerunLoadingIndex}
        />

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