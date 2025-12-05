// ProcessingPanel.js
import React, { useRef, useState } from 'react';

const pipelineStepsMap = [
  { key: 'preprocessing', label: '1. Skew Correction' },
  { key: 'ocrRecognition', label: '2. Cleaning & Binarizing' },
  { key: 'segmentation', label: '3. Equation Segmentation' }, // NEW STEP
  { key: 'modelInference', label: '4. LaTeX OCR Inference' }, // NEW STEP
  { key: 'reassembly', label: '5. Document Reassembly' }, // NEW STEP
  { key: 'validationOutput', label: '6. Validation & Output' },
];


export default function ProcessingPanel({
  file,
  previewUrl,
  latexOutput,
  pipelineStatus,
  isProcessing,
  onFileUpload,
  onProcess,
  onStepClick
}) {
  const fileInputRef = useRef(null);
  const [activeStepImage, setActiveStepImage] = useState(null);

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const getStepIcon = (status) => {
    switch (status) {
      case 'success':
        return '✅';
      case 'processing':
        return '🔄';
      case 'failed':
        return '❌';
      default:
        return '⚪';
    }
  };

  const handleStepClick = (stepKey, label) => {
    const stepData = pipelineStatus[stepKey];
    if (stepData && stepData.image) {
      onStepClick(label, stepData.image); // Pass the step label and image to the parent
    }
  };

  return (
    <div className="main-content">
      <h2>OCR Pipeline Dashboard</h2>
      <p>Upload images containing mathematical equations to extract LaTeX code</p>

      {/* Upload Section */}
      <div
        className="upload-section"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <div className="drop-zone">
          <span className="upload-icon">☁️</span>
          <p>Drop your image here or click to browse</p>
          <button onClick={() => fileInputRef.current.click()} disabled={isProcessing}>
            Select File
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="image/png, image/jpeg, .pdf"
            style={{ display: 'none' }}
          />
          <p className="support-text">Supports PNG, JPG, PDF (Max 10MB)</p>
        </div>
      </div>

      <div className="io-panels">
        {/* Input Image Viewer - Always shows original previewUrl */}
        <div className="panel input-panel">
          <div className="panel-header">Input Image</div>
          <div className="panel-body image-viewer">
            <div className="image-placeholder">
              {previewUrl ? ( // Use original previewUrl only
                <img src={previewUrl} alt="Input Equation" style={{ maxWidth: '100%', maxHeight: '100%' }} />
              ) : (
                'No image uploaded'
              )}
            </div>
          </div>
          <div className="image-info">
            <p>Status: <span id="status-text">{isProcessing ? 'Processing...' : (file ? 'File Loaded' : 'Ready')}</span></p>
            <p>Resolution: <span id="resolution-text">--</span></p>
            <p>File Size: <span id="size-text">{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : '--'}</span></p>
          </div>
        </div>

        {/* LaTeX Output Viewer */}
        <div className="panel output-panel">
          <div className="panel-header">LaTeX Output</div>
          <textarea
            className="latex-output-box"
            value={latexOutput || '$ \\text{LaTeX output will appear here} $'}
            readOnly
          />
          <button
            className="process-btn"
            onClick={onProcess}
            disabled={!file || isProcessing}
          >
            {isProcessing ? 'Processing...' : 'Process Image'}
          </button>
        </div>
      </div>

      {/* Pipeline Status Section (Interactive) */}
      <div className="pipeline-status">
        <h3>Processing Pipeline Status (Click a successful step to view intermediate image)</h3>
        <ul id="pipeline-steps">
          {pipelineStepsMap.map(step => (
            <li
              key={step.key}
              className={pipelineStatus[step.key].status}
              // Pass the label to the handler
              onClick={() => handleStepClick(step.key, step.label)}
              style={{ cursor: pipelineStatus[step.key].image ? 'pointer' : 'default' }}
            >
              <span className="step-label">{step.label}</span>
              <span className="step-status">
                {getStepIcon(pipelineStatus[step.key].status)} {pipelineStatus[step.key].status}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}