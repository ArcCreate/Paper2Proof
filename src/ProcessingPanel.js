import React, { useRef, useMemo } from 'react';
import { BlockMath } from 'react-katex';
import "katex/dist/katex.min.css";

export function extractEquations(latexString) {
  if (!latexString) return [];

  // Match anything between \[ ... \]
  const matches = latexString.match(/\\\[(.*?)\\\]/gs);

  if (!matches) return [];
  return matches.map(eq =>
    eq.replace(/\\\[/, "").replace(/\\\]/, "").trim()
  );
}

const copyToClipboard = (text) => {
  navigator.clipboard.writeText(text).then(() => {
    // Simple visual feedback (e.g., alert or console log)
    console.log(`Copied: ${text.substring(0, 30)}...`);
  }).catch(err => {
    console.error('Could not copy text: ', err);
  });
};

const LatexRenderer = ({ latex, jobId, onRerunEquation, rerunLoadingIndex }) => {
  if (!latex) {
    return (
      <div style={{ textAlign: 'center', opacity: 0.5, padding: '20px' }}>
        <div style={{ marginBottom: '10px' }}>∅</div>
        <div>Processing pending</div>
      </div>
    );
  }

  // Parse string to list of equations
  const equations = typeof latex === "string" ? extractEquations(latex) : latex;

  return (
    <div>
      {equations.map((eq, index) => {
        const isLoading = rerunLoadingIndex === index;

        return (
          <div
            key={index}
            onClick={() => copyToClipboard(eq)}
            style={{
              marginBottom: "20px",
              position: 'relative',
              cursor: 'copy',
              padding: '5px 0',
              border: isLoading ? '1px solid #ddd' : 'none',
              borderRadius: '4px',
              overflow: 'hidden'
            }}
          >
            {isLoading && (
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                height: '100%',
                background: 'rgba(255, 255, 255, 0.85)',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                zIndex: 10,
              }}>
                ⏳ **Rerunning Inference...**
              </div>
            )}

            <BlockMath
              math={eq}
              renderError={(error) => (
                <div
                  onClick={(e) => e.stopPropagation()}
                  style={{
                    border: '1px solid red',
                    padding: '10px',
                    borderRadius: '4px',
                    background: '#ffebeb',
                  }}
                >
                  <span style={{ color: "red", fontWeight: 'bold' }}>{error.name}</span>
                  {jobId && onRerunEquation && (
                    <button
                      className="retry-btn"
                      // Disable button while loading
                      disabled={isLoading}
                      onClick={(e) => {
                        e.stopPropagation();
                        onRerunEquation(jobId, index)
                      }}
                      style={{
                        marginLeft: '10px', padding: '5px 10px',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        backgroundColor: isLoading ? '#e0e0e0' : '#f9f9f9',
                        border: '1px solid #ccc',
                        borderRadius: '3px'
                      }}
                    >
                      {isLoading ? 'Loading...' : `Rerun Inference (Eq ${index + 1})`}
                    </button>
                  )}
                </div>
              )}
            />

          </div>
        );
      })}
    </div>
  );
};


const pipelineStepsMap = [
  { key: 'preprocessing', label: 'Image Preprocessing', desc: 'Noise reduction, contrast enhancement' },
  { key: 'ocrRecognition', label: 'Character Recognition', desc: 'OCR processing with math symbols' },
  { key: 'segmentation', label: 'Equation Segmentation', desc: 'Isolating distinct formula regions' },
  { key: 'latexGeneration', label: 'LaTeX Generation', desc: 'Converting to LaTeX format' },
  { key: 'validationOutput', label: 'Validation & Output', desc: 'Final quality check' },
];

export default function ProcessingPanel({
  file,
  previewUrl,
  latexOutput,
  pipelineStatus,
  isProcessing,
  onFileUpload,
  onProcess,
  onStepClick,
  jobId,
  onRerunEquation,
  rerunLoadingIndex
}) {
  const fileInputRef = useRef(null);

  // Helper to calculate progress percentage
  const progressPercent = useMemo(() => {
    // We map over our display steps to calculate progress based on what is visible
    const totalSteps = pipelineStepsMap.length;

    let completedCount = 0;

    // Check status of each mapped step
    if (pipelineStatus.preprocessing?.status === 'success') completedCount++;
    if (pipelineStatus.ocrRecognition?.status === 'success') completedCount++;
    if (pipelineStatus.segmentation?.status === 'success') completedCount++;
    if (pipelineStatus.modelInference?.status === 'success') completedCount++;
    if (pipelineStatus.validationOutput?.status === 'success') completedCount++;

    if (totalSteps === 0) return 0;
    if (isProcessing && completedCount === 0) return 5;

    return Math.round((completedCount / totalSteps) * 100);
  }, [pipelineStatus, isProcessing]);

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

  const getStatusText = (status) => {
    switch (status) {
      case 'success': return 'Complete';
      case 'processing': return 'Processing...';
      case 'pending': return 'Pending';
      case 'failed': return 'Failed';
      default: return 'Pending';
    }
  };

  const getStatusColorClass = (status) => {
    switch (status) {
      case 'success': return 'text-success';
      case 'processing': return 'text-processing';
      default: return 'text-pending';
    }
  };

  const getIconContent = (status) => {
    if (status === 'success') return '✓';
    if (status === 'processing') return '⟳';
    if (status === 'failed') return '!';
    return '';
  };

  // 2. MAPPED SEGMENTATION DATA HERE
  const displaySteps = [
    {
      ...pipelineStepsMap[0],
      status: pipelineStatus.preprocessing?.status || 'pending',
      data: pipelineStatus.preprocessing
    },
    {
      ...pipelineStepsMap[1],
      status: pipelineStatus.ocrRecognition?.status || 'pending',
      data: pipelineStatus.ocrRecognition
    },
    {
      ...pipelineStepsMap[2],
      status: pipelineStatus.segmentation?.status || 'pending',
      data: pipelineStatus.segmentation
    },
    {
      ...pipelineStepsMap[3],
      status: pipelineStatus.modelInference?.status || 'pending',
      data: pipelineStatus.modelInference
    },
    {
      ...pipelineStepsMap[4],
      status: pipelineStatus.validationOutput?.status || 'pending',
      data: pipelineStatus.validationOutput
    },
  ];

  return (
    <div className="dashboard-layout">

      {/* 1. Upload Section (Left) */}
      <div className="card upload-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3>Upload Image</h3>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="image/png, image/jpeg, .pdf"
            style={{ display: 'none' }}
          />
          <div style={{ color: '#999', fontSize: '1.2rem' }}></div>
        </div>

        <div
          className="drop-zone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {previewUrl ? (
            <div style={{ textAlign: 'center' }}>
              <img src={previewUrl} alt="Preview" style={{ maxHeight: '200px', maxWidth: '100%', marginBottom: '15px', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }} />
              <p style={{ fontSize: '0.9rem', color: '#666', fontWeight: '500' }}>{file.name}</p>
              {!isProcessing && (
                <button className="browse-btn" onClick={onProcess}>
                  Start Processing
                </button>
              )}
            </div>
          ) : (
            <>
              <div className="upload-icon-circle">☁️</div>
              <h4 style={{ margin: '0 0 10px 0' }}>Drop your math equation image here</h4>
              <p style={{ fontSize: '0.9rem', color: '#888', margin: 0 }}>Support for PNG, JPG, PDF files up to 10MB</p>
              <button
                className="browse-btn"
                onClick={() => fileInputRef.current.click()}
                disabled={isProcessing}
              >
                Browse Files
              </button>
            </>
          )}
        </div>
      </div>

      {/* 2. Pipeline Section (Right) */}
      <div className="card pipeline-card">
        <h3>Processing Pipeline</h3>

        <ul className="pipeline-list">
          {displaySteps.map((step, index) => (
            <li
              key={index}
              className={`pipeline-step ${step.status}`}
              onClick={() => {
                if (step.data && step.data.images && step.data.images.length > 0) {
                  onStepClick(step.label, step.data.images);
                }
              }}
              // Only show pointer cursor if there are images to show
              style={{ cursor: (step.data?.images?.length > 0) ? 'pointer' : 'default' }}
            >
              <div className="step-icon">
                {getIconContent(step.status)}
              </div>
              <div className="step-info">
                <span className="step-label">{step.label}</span>
                <span className="step-desc">{step.desc}</span>
              </div>

              <div style={{ textAlign: 'right' }}>
                <span className={`step-status-text ${getStatusColorClass(step.status)}`}>
                  {getStatusText(step.status)}
                </span>
                {/* Visual hint that images are available */}
                {step.data?.images?.length > 0 && (
                  <div style={{ fontSize: '0.7rem', color: '#666', marginTop: '2px' }}>
                    View {step.data.images.length} Image{step.data.images.length > 1 ? 's' : ''}
                  </div>
                )}
              </div>
            </li>
          ))}
        </ul>

        {/* Overall Progress Bar */}
        <div className="progress-container">
          <div className="progress-label">
            <span>Overall Progress</span>
            <span>{progressPercent}%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${progressPercent}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* 3. Output Section (Bottom) */}
      <div className="card output-section">
        <div className="output-header">
          <h3>LaTeX Output</h3>
          <div className="action-buttons">
            <button onClick={() => navigator.clipboard.writeText(latexOutput)}>
              📋 Copy
            </button>
            <button>⬇ Export</button>
          </div>
        </div>

        <div className="output-grid">
          {/* Left: Raw Code */}
          <div className="code-editor">
            <span className="editor-label">Raw LaTeX Code</span>
            <textarea
              className="raw-latex-textarea"
              value={latexOutput || ""}
              readOnly
              placeholder="\begin{align}..."
            />

          </div>

          {/* Right: Preview */}
          <div className="render-preview">
            <LatexRenderer
              latex={latexOutput}
              jobId={jobId}
              onRerunEquation={onRerunEquation}
              rerunLoadingIndex={rerunLoadingIndex}
            />
          </div>
        </div>
      </div>

    </div>
  );
}