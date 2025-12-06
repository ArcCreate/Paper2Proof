import React, { useEffect } from 'react';

export default function ImageModal({ title, images, onClose }) {
    // Prevent scrolling on the body when modal is open
    useEffect(() => {
        document.body.style.overflow = 'hidden';
        return () => {
            document.body.style.overflow = 'unset';
        };
    }, []);

    if (!images || images.length === 0) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>

                {/* Header */}
                <div className="modal-header">
                    <h3 className="modal-title">{title}</h3>
                    <button className="modal-close-btn" onClick={onClose}>
                        &times;
                    </button>
                </div>

                {/* Body / Gallery */}
                <div className="modal-body">
                    <div className="gallery-list">
                        {images.map((imgData, index) => (
                            <div key={index} className="gallery-item">
                                <div className="gallery-label-container">
                                    <span className="gallery-step-number">{index + 1}</span>
                                    <span className="gallery-label">{imgData.label}</span>
                                </div>

                                <div className="image-wrapper">
                                    <img
                                        src={imgData.src}
                                        alt={imgData.label}
                                        className="gallery-image"
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Footer*/}
                <div className="modal-footer">
                    <button className="secondary-btn" onClick={onClose}>Close Preview</button>
                </div>
            </div>
        </div>
    );
}