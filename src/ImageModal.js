import React from 'react';

export default function ImageModal({ title, images, onClose }) {
    // If no images array is passed or it's empty, don't render
    if (!images || images.length === 0) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>{title}</h3>
                    <button className="close-button" onClick={onClose}>&times;</button>
                </div>
                <div className="modal-body scrollable-gallery">
                    {images.map((imgData, index) => (
                        <div key={index} className="gallery-item">
                            <h4 className="gallery-label">{imgData.label}</h4>
                            <div className="image-wrapper">
                                <img
                                    src={imgData.src}
                                    alt={imgData.label}
                                    className="step-image"
                                />
                            </div>
                            {/* Add a divider if it's not the last image */}
                            {index < images.length - 1 && <hr className="gallery-divider" />}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}