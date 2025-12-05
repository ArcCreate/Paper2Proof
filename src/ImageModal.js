// ImageModal.js
import React from 'react';
// Assuming you style this modal in your styles.css

export default function ImageModal({ title, imageSrc, onClose }) {
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>{title}</h3>
                    <button className="close-button" onClick={onClose}>&times;</button>
                </div>
                <div className="modal-body">
                    <img
                        src={imageSrc}
                        alt={title}
                        style={{ maxWidth: '100%', maxHeight: '80vh', display: 'block', margin: 'auto' }}
                    />
                </div>
            </div>
        </div>
    );
}