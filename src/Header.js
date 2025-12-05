// Header.js
import React from 'react';

export default function Header() {
  return (
    <header className="app-header">
      <div className="logo">MathScanner <span>Equation OCR Pipeline</span></div>
      <nav className="user-nav">
        <a href="#">History</a>
        <a href="#">Settings</a>
        <a href="#">Help</a>
        <button className="profile-btn">👤</button>
      </nav>
    </header>
  );
}