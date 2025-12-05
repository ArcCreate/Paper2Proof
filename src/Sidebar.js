// Sidebar.js
import React from 'react';

const mockScans = [
  { name: 'Calculus_Problem...', time: '2 mins ago', active: true },
  { name: 'Algebra_Equations', time: '1 hour ago', active: false },
  { name: 'Matrix_Operations', time: '1 hour ago', active: false },
];

const mockStats = [
  { label: 'Total Scans', value: 247 },
  { label: 'Success Rate', value: '94%' },
  { label: 'This Month', value: 38 },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      
      <section className="recent-scans">
        <h3>RECENT SCANS</h3>
        <ul>
          {mockScans.map((scan, index) => (
            <li key={index} className={`scan-item ${scan.active ? 'active' : ''}`}>
              {scan.name} <br /><span>{scan.time}</span>
            </li>
          ))}
        </ul>
      </section>
      
      <section className="statistics">
        <h3>STATISTICS</h3>
        {mockStats.map((stat, index) => (
          <div key={index} className="stat-line">
            <span>{stat.label}</span>
            <span style={{ fontWeight: 'bold' }}>{stat.value}</span>
          </div>
        ))}
      </section>
      
      <section className="quick-actions">
        <h3>QUICK ACTIONS</h3>
        <ul>
          <li>Open Folder</li>
          <li>Export Results</li>
        </ul>
      </section>
    </aside>
  );
}