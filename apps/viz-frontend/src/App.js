import React, { useState } from 'react';
import './App.css';
import RunSelector from './RunSelector';
import PatientPlots from './PatientPlots';

function App() {
    const [selectedRun, setSelectedRun] = useState('');

    return (
        <div className="App">
            <header className="App-header">
                <h1>Patient Data Visualizer</h1>
                <div className="RunSelector">
                    <RunSelector onRunSelect={setSelectedRun} />
                </div>
            </header>
            {selectedRun && (
                <div className="PlotContainer">
                    <PatientPlots runId={selectedRun} />
                </div>
            )}
        </div>
    );
}

export default App;
