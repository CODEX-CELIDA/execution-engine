import React, { useState } from 'react';
import './App.css';
import RunSelector from './RunSelector';
import PatientPlots from './PatientPlots';

function App() {
    const [runDetails, setRunDetails] = useState({
        runId: '',
        personId: '',
        personSourceValue: '',
        selectedDate: '',
    });

    const handleRunSubmit = (details) => {
        console.log('Run Submit Details:', details); // Debugging line
        setRunDetails(details);
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Execution Engine Results Viewer</h1>
                <div className="RunSelector">
                    <RunSelector onSubmit={handleRunSubmit} />
                </div>
            </header>
            {runDetails.runId && (
                <div className="PlotContainer">
                    <PatientPlots
                        runId={runDetails.runId}
                        personId={runDetails.personId}
                        personSourceValue={runDetails.personSourceValue}
                        selectedDate={runDetails.selectedDate}
                    />
                </div>
            )}
        </div>
    );
}

export default App;
