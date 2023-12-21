import React, { useEffect, useState } from 'react';

const RunSelector = ({ onRunSelect }) => {
    const [runs, setRuns] = useState([]);
    const [selectedRun, setSelectedRun] = useState('');

    useEffect(() => {
        // Fetch the list of recommendation runs from your API
        fetch('http://localhost:8000/recommendation_runs')
            .then(response => response.json())
            .then(data => setRuns(data))
            .catch(error => console.error('Error fetching runs:', error));
    }, []);

    const handleChange = (event) => {
        setSelectedRun(event.target.value);
        onRunSelect(event.target.value);
    };

    return (
        <select value={selectedRun} onChange={handleChange}>
            <option value="">Select a Run</option>
            {runs.map(run => (
                <option key={run.recommendation_run_id} value={run.recommendation_run_id}>
                    {run.recommendation_run_id}
                </option>
            ))}
        </select>
    );
};

export default RunSelector;
