import React, { useEffect, useState } from 'react';

const RunSelector = ({ onSubmit }) => {
    const [runs, setRuns] = useState([]);
    const [selectedRun, setSelectedRun] = useState('');
    const [personId, setPersonId] = useState('');
    const [personSourceValue, setPersonSourceValue] = useState('');
    const [selectedDate, setSelectedDate] = useState('');

    useEffect(() => {
        // Fetch the list of recommendation runs from your API
        fetch(`${process.env.REACT_APP_API_URL}/execution_run/list`)
            .then(response => response.json())
            .then(data => setRuns(data))
            .catch(error => console.error('Error fetching runs:', error));
    }, []);

    const handleRunChange = (event) => {
        setSelectedRun(event.target.value);
    };

    const handlePersonIdChange = (event) => {
        setPersonId(event.target.value);
    };

    const handlePersonSourceValueChange = (event) => {
        setPersonSourceValue(event.target.value);
    };

    const handleDateChange = (event) => {
        setSelectedDate(event.target.value);
    };

    const handleSubmit = () => {
        if (onSubmit) {
            onSubmit({ runId: selectedRun, personId, personSourceValue, selectedDate });
        }
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            handleSubmit();
        }
    };

    return (
        <div>
            <select value={selectedRun} onChange={handleRunChange}>
                <option value="">Select a Run</option>
                {runs.map(run => (
                    <option key={run.recommendation_name} value={run.run_id}>
                        {run.recommendation_name}
                    </option>
                ))}
            </select>
            <input
                type="text"
                placeholder="Person ID"
                value={personId}
                onChange={handlePersonIdChange}
                onKeyPress={handleKeyPress}
            />
            <input
                type="text"
                placeholder="Person Source Value"
                value={personSourceValue}
                onChange={handlePersonSourceValueChange}
                onKeyPress={handleKeyPress}
            />
            <select id="date_selector" value={selectedDate} onChange={handleDateChange}>
                <option value="">Select Date</option>
                <option value="2020-02-22">2020-02-22</option>
                <option value="2020-05-06">2020-05-06</option>
                <option value="2020-11-08">2020-11-08</option>
                <option value="2021-01-29">2021-01-29</option>
                <option value="2021-04-01">2021-04-01</option>
                <option value="2021-11-29">2021-11-29</option>
                <option value="2022-03-20">2022-03-20</option>
                <option value="2022-09-14">2022-09-14</option>
                <option value="2022-12-22">2022-12-22</option>
                <option value="2023-03-25">2023-03-25</option>
            </select>
            <button onClick={handleSubmit}>Submit</button>
        </div>
    );
};

export default RunSelector;
