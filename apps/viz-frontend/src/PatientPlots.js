import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

const PatientPlots = ({ runId }) => {
    const [patientData, setPatientData] = useState([]);

    useEffect(() => {
        if (runId) {
            fetch(`http://localhost:8000/intervals/${runId}`)
                .then(response => response.json())
                .then(data => {
                    const grouped = groupByPatient(data);
                    setPatientData(grouped);
                })
                .catch(error => console.error('Error fetching intervals:', error));
        }
    }, [runId]);

    const groupByPatient = (data) => {
        return data.reduce((acc, item) => {
            const patientKey = item.person_id;
            acc[patientKey] = acc[patientKey] || [];
            acc[patientKey].push(item);
            return acc;
        }, {});
    };

    return (
        <div>
            {Object.entries(patientData).map(([personId, intervals]) => {
                // Create traces for each plan-criterion combination
                const traces = createTraces(intervals);

                return (
                    <Plot
                        key={personId}
                        data={traces}
                        layout={{
                            width: 1000,
                            height: 400,
                            title: `Patient ${personId}`,
                            barmode: 'stacked',
                            yaxis: {
                                tickmode: 'array',
                                tickvals: traces.map((_, i) => i),
                                ticktext: traces.map(trace => trace.name),
                                automargin: true,
                            },
                            xaxis: {
                                type: 'date'
                            },
                            showlegend: false
                        }}
                        useResizeHandler={true}
                    />
                );
            })}
        </div>
    );
};

const createTraces = (intervals) => {
    const traceMap = {};

    const MIN_DURATION = 1000000;

    intervals.forEach((interval, index) => {
        const traceKey = `${interval.plan_name || 'N/A'}-${interval.criterion_name || 'N/A'}`;
        if (!traceMap[traceKey]) {
            traceMap[traceKey] = {
                x: [],
                y: [],
                base: [],
                name: traceKey,
                type: 'bar',
                orientation: 'h', // Horizontal bar
                marker: { color: getColor(interval.interval_type) }
            };
        }
        const duration = Math.max(MIN_DURATION, new Date(interval.interval_end) - new Date(interval.interval_start));

        traceMap[traceKey].x.push(duration);
        traceMap[traceKey].base.push(new Date(interval.interval_start)); // Position on y-axis
        //traceMap[traceKey].x.push(new Date(interval.interval_end)); // Duration
        traceMap[traceKey].y.push(index); // Position on y-axis
    });

    return Object.values(traceMap);
};

const getColor = (intervalType) => {
    switch (intervalType) {
        case 'POSITIVE': return 'green';
        case 'NEGATIVE': return 'red';
        // Add more cases as needed
        default: return 'grey';
    }
};

export default PatientPlots;
