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
        <div  style={{ width: '100%', height: '100%' }}>
            {Object.entries(patientData).map(([personId, intervals]) => {
                // Create traces for each plan-criterion combination
                const traces = createTraces(intervals);

                let accumulatedBarCount = 0;
                const tickvals = traces.map(trace => {
                    const middlePosition = accumulatedBarCount + (trace.x.length -1) / 2;
                    accumulatedBarCount += trace.x.length; // Update the count for the next trace
                    return middlePosition;
                });
                const ticktext = traces.map(trace => trace.name);

                return (
                    <Plot
                        key={personId}
                        data={traces}
                        layout={{
                            height: 400,
                            title: `Patient ${personId}`,
                            barmode: 'stack',
                            yaxis: {
                                tickmode: 'array',
                                tickvals: tickvals,
                                ticktext: ticktext,
                                automargin: true,
                            },
                            xaxis: {
                                type: 'date',
                                //tickformat: '%H:%M\n%d-%b', // Hour and minute, new line, day and month
                                gridcolor: 'lightgrey',
                                //dtick: 36, // One hour in milliseconds
                                //minor: {
                                //    dtick: 86400000, // One day in milliseconds
                                //    gridcolor: 'grey', // Grid color for days
                                //    gridwidth: 2 // Grid line width for days
                                //},
                            },
                            showlegend: false
                        }}
                        useResizeHandler={true}
                        style={{width: '100%', height: '100%'}}
                    />
                );
            })}
        </div>
    );
};

const createTraces = (intervals) => {
    const traceMap = {};
    const labelPositionMap = {};
    var labelPosition = 0;

    const MIN_DURATION = 1000000;

    intervals.forEach((interval, index) => {
        const traceKey = `[${interval.cohort_category}] ${interval.pi_pair_name || 'N/A'}-${interval.criterion_name || 'N/A'}`;

        // Assign a consistent y-axis position for each unique label
        if (labelPositionMap[traceKey] === undefined) {
            labelPositionMap[traceKey] = labelPosition;
            labelPosition += Object.keys(labelPositionMap).length;
        }

        if (!traceMap[traceKey]) {
            traceMap[traceKey] = {
                x: [],
                y: [],
                base: [],
                yticks: [],
                name: traceKey,
                type: 'bar',
                orientation: 'h', // Horizontal bar
                marker: { color: getColor(interval.interval_type) }
            };
        }
        const duration = Math.max(MIN_DURATION, new Date(interval.interval_end) - new Date(interval.interval_start));

        traceMap[traceKey].base.push(new Date(interval.interval_start)); // x start
        traceMap[traceKey].x.push(duration); // x duration
        traceMap[traceKey].y.push(index); // Consistent position on y-axis
        traceMap[traceKey].yticks.push(index); // Label position on y-axis
    });

    return Object.values(traceMap);
};

const getColor = (intervalType) => {
    switch (intervalType) {
        case 'POSITIVE': return 'green';
        case 'NEGATIVE': return 'red';
        case 'NO_DATA': return 'grey';
        case 'NOT_APPLICABLE': return 'lightgrey';
        default: return 'black';
    }
};

export default PatientPlots;
