import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

const PatientPlots = ({ runId, personId, personSourceValue, selectedDate }) => {
    const [patientData, setPatientData] = useState([]);

    useEffect(() => {
        if (runId) {
            const queryParams = new URLSearchParams();
            queryParams.append('run_id', runId);
            if (personId) queryParams.append('person_id', personId);
            if (personSourceValue) queryParams.append('person_source_value', personSourceValue);

            fetch(`${process.env.REACT_APP_API_URL}/intervals/${runId}?${queryParams.toString()}`)
                .then(response => response.json())
                .then(data => {
                    const grouped = groupByPatient(data);
                    setPatientData(grouped);
                })
                .catch(error => console.error('Error fetching intervals:', error));
        }
    }, [runId, personId, personSourceValue]);

    const groupByPatient = (data) => {
        return data.reduce((acc, item) => {
            const patientKey = item.person_id;
            acc[patientKey] = acc[patientKey] || [];
            acc[patientKey].push(item);
            return acc;
        }, {});
    };

    const createTraces = (intervals) => {
        const traceMap = {};
        let labelPosition = 0;

        intervals.forEach(interval => {
            const traceKey = `[${interval.cohort_category}] ${interval.pi_pair_name || 'N/A'}-${interval.criterion_description || 'N/A'}`;

            if (!traceMap[traceKey]) {
                traceMap[traceKey] = {
                    x: [],
                    y: [],
                    base: [],
                    name: traceKey,
                    type: 'bar',
                    orientation: 'h',
                    marker: { color: [] }, // Initialize as an array to hold individual colors
                    labelPosition: labelPosition++,
                };
            }

            const duration = new Date(interval.interval_end) - new Date(interval.interval_start);

            traceMap[traceKey].base.push(new Date(interval.interval_start));
            traceMap[traceKey].x.push(duration);
            traceMap[traceKey].y.push(traceMap[traceKey].labelPosition);
            traceMap[traceKey].marker.color.push(getColor(interval.interval_type)); // Add individual color
        });

        // Sort traceMap by traceKey
        return Object.values(traceMap).sort((a, b) => a.name.localeCompare(b.name));
    };

    const getColor = (intervalType) => {
        switch (intervalType) {
            case 'POSITIVE': return 'green';
            case 'NEGATIVE': return 'red';
            case 'NO_DATA': return 'grey';
            case 'NOT_APPLICABLE': return 'lightblue';
            default: return 'black';
        }
    };

    return (
        <div style={{ width: '100%', height: '100%' }}>
            {Object.entries(patientData).map(([personId, intervals]) => {
                const traces = createTraces(intervals);

                // Add vertical lines for the selected date
                if (selectedDate) {
                    const startOfDay = new Date(selectedDate + 'T00:00:00');
                    const endOfDay = new Date(selectedDate + 'T23:59:59');

                    const maxY = Math.max(...traces.map(trace => trace.labelPosition));

                    traces.push({
                        x: [startOfDay, startOfDay],
                        y: [0, maxY],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: 'blue', dash: 'dash' },
                        name: 'Start of Selected Date',
                        showlegend: false,
                    });

                    traces.push({
                        x: [endOfDay, endOfDay],
                        y: [0, maxY],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: 'blue', dash: 'dash' },
                        name: 'End of Selected Date',
                        showlegend: false,
                    });
                }

                // Generate tickvals and ticktext from the sorted traces
                const tickvals = traces.map(trace => trace.labelPosition);
                const ticktext = traces.map(trace => trace.name);

                // Set height proportional to the number of traceKeys
                const plotHeight = traces.length * 30 + 200;

                return (
                    <Plot
                        key={personId}
                        data={traces}
                        layout={{
                            height: plotHeight,
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
                                gridcolor: 'lightgrey',
                                range: selectedDate
                                    ? [new Date(selectedDate + 'T00:00:00'), new Date(selectedDate + 'T23:59:59')]
                                    : undefined,
                            },
                            showlegend: false,
                        }}
                        useResizeHandler={true}
                        style={{ width: '100%', height: '100%' }}
                    />
                );
            })}
        </div>
    );
};

export default PatientPlots;
