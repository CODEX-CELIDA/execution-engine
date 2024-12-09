import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

// Helper function to generate traceKey
const generateTraceKey = (interval) => {
    const descriptions = [];

    if (interval.pi_pair_name) descriptions.push(interval.pi_pair_name);
    if (interval.criterion_description) descriptions.push(interval.criterion_description);

    const descriptionPart = descriptions.join('-');
    return `[${interval.cohort_category}]` + (descriptionPart ? ` ${descriptionPart}` : '');
};

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

        // Collect traces
        intervals.forEach(interval => {
            const traceKey = generateTraceKey(interval); // Use the helper function

            if (!traceMap[traceKey]) {
                traceMap[traceKey] = {
                    x: [],
                    y: [],
                    width: [],
                    base: [],
                    name: traceKey,
                    type: 'bar',
                    orientation: 'h',
                    marker: { color: [] },
                    cohort_category: interval.cohort_category,
                    pi_pair_name: interval.pi_pair_name,
                    criterion_description: interval.criterion_description,
                };
            }

            const duration = new Date(interval.interval_end) - new Date(interval.interval_start);

            traceMap[traceKey].base.push(new Date(interval.interval_start));
            traceMap[traceKey].x.push(Math.max(duration, 20000));
            traceMap[traceKey].marker.color.push(getColor(interval.interval_type));
            traceMap[traceKey].width.push(interval.interval_type === 'NO_DATA' ? 0.5 : 0.8);
        });

        // Convert traceMap to an array
        const traces = Object.values(traceMap);

        // Define groups based on your criteria
        const groups = [
            {
                name: 'Single Criteria (Base)',
                traces: [],
                criteria: (trace) =>
                    (trace.cohort_category === 'BASE'),
            },
            {
                name: 'Single Criteria (Population)',
                traces: [],
                criteria: (trace) =>
                    (trace.cohort_category === 'POPULATION' && trace.criterion_description),
            },
            {
                name: 'Single Criteria (Intervention)',
                traces: [],
                criteria: (trace) =>
                    (trace.cohort_category === 'INTERVENTION' && trace.criterion_description),
            },
            // Groups for each pi_pair_name
            // We'll collect unique pi_pair_names and create groups dynamically
            {
                name: 'Full Recommendation',
                traces: [],
                criteria: (trace) =>
                    (trace.cohort_category === 'POPULATION' && !trace.pi_pair_name && !trace.criterion_description) ||
                    (trace.cohort_category === 'INTERVENTION' && !trace.pi_pair_name && !trace.criterion_description) ||
                    (trace.cohort_category === 'POPULATION_INTERVENTION' && !trace.pi_pair_name && !trace.criterion_description),
            },
        ];

        // Collect pi_pair_names for dynamic groups
        const piPairNames = [...new Set(traces
            .filter(trace => trace.pi_pair_name)
            .map(trace => trace.pi_pair_name))];

        // Create a group for each pi_pair_name
        piPairNames.sort().forEach(piPairName => {
            groups.splice(-1, 0, { // Insert before 'Recommendations' group
                name: "P/I Pair: " + piPairName,
                traces: [],
                criteria: (trace) => trace.pi_pair_name === piPairName,
            });
        });

        // Assign traces to groups
        traces.forEach(trace => {
            for (let group of groups) {
                if (group.criteria(trace)) {
                    group.traces.push(trace);
                    break; // A trace belongs to only one group
                }
            }
        });

        // Assign label positions and collect annotations
        let labelPosition = 0;
        const annotations = [];
        const sortedTraces = [];
        const yGap = 0.5; // Gap between groups
        const shapes = [];

        groups.forEach((group, groupIndex) => {
    // Add group header as an annotation
    annotations.push({
        xref: 'paper',
        y: labelPosition,
        x: 0,
        xanchor: 'right',
        yanchor: 'middle',
        text: `<b>${group.name}</b>`,
        showarrow: false,
        font: { size: 14 },
        align: 'right',
    });

    // Add a background rectangle behind the heading
    shapes.push({
        type: 'rect',
        xref: 'paper',
        x0: -0.5,
        x1: 1,
        yref: 'y',
        y0: labelPosition,
        y1: labelPosition + 1,
        fillcolor: '#f0f0f0',
        line: {
            width: 0,
        },
        layer: 'below',
    });

    labelPosition += 1; // Add gap before the group

            group.traces.forEach(trace => {
                trace.labelPosition = labelPosition;
                trace.y = Array(trace.x.length).fill(trace.labelPosition);
                sortedTraces.push(trace);
                labelPosition++;
            });

            // Add extra gap after group except the last one
            if (groupIndex < groups.length - 1) {
                labelPosition += yGap; // Gap after the group
            }
        });

        return { traces: sortedTraces, annotations, shapes };
    };

    const getColor = (intervalType) => {
        switch (intervalType) {
            case 'POSITIVE':
                return 'green';
            case 'NEGATIVE':
                return 'red';
            case 'NO_DATA':
                return 'lightgrey';
            case 'NOT_APPLICABLE':
                return 'lightblue';
            default:
                return 'black';
        }
    };

    return (
        <div style={{ width: '100%', height: '100%' }}>
            {Object.entries(patientData).map(([personId, intervals]) => {
                const { traces, annotations, shapes } = createTraces(intervals);

                // Add vertical lines for the selected date
                if (selectedDate) {
                    const startOfDay = new Date(`${selectedDate}T00:00:00`);
                    const endOfDay = new Date(`${selectedDate}T23:59:59`);

                    const maxY = Math.max(...traces.map(trace => trace.labelPosition));

                    traces.push({
                        x: [startOfDay, startOfDay],
                        y: [0, maxY + 1],
                        zorder: 1,
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: 'blue', dash: 'dash' },
                        name: 'Start of Selected Date',
                        showlegend: false,
                    });

                    traces.push({
                        x: [endOfDay, endOfDay],
                        y: [0, maxY + 1],
                        zorder: 1,
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: 'blue', dash: 'dash' },
                        name: 'End of Selected Date',
                        showlegend: false,
                    });
                }

                // Generate tickvals and ticktext from the sorted traces
                const tickvals = traces
                    .filter(trace => trace.labelPosition !== undefined)
                    .map(trace => trace.labelPosition);
                const ticktext = traces
                    .filter(trace => trace.labelPosition !== undefined)
                    .map(trace => trace.name);

                // Set height proportional to the number of traces
                const plotHeight = (Math.max(...tickvals) + 2) * 30 + 200;

                return (
                <Plot
                    key={personId}
                    data={traces}
                    layout={{
                        height: plotHeight,
                        title: `Patient ${personId}`,
                        barmode: 'stack',
                        dragmode: 'pan', // Set panning as the default interaction
                        yaxis: {
                            tickmode: 'array',
                            tickvals: tickvals,
                            ticktext: ticktext,
                            automargin: true,
                            autorange: 'reversed',
                            fixedrange: true, // Disable zooming and panning on y-axis
                        },
                        xaxis: {
                            type: 'date',
                            gridcolor: 'lightgrey',
                            range: selectedDate
                                ? [new Date(`${selectedDate}T00:00:00`), new Date(`${selectedDate}T23:59:59`)]
                                : undefined,
                        },
                        annotations: annotations,
                        //shapes: shapes, // Ensure shapes are included
                        margin: { t: 50, b: 50, l: 150, r: 50 }, // Adjusted left margin
                        showlegend: false,
                    }}
                     config={{
                         scrollZoom: true,   // Allow scroll zooming
                         displayModeBar: true,
                         displaylogo: false,
                         modeBarButtonsToRemove: ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'toImage'],
                         //modeBarButtonsToAdd: ['toImage'],
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
