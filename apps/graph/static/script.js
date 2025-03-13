// script.js

async function loadRecommendations() {
    const response = await fetch(`${eeApiUrl}/recommendation/list`);
    const recommendations = await response.json();
    const recommendationList = document.getElementById('recommendation-list');
    recommendationList.innerHTML = '';
    recommendations.forEach(rec => {
        const div = document.createElement('div');
        div.className = 'recommendation-item';
        div.innerHTML = `
            <div class="recommendation-title">${rec.recommendation_id}: ${rec.recommendation_name}</div>
            <div class="recommendation-detail">Version: ${rec.recommendation_version}</div>
            <div class="recommendation-detail">Package Version: ${rec.recommendation_package_version}</div>
        `;
        div.onclick = () => loadGraph(rec.recommendation_id);
        recommendationList.appendChild(div);
    });
}

function checkCount(ele) {
    if (ele && ele.data) {
        const count_min = ele.data('count_min');
        const count_max = ele.data('count_max');

        if (count_min !== null && count_max === null) {
            return `>=${count_min}`;
        } else if (count_min === null && count_max !== null) {
            return `<=${count_max}`;
        } else if (count_min !== null && count_max !== null) {
            return `${count_min} - ${count_max}`;
        }
    }
}

async function loadGraph(recommendationId) {
    const response = await fetch(`${eeApiUrl}/recommendation/${recommendationId}/execution_graph`);
    const data = await response.json();
    const graphData = data.recommendation_execution_graph;

    // Extract unique node types
    const nodeTypes = [...new Set(graphData.nodes.map(node => node.data.type))];
    const nodeCategories = [...new Set(graphData.nodes.map(node => node.data.category))];

    // Generate colors for each type
    const nodeColors = {
        "BASE": "#ff0000",
        "POPULATION": "#00ff00",
        "INTERVENTION": "#9999ff",
        "POPULATION_INTERVENTION": "#ff00ff",
    };
    const nodeShapes = {
        "Symbol": "round-rectangle",
        "&": "rhomboid",
        "|": "diamond",
        "Not": "triangle",
        "NoDataPreservingAnd": "rhomboid",
        "NoDataPreservingOr": "diamond",
        "NonSimplifiableAnd": "rhomboid",
        "NonSimplifiableOr": "diamond",
        "LeftDependentToggle": "octagon",
    }

    // Initialize Cytoscape
    var cy = cytoscape({
        container: document.getElementById('cy'),
        elements: [...graphData.nodes, ...graphData.edges],
        style: [
            {
                selector: 'node',
                style: {
                    'label': function(ele) {
                        if (ele.data('type') === 'Symbol') {
                            if (ele.data('category') == 'BASE') {
                                return ele.data('class')
                            }
                            var label;
                            var concept = ele.data('concept');
                            var value = ele.data('value');
                            var dose = ele.data('dose');
                            var timing = ele.data('timing');
                            var route = ele.data('route');

                            if (concept) {
                                label = concept["concept_name"];
                            } else {
                                label = ele.data('class');
                            }

                            if (value) {
                                label += " " + value;
                            }
                            if (dose) {
                                label += "\n" + dose;
                            }
                            if (timing) {
                                label += "\n" + timing;
                            }
                            if (route) {
                                label += "\n[" + route + "]";
                            }
                            return label;

                        } else if (ele.data('type').startsWith('Temporal') && ele.data('type').endsWith('Count')) {
                            var label = ele.data('class');
                            var intervalLabel = "";

                            if (ele.data('interval_type')) {
                                intervalLabel = ele.data('interval_type');
                            } else if (ele.data('start_time') && ele.data('end_time')) {
                                intervalLabel = ele.data('start_time') + " - " + ele.data('end_time');
                            } else if (ele.data('interval_criterion')) {
                                intervalLabel = ele.data('interval_criterion');
                            }

                            label += "[" + checkCount(ele) + "; " + intervalLabel + "]";
                            return label;
                        } else if (ele.data('type').endsWith('Count')) {
                            return ele.data('class') + '[' + checkCount(ele) + ']';
                        }


                        if (ele.data("is_sink")) {
                            return ele.data('category') + " [SINK]"
                        }
                        return ele.data('class')
                    },
                    'background-color': function(ele) {
                        return nodeColors[ele.data('category')] || '#666'; // Assign color based on 'type', with a default
                    },
                    'shape': function(ele) {
                        return nodeShapes[ele.data("is_atom") ? "Symbol" : ele.data('type')] || 'star'; // Assign color based on 'type', with a default
                    },
                    'text-valign': 'center',
                    'color': '#000000',
                    'width': function(ele) {
                        return ele.data('is_atom') ? '120px': '40px';
                    },
                    'height': function(ele) {
                        return ele.data('is_atom') ? '80px': '40px';
                    },
                    'font-size': '10px',
                    'text-wrap': 'wrap',
                    'text-max-width': '120px' // Adjust width as needed
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'target-arrow-shape': 'triangle', // Set arrow shape to triangle
                    'curve-style': 'bezier' // Makes the edge curved for better visibility of direction
                }
            }
        ],
        layout: {
            name: 'klay', // Use 'klay' layout for better visualization
            nodeDimensionsIncludeLabels: true,
            fit: true,
            padding: 20,
            animate: true,
            animationDuration: 500,
            klay: {
                spacing: 20,
                direction: 'DOWN',
            }
        }
    });

    // Add event listener for node click
    cy.on('tap', 'node', function(evt) {
        hideTippys(cy);
        const node = evt.target;
        if (!node.tippy) {
            node.tippy = createTippy(node);
        }
        node.tippy.show();
    });

    // Hide popper when clicking on the canvas
    cy.on('tap', function(evt) {
        if (evt.target === cy) {
            hideTippys(cy);
        }
    });
}

function createTippy(node) {
    let content = '';

    function formatData(data, prefix = '') {
        for (let key in data) {
            if (data.hasOwnProperty(key)) {
                if (Array.isArray(data[key])) {
                    content += `<strong>${prefix}${key}:</strong><br>`;
                    data[key].forEach((item, index) => {
                        content += `<strong>${prefix}${key}[${index}]:</strong><br>`;
                        formatData(item, prefix + '&nbsp;&nbsp;&nbsp;');
                    });
                } else if (typeof data[key] === 'object' && data[key] !== null) {
                    content += `<strong>${prefix}${key}:</strong><br>`;
                    formatData(data[key], prefix + '&nbsp;&nbsp;&nbsp;');
                } else {
                    content += `<strong>${prefix}${key}:</strong> ${data[key]}<br>`;
                }
            }
        }
    }

    formatData(node.data());

    let ref = node.popperRef(); // used only for positioning
    let dummyDomEle = document.createElement('div');
    document.body.appendChild(dummyDomEle); // Ensure dummyDomEle has a parent

    return tippy(dummyDomEle, {
        content: () => {
            let div = document.createElement('div');
            div.innerHTML = content;
            return div;
        },
        placement: 'top',
        hideOnClick: true,
        interactive: true,
        trigger: 'manual',
        allowHTML: true,
        getReferenceClientRect: () => ref.getBoundingClientRect()
    });
}

function hideTippys(cy) {
    cy.elements().forEach(ele => {
        if (ele.tippy) {
            ele.tippy.hide();
        }
    });
}

loadRecommendations();
