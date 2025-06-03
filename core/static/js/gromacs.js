/**
 * GROMACS Simulation Management for Proteus
 */
class GromacsSimulation {
    constructor(predictionId) {
        this.predictionId = predictionId;
        this.simulationBtn = document.getElementById('runSimulation');
        this.statusArea = null;
        this.pollInterval = null;

        this.initialize();
    }

    initialize() {
        if (this.simulationBtn) {
            this.simulationBtn.addEventListener('click', this.startSimulation.bind(this));
        }
        // Check for existing simulations on load
        this.checkExistingSimulation();
    }

    startSimulation() {
        if (confirm('Are you sure you want to run a GROMACS simulation? This may take several minutes.')) {
            // Disable button and show loading state
            this.simulationBtn.disabled = true;
            this.simulationBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...';

            // Make AJAX call to start simulation
            fetch(`/predictions/${this.predictionId}/simulate/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': this.getCsrfToken(),
                }
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        this.simulationBtn.innerHTML = 'Simulation Started';
                        // Set up polling for status updates
                        this.startStatusPolling();
                    } else {
                        this.simulationBtn.innerHTML = 'Run GROMACS Simulation';
                        this.simulationBtn.disabled = false;
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    this.simulationBtn.innerHTML = 'Run GROMACS Simulation';
                    this.simulationBtn.disabled = false;
                    alert('An error occurred while starting the simulation.');
                });
        }
    }

    startStatusPolling() {
        // Create status display area if it doesn't exist
        this.createStatusArea();

        // Poll every 10 seconds
        this.pollInterval = setInterval(() => {
            fetch(`/predictions/${this.predictionId}/simulation_status/`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        this.updateStatusDisplay(data);
                    }
                })
                .catch(error => {
                    console.error('Error checking simulation status:', error);
                });
        }, 10000);
    }

    createStatusArea() {
        if (!this.statusArea) {
            const viewerParent = document.getElementById('viewer').parentElement;
            this.statusArea = document.createElement('div');
            this.statusArea.id = 'simulationStatus';
            this.statusArea.className = 'alert alert-info mt-3';
            this.statusArea.innerHTML = '<strong>Simulation Status:</strong> Starting...';
            viewerParent.appendChild(this.statusArea);
        }
    }

    updateStatusDisplay(data) {
        const statusArea = document.getElementById('simulationStatus');
        if (!statusArea) return;

        let statusHtml = '';
        let statusClass = '';

        // Create metrics section if available
        let metricsHtml = '';
        if (data.metrics && Object.keys(data.metrics).length > 0) {
            const metrics = data.metrics;
            metricsHtml = `
                <div class="simulation-metrics">
                    ${metrics.rmsd !== undefined ? `
                        <div class="metric-card">
                            <div class="metric-value">${metrics.rmsd.toFixed(3)}</div>
                            <div class="metric-label">RMSD (nm)</div>
                        </div>
                    ` : ''}
                    ${metrics.rg !== undefined ? `
                        <div class="metric-card">
                            <div class="metric-value">${metrics.rg.toFixed(3)}</div>
                            <div class="metric-label">Radius of Gyration (nm)</div>
                        </div>
                    ` : ''}
                    ${metrics.energy !== undefined ? `
                        <div class="metric-card">
                            <div class="metric-value">${(metrics.energy / 1000).toFixed(1)}k</div>
                            <div class="metric-label">Potential Energy (kJ/mol)</div>
                        </div>
                    ` : ''}
                    ${metrics.stability_score !== undefined ? `
                        <div class="metric-card">
                            <div class="metric-value">${metrics.stability_score.toFixed(1)}%</div>
                            <div class="metric-label">Stability Score</div>
                        </div>
                    ` : ''}
                </div>
            `;
        }

        // Create activity log section if available
        let activityHtml = '';
        if (data.recent_logs && data.recent_logs.length > 0) {
            activityHtml = `
                <div class="mt-4">
                    <h4 class="font-semibold text-sm text-base-content/70 mb-2">Recent Activity</h4>
                    <div class="activity-log">
                        ${data.recent_logs.map(log => `<div>${log}</div>`).join('')}
                    </div>
                </div>
            `;
        }

        if (data.simulation_status === 'completed') {
            clearInterval(this.pollInterval);

            statusHtml = `
                <div class="simulation-status-card">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-3">
                            <span class="status-badge completed">
                                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                    <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                                </svg>
                                Simulation Completed
                            </span>
                        </div>
                        <div class="text-sm text-base-content/60">
                            ${data.completion_time ? new Date(data.completion_time).toLocaleString() : ''}
                        </div>
                    </div>
                    
                    ${metricsHtml}
                    
                    <div class="action-buttons">
                        <a href="/download_trajectory/${this.predictionId}/" class="btn btn-outline btn-sm">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                            Download Data
                        </a>
                    </div>
                    
                    ${activityHtml}
                </div>
            `;
        } else if (data.simulation_status === 'running') {
            statusHtml = `
                <div class="simulation-status-card">
                    <div class="flex items-center gap-3">
                        <span class="status-badge running">
                            <span class="loading loading-spinner loading-sm"></span>
                            Simulation Running
                        </span>
                    </div>
                    
                    ${data.progress ? `
                        <div class="mt-4">
                            <div class="flex justify-between text-sm mb-2">
                                <span>Progress</span>
                                <span>${data.progress.percentage?.toFixed(1) || 0}%</span>
                            </div>
                            <div class="w-full bg-base-300 rounded-full h-2">
                                <div class="bg-primary h-2 rounded-full transition-all duration-300" 
                                     style="width: ${data.progress.percentage || 0}%"></div>
                            </div>
                            ${data.progress.current_step ? `
                                <div class="text-sm text-base-content/70 mt-2">
                                    Current step: ${data.progress.current_step}
                                </div>
                            ` : ''}
                        </div>
                    ` : ''}
                    
                    ${activityHtml}
                </div>
            `;
        } else if (data.simulation_status === 'failed') {
            clearInterval(this.pollInterval);

            statusHtml = `
                <div class="simulation-status-card">
                    <div class="flex items-center gap-3">
                        <span class="status-badge failed">
                            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                            </svg>
                            Simulation Failed
                        </span>
                    </div>
                    
                    ${data.notes ? `
                        <div class="mt-3 p-3 bg-error/10 border border-error/20 rounded-lg">
                            <div class="text-sm text-error">${data.notes}</div>
                        </div>
                    ` : ''}
                    
                    ${activityHtml}
                    
                    <div class="action-buttons">
                        <button onclick="window.location.reload()" class="btn btn-outline btn-sm">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            Retry
                        </button>
                    </div>
                </div>
            `;
        } else {
            // Default/pending state
            statusHtml = `
                <div class="flex items-center gap-3">
                    <span class="loading loading-spinner loading-sm text-primary"></span>
                    <span class="text-base-content/70">${data.simulation_status || 'Checking simulation status...'}</span>
                </div>
            `;
        }

        statusArea.innerHTML = statusHtml;
    }

    checkExistingSimulation() {
        fetch(`/predictions/${this.predictionId}/simulation_status/`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.simulation_status) {
                    // If there's an active simulation, start the polling
                    if (data.simulation_status === 'running' || data.simulation_status === 'pending') {
                        if (this.simulationBtn) {
                            this.simulationBtn.disabled = true;
                            this.simulationBtn.innerHTML = 'Simulation In Progress';
                        }
                        this.startStatusPolling();
                    }
                    // If there's a completed simulation, show the results
                    else if (data.simulation_status === 'completed') {
                        this.createStatusArea();
                        this.statusArea.className = 'alert alert-success mt-3';
                        this.statusArea.innerHTML = `
                            <strong>Simulation Status:</strong> Completed
                            <div class="mt-2">
                                <a href="/static/viewer.html?trajectory=${this.predictionId}" class="btn btn-sm btn-success">View Results</a>
                                <a href="/download_trajectory/${this.predictionId}/" class="btn btn-sm btn-outline-secondary">Download Trajectory</a>
                            </div>
                        `;
                    }
                }
            })
            .catch(error => {
                console.error('Error checking existing simulation:', error);
            });
    }

    getCsrfToken() {
        // Get CSRF token from cookie
        const name = 'csrftoken';
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
}

// Initialize the GROMACS simulation functionality when the DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    const predictionIdElement = document.getElementById('prediction-id');
    if (predictionIdElement) {
        const predictionId = predictionIdElement.dataset.id;
        if (predictionId) {
            new GromacsSimulation(predictionId);
        }
    }
});