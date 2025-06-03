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

        // Poll every 5 seconds for more responsive updates
        this.pollInterval = setInterval(() => {
            fetch(`/predictions/${this.predictionId}/simulation_status_realtime/`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        this.updateStatusDisplay(data);
                    }
                })
                .catch(error => {
                    console.error('Error checking simulation status:', error);
                });
        }, 5000);
    }

    createStatusArea() {
        if (!this.statusArea) {
            // Find the existing simulationStatus div instead of creating a new one
            this.statusArea = document.getElementById('simulationStatus');
            if (!this.statusArea) {
                // Fallback: create if it doesn't exist
                const viewerParent = document.getElementById('viewer').parentElement;
                this.statusArea = document.createElement('div');
                this.statusArea.id = 'simulationStatus';
                this.statusArea.className = 'py-2';
                viewerParent.appendChild(this.statusArea);
            }
        }
    }

    updateStatusDisplay(data) {
        if (!this.statusArea) return;

        // Get status info
        const status = data.simulation_status;
        const progress = data.progress || {};
        const recentLogs = data.recent_logs || [];

        // Create status badge
        let statusBadgeHtml = this.createStatusBadge(status);

        // Create progress bar if available
        let progressHtml = '';
        if (progress.percentage !== undefined) {
            progressHtml = `
                <div class="w-full bg-gray-200 rounded-full h-2.5 mb-3">
                    <div class="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
                         style="width: ${progress.percentage}%"></div>
                </div>
                <div class="flex justify-between text-sm text-gray-600 mb-2">
                    <span>${progress.current_step || 'Processing'}</span>
                    <span>${progress.percentage.toFixed(1)}%</span>
                </div>
                ${progress.eta ? `<div class="text-xs text-gray-500 mb-3">ETA: ${progress.eta}</div>` : ''}
            `;
        }

        // Create metrics section
        let metricsHtml = '';
        if (status === 'completed' && data.metrics) {
            metricsHtml = `
                <div class="simulation-metrics">
                    ${data.metrics.duration ? `
                        <div class="metric-card">
                            <div class="metric-value">${data.metrics.duration}</div>
                            <div class="metric-label">Duration</div>
                        </div>
                    ` : ''}
                    ${data.metrics.frames ? `
                        <div class="metric-card">
                            <div class="metric-value">${data.metrics.frames}</div>
                            <div class="metric-label">Frames</div>
                        </div>
                    ` : ''}
                    ${data.metrics.temperature ? `
                        <div class="metric-card">
                            <div class="metric-value">${data.metrics.temperature}K</div>
                            <div class="metric-label">Temperature</div>
                        </div>
                    ` : ''}
                </div>
            `;
        }

        // Create action buttons
        let actionsHtml = '';
        if (status === 'completed') {
            clearInterval(this.pollInterval);
            actionsHtml = `
                <div class="action-buttons">
                    ${data.trajectory_path ? `
                        <a href="/static/viewer.html?trajectory=${this.predictionId}" 
                           class="btn btn-primary btn-sm">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
                            </svg>
                            View 3D Results
                        </a>
                        <a href="/download_trajectory/${this.predictionId}/" 
                           class="btn btn-outline btn-sm">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                            Download PyMOL Files
                        </a>
                    ` : ''}
                </div>
            `;
        } else if (status === 'failed') {
            clearInterval(this.pollInterval);
        }

        // Create activity log
        let activityHtml = '';
        if (recentLogs.length > 0) {
            activityHtml = `
                <div class="mt-4">
                    <h4 class="text-sm font-semibold text-gray-700 mb-2">Recent Activity</h4>
                    <div class="activity-log">
                        ${recentLogs.map(log => `<div>${this.escapeHtml(log)}</div>`).join('')}
                    </div>
                </div>
            `;
        }

        // Combine all elements
        this.statusArea.innerHTML = `
            <div class="simulation-status-card">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-lg font-semibold text-gray-800">Molecular Dynamics Simulation</h3>
                    ${statusBadgeHtml}
                </div>
                ${progressHtml}
                ${status === 'failed' && data.notes ? `
                    <div class="bg-red-50 border border-red-200 rounded-lg p-3 mb-3">
                        <div class="text-red-800 text-sm">
                            <strong>Error:</strong> ${this.escapeHtml(data.notes)}
                        </div>
                    </div>
                ` : ''}
                ${metricsHtml}
                ${actionsHtml}
                ${activityHtml}
            </div>
        `;
    }

    createStatusBadge(status) {
        const statusConfig = {
            'pending': {
                class: 'status-badge running',
                icon: '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>',
                text: 'Pending'
            },
            'running': {
                class: 'status-badge running',
                icon: '<svg class="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>',
                text: 'Running'
            },
            'completed': {
                class: 'status-badge completed',
                icon: '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>',
                text: 'Completed'
            },
            'failed': {
                class: 'status-badge failed',
                icon: '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.866-.833-2.535 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"></path></svg>',
                text: 'Failed'
            }
        };

        const config = statusConfig[status] || statusConfig['pending'];
        return `
            <div class="${config.class}">
                ${config.icon}
                <span>${config.text}</span>
            </div>
        `;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    checkExistingSimulation() {
        fetch(`/predictions/${this.predictionId}/simulation_status_realtime/`)
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
                        this.updateStatusDisplay(data);
                    }
                    // If simulation failed, show the failure
                    else if (data.simulation_status === 'failed') {
                        this.createStatusArea();
                        this.updateStatusDisplay(data);
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