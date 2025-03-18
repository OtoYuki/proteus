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
        let statusClass = 'alert-info';
        let statusText = data.simulation_status;

        if (data.simulation_status === 'completed') {
            statusClass = 'alert-success';
            clearInterval(this.pollInterval);

            // Add link to download trajectory if available
            if (data.trajectory_path) {
                this.statusArea.innerHTML = `
                    <strong>Simulation Status:</strong> Completed
                    <div class="mt-2">
                        <a href="/static/viewer.html?trajectory=${this.predictionId}" class="btn btn-sm btn-success">View Results</a>
                        <a href="/download_trajectory/${this.predictionId}/" class="btn btn-sm btn-outline-secondary">Download Trajectory</a>
                    </div>
                `;
            }
        } else if (data.simulation_status === 'failed') {
            statusClass = 'alert-danger';
            statusText = 'Failed: ' + (data.notes || 'Unknown error');
            clearInterval(this.pollInterval);
        }

        this.statusArea.className = `alert ${statusClass} mt-3`;
        if (data.simulation_status !== 'completed') { // Don't override if completed (we did that above)
            this.statusArea.innerHTML = `<strong>Simulation Status:</strong> ${statusText}`;
        }
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