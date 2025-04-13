/**
 * Mol*Star initialization and configuration for Proteus
 * This script handles the initialization of the Mol*Star viewer and provides
 * functions for loading and displaying protein structures.
 */

// Define the class directly without the conditional check to ensure it's always available
class ProteusMolStar {
    constructor(elementId, options = {}) {
        this.elementId = elementId;
        this.plugin = null;
        this.options = {
            viewport: {
                showAnimation: true,
                showControls: true,
                showSelectionMode: true,
                background: { color: 'white' }
            },
            layout: {
                initial: {
                    isExpanded: false,
                    showControls: true,
                    showRemoteState: false,
                    showSequence: true,
                    showLog: false,
                    showLeftPanel: true
                }
            },
            components: {
                remoteState: 'none'
            },
            ...options
        };
        this.isLoaded = false;
    }

    async init() {
        try {
            console.log('Initializing Mol*Star viewer...');

            // Check if molstar is available
            if (typeof molstar === 'undefined') {
                console.error('Mol*Star library not loaded. Please check your script imports.');
                this.showError('Mol*Star library not loaded. Please check your console for more details.');
                return false;
            }

            // Create Mol*Star viewer with custom settings
            this.plugin = await molstar.createPluginUI(document.getElementById(this.elementId), this.options);

            console.log('Mol*Star viewer initialized successfully.');
            this.isLoaded = true;
            return true;
        } catch (error) {
            console.error('Error initializing Mol*Star viewer:', error);
            this.showError(`Could not initialize the molecular viewer: ${error.message}`);
            return false;
        }
    }

    async loadStructureFromUrl(url, format = 'pdb', options = {}) {
        if (!this.isLoaded) {
            console.error('Mol*Star viewer not initialized. Call init() first.');
            return false;
        }

        try {
            console.log(`Loading structure from URL: ${url}`);

            // Load the structure
            const data = await this.plugin.builders.data.download({ url }, { state: { isGhost: true } });
            const trajectory = await this.plugin.builders.structure.parseTrajectory(data, format);

            // Apply representation
            await this.plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default', {
                structure: {
                    name: options.name || 'Structure'
                }
            });

            console.log('Structure loaded successfully.');
            return trajectory;
        } catch (error) {
            console.error('Error loading structure:', error);
            this.showError(`Error loading structure: ${error.message}`);
            return false;
        }
    }

    async updateRepresentation(representation = 'cartoon', colorScheme = 'polymer-id') {
        if (!this.isLoaded) {
            console.error('Mol*Star viewer not initialized. Call init() first.');
            return false;
        }

        try {
            const state = this.plugin.managers.structure.hierarchy.current;
            if (!state) {
                console.warn('No structure loaded to update representation.');
                return false;
            }

            // Remove existing representations
            await this.plugin.builders.structure.component.removeAll();

            // Add new representation with selected coloring
            await this.plugin.builders.structure.representation.addRepresentation(state.components[0], {
                type: representation,
                color: colorScheme
            });

            return true;
        } catch (error) {
            console.error('Error updating representation:', error);
            return false;
        }
    }

    getStructureStats() {
        if (!this.isLoaded) {
            console.error('Mol*Star viewer not initialized. Call init() first.');
            return null;
        }

        try {
            const state = this.plugin.state.data;
            const model = this.plugin.managers.structure.hierarchy.current?.models[0];

            if (model) {
                return {
                    atomCount: model.atomCount,
                    residueCount: model.residueCount,
                    chainCount: model.chains.length
                };
            }
            return null;
        } catch (error) {
            console.error('Error getting structure statistics:', error);
            return null;
        }
    }

    showError(message) {
        const element = document.getElementById(this.elementId);
        if (element) {
            element.innerHTML = `
                <div class="flex items-center justify-center h-full">
                    <div class="text-center p-4">
                        <p class="font-bold text-error">${message}</p>
                    </div>
                </div>
            `;
        }
    }

    dispose() {
        if (this.plugin) {
            this.plugin.dispose();
            this.plugin = null;
            this.isLoaded = false;
        }
    }
}

// Make sure we explicitly define it on the window object
window.ProteusMolStar = ProteusMolStar;