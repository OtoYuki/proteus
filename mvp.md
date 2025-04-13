# Proteus Application - Intermediate Product Definition

## 1. Project Goal

**Overall Goal:** Develop Proteus, a Django-based web application providing a semi-automated computational pipeline for synthetic protein design, integrating structure prediction (AlphaFold/ColabFold), validation (Molecular Dynamics simulations with GROMACS), and ML-based ranking for industrial and environmental applications.

**Intermediate Product Goal:** To create a comprehensive application incorporating the full database schema provided. This version allows authenticated users (with basic roles) to submit protein sequences, receive 3D structure predictions (ColabFold), view predictions (Mol\*Star), request and view results from MD simulations (GROMACS) and ML ranking models, while tracking jobs, logs, and system metrics.

## 2. Core Intermediate Features

1. **User Authentication & Roles:**
    * Implement the custom Django `User` model using email for authentication, linked to the `Role` model (`role_id`, `role_name`).
    * Provide user registration (assigning default 'User' role), login, and logout.
    * Implement basic role differentiation (e.g., 'User', 'Admin').
    * Ensure secure password handling.
    * Restrict actions based on authentication and potentially roles.

2. **Protein Sequence Submission & Management:**
    * Utilize the `ProteinSequence` model fully: Allow users to submit sequences (FASTA), provide a `sequence_name`, `description`, `organism`, `source`. Calculate and store `sequence_length`. Track `upload_date` and `status`.
    * Allow users to view, manage, and potentially delete their submitted sequences.

3. **Structure Prediction Workflow (ColabFold):**
    * Integrate ColabFold for structure prediction, triggered by sequence submission or user request.
    * Use Celery and the `JobQueue` model (`job_id`, `user`, `job_type`='Prediction', `status`, `created_at`, `started_at`, `completed_at`, `job_parameters`, `priority`) to manage asynchronous prediction tasks.
    * Run ColabFold within a Docker container.
    * Store results in the `Prediction` model (`prediction_id`, `sequence` (FK), `pdb_file_path`, `pae_file_path`, `plddt_score`, `prediction_date`, `status`, `confidence_score`, `model_version`, `prediction_metadata`).

4. **Molecular Dynamics Simulation Workflow (GROMACS):**
    * Allow authenticated users to request an MD simulation for a completed `Prediction`.
    * Use Celery and `JobQueue` (`job_type`='Simulation') to manage asynchronous simulation tasks.
    * Run GROMACS steps (topology, solvation, minimization, equilibration, production run) within a Docker container based on predefined parameters (store parameters in `JobQueue.job_parameters` or `ValidationMetric.simulation_parameters`).
    * Store simulation results in the `ValidationMetric` model (`metric_id`, `prediction` (FK), `rmsd`, `rg`, `energy`, `trajectory_path`, `validation_date`, `status`, `simulation_parameters`, `stability_score`, `validation_notes`, `modified_date`).

5. **ML Ranking Workflow:**
    * Allow authenticated users to request ML-based ranking for a completed `Prediction` (potentially triggered automatically after prediction or simulation).
    * Use Celery and `JobQueue` (`job_type`='Ranking') to manage asynchronous ranking tasks.
    * Assume an ML model exists (implementation details TBD) that takes prediction/validation data as input.
    * Store ranking results in the `MLRanking` model (`ranking_id`, `prediction` (FK), `stability_score`, `solubility_score`, `binding_efficiency_score`, `ranking_date`, `overall_score`, `feature_importance`, `model_version`, `ranking_notes`).

6. **Job & Result Management/Visualization:**
    * Provide views for users to track the status of their jobs (`Prediction`, `Simulation`, `Ranking`) via the `JobQueue` model.
    * Create detail views for `Prediction` results:
        * Display key prediction info (`plddt_score`, `confidence_score`, metadata).
        * Embed Mol\*Star viewer to interactively display the 3D protein structure (PDB).
        * Link to associated `ValidationMetric` and `MLRanking` results.
        * Allow download of PDB, PAE files.
    * Create views to display `ValidationMetric` results (RMSD, Rg, energy, stability score, notes). Allow download of trajectory files if applicable.
    * Create views to display `MLRanking` results (scores, notes).

7. **System Logging:**
    * Implement comprehensive logging using the `Log` model.
    * Log key user actions (login, logout, submission, job requests) and system events (job status changes, errors).
    * Capture `user` (FK), `action`, `timestamp`, `details`, `ip_address`, `session_id`, `status`, `component`.

8. **System Metrics Monitoring:**
    * Implement a background task (e.g., periodic Celery task) to collect system metrics.
    * Store metrics (CPU, memory, disk usage, active jobs count, other performance data) in the `SystemMetric` model (`metric_id`, `timestamp`, `cpu_usage`, `memory_usage`, `disk_usage`, `active_jobs`, `performance_metrics`, `status`).

## 3. Technology Stack

* **Backend Framework:** Django 4.2
* **Programming Language:** Python 3.12
* **Database:** PostgreSQL
* **Task Queue:** Celery with RabbitMQ broker
* **Containerization:** Docker
* **Prediction Tool:** ColabFold
* **Simulation Tool:** GROMACS
* **ML Tool:** (To be integrated - specific library TBD)
* **Frontend:** HTML, CSS, JavaScript
* **3D Visualization:** Mol\*Star

## 4. Database Schema (Models)

Utilize the full schema as defined in `models.py`:

* **`Role`**: Defines user roles (e.g., "User", "Admin"). Fields: `role_id`, `role_name`.
* **`User`**: Custom user model using email. Fields: `user_id` (UUID PK), `role` (FK), `email`, `password`, `is_staff`, `is_superuser`, `created_at`, `is_active`, `last_login`. Uses `UserManager`.
* **`ProteinSequence`**: Stores user-submitted sequences. Fields: `sequence_id` (UUID PK), `user` (FK), `sequence_name`, `sequence_fasta`, `upload_date`, `status`, `description`, `sequence_length`, `organism`, `source`.
* **`Prediction`**: Stores ColabFold prediction results. Fields: `prediction_id` (UUID PK), `sequence` (FK), `pdb_file_path`, `pae_file_path`, `plddt_score`, `prediction_date`, `status`, `confidence_score`, `model_version`, `prediction_metadata` (JSON).
* **`ValidationMetric`**: Stores GROMACS simulation results/metrics. Fields: `metric_id` (UUID PK), `prediction` (FK), `rmsd`, `rg`, `energy`, `trajectory_path`, `validation_date`, `status`, `simulation_parameters` (JSON), `stability_score`, `validation_notes`, `modified_date`.
* **`MLRanking`**: Stores ML model ranking results. Fields: `ranking_id` (UUID PK), `prediction` (FK), `stability_score`, `solubility_score`, `binding_efficiency_score`, `ranking_date`, `overall_score`, `feature_importance` (JSON), `model_version`, `ranking_notes`.
* **`Log`**: Records system and user activity. Fields: `log_id` (UUID PK), `user` (FK, nullable), `action`, `timestamp`, `details`, `ip_address`, `session_id`, `status`, `component`.
* **`SystemMetric`**: Stores system performance data. Fields: `metric_id` (UUID PK), `timestamp`, `cpu_usage`, `memory_usage`, `disk_usage`, `active_jobs`, `performance_metrics` (JSON), `status`.
* **`JobQueue`**: Tracks asynchronous tasks. Fields: `job_id` (UUID PK), `user` (FK), `job_type`, `created_at`, `started_at`, `completed_at`, `status`, `job_parameters` (JSON), `priority`.

## 5. Core UI/UX Guidelines

* **Branding:** Strictly adhere to the Proteus Brand Kit (Logo, Colors: `#314528` accent, `#000000` text, `#555555` secondary text, `#bec6c3` background, `#ffffff` panels; Typography: 'TAN Ashford' headers, 'Open Sans' body). Ensure AA contrast.
* **Design Elements:** Apply Glassmorphism; use simple line icons (`#314528`); maintain 8px grid spacing.
* **Layout:** Clean, professional, intuitive for scientific users.
* **Forms:** Clearly labeled, validated input fields.
* **Visualization:** Integrate Mol\*Star for interactive 3D structure viewing. Clearly display prediction, validation, and ranking scores/metrics where available.
* **Job Tracking:** Provide clear status updates and links to results for user-submitted jobs.

## 6. Deployment & Setup Notes

* Containerize using Docker and `docker-compose`.
* Provide Dockerfiles for Django app, Celery worker(s), potentially GROMACS/ColabFold runners if not managed directly by Celery tasks calling Docker API.
* Configure volume mounts for persistent data (PostgreSQL data, prediction/simulation output files). File paths (`/home/sire/colabfold_data/...`) should be configurable.
* Requires Python 3.12, PostgreSQL, RabbitMQ.
* Setup instructions for ColabFold and GROMACS (likely within Docker).
* Ensure appropriate resource allocation for containers (memory, CPU, potentially GPU for ColabFold/ML).

## 7. Exclusions from this Version

While this version is more comprehensive, some aspects might still be simplified or excluded:

* **Advanced ML Model:** The specific ML model logic might be basic initially.
* **Sophisticated Admin Interface:** Rely primarily on the default Django admin, potentially with minor customizations.
* **Real-time Updates:** Job status updates might rely on page refreshes rather than WebSockets initially.
* **Public API:** No external API access.
* **Detailed System Metric Analysis:** Focus on collection; visualization might be basic.
