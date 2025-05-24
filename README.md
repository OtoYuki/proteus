# Proteus: Integrated Computational Pipeline for Synthetic Protein Design

**Proteus is a sophisticated, Django-based web application that provides a semi-automated computational pipeline for synthetic protein design. It seamlessly integrates state-of-the-art tools for protein structure prediction, molecular dynamics simulation, and machine learning-based ranking into a unified, accessible platform.**

This project aims to streamline the traditionally complex and fragmented workflow of computational protein design, empowering researchers and scientists to accelerate discovery in synthetic biology.

## Overview

The primary goal of Proteus is to address the challenge of efficiently managing the multi-step computational workflow involved in designing and evaluating synthetic proteins. Traditional methods often require manual execution of disparate tools and cumbersome data transfer, hindering productivity and accessibility. Proteus solves this by:

* **Integrating** key computational biology tools into a cohesive pipeline.
* **Automating** job submission, execution, and data management.
* **Providing** an intuitive web-based interface for users.

**Target Users:** Researchers and professionals in synthetic biology, computational biology, bioinformatics, and related fields who require robust tools for designing, predicting, and validating protein structures.

## Key Features & Specialties

Proteus stands out due to its comprehensive and integrated approach:

* **🧬 End-to-End Integrated Workflow:** A complete pipeline from protein sequence submission to structure prediction (ColabFold), validation via molecular dynamics (GROMACS), and ML-based ranking.
* **⚙️ Automated Job Management:** Leverages Celery and a persistent Job Queue ([`core/models.py`](core/models.py)) to manage the lifecycle of computationally intensive tasks (prediction, simulation, ranking) asynchronously, ensuring the web application remains responsive. See [`core/tasks.py`](core/tasks.py).
* **📊 Integrated Data Management:** Utilizes a PostgreSQL database to store user data, sequences, prediction results (PDB files, scores), simulation metrics (RMSD, Rg, energy), ML rankings, system logs, and performance metrics in a structured and interconnected manner.
* **🖥️ Web-Based Accessibility:** A user-friendly interface built with Django ([`core/templates/`](core/templates/)) lowers the barrier to entry, making powerful computational tools accessible to a broader range of scientific users.
* **🐳 Reproducibility & Consistency:** Employs Docker containers to run core computational tools (ColabFold, GROMACS), ensuring reproducible results and simplifying dependency management across different environments. Execution logic is detailed in [`core/tasks.py`](core/tasks.py).
* **🔬 Interactive 3D Visualization:** Integrates Mol\*Star for interactive visualization of predicted protein structures directly within the web browser.
* **🤖 ML-Based Ranking:** Incorporates a machine learning component ([`MLRanking` model in `core/models.py`](core/models.py)) to rank protein candidates based on a variety of predicted and simulated properties, aiding in decision-making.
* **📈 System Monitoring:** Includes capabilities for collecting and storing system metrics (CPU, memory, disk usage, active jobs) during computational tasks using `psutil` ([`SystemMetric` model and usage in `core/tasks.py`](core/tasks.py)).

## Technology Stack

Proteus is built with a modern and robust technology stack:

* **Backend:** Python 3.12, Django 4.2.19
  * Main application logic resides in the [`core`](core/) app.
* **Database:** PostgreSQL (interfaced via `psycopg2-binary==2.9.10`)
  * Schema defined in [`core/models.py`](core/models.py).
  * Database configuration in [`proteus/settings.py`](proteus/settings.py).
* **Task Queue:** Celery 5.4.0 with RabbitMQ (as the message broker, using `amqp==5.3.1`, `kombu==5.4.2`, `vine==5.1.0`, `billiard==4.2.1`)
  * Asynchronous tasks defined in [`core/tasks.py`](core/tasks.py).
* **Containerization:** Docker
  * Used for running ColabFold and GROMACS, managed via `subprocess` calls in Celery tasks.
* **Computational Tools:**
  * **Protein Structure Prediction:** ColabFold (leveraging AlphaFold models)
  * **Molecular Dynamics Simulation:** GROMACS
* **Frontend:** HTML, CSS, JavaScript
  * Templates located in [`core/templates/`](core/templates/).
* **3D Visualization:** Mol\*Star
* **System Monitoring:** `psutil` Python library
* **Core Python Libraries:** `numpy`, `pandas`, `scipy`, `biopython` for data handling and scientific computation.
* **Web & API:** `django-cors-headers==4.7.0` for Cross-Origin Resource Sharing.

A full list of Python dependencies is available in [`requirements.txt`](requirements.txt).

## Architecture

The architecture of Proteus is designed for scalability, reliability, and ease of use:

* **Asynchronous Pipeline Architecture:** The core computational pipeline (prediction, simulation, ranking) is implemented using Celery. User requests trigger Django views ([`core/views.py`](core/views.py)), which in turn dispatch tasks to Celery workers. This non-blocking approach ensures the web interface remains responsive even during long-running computations.
* **Dockerized Execution Environment:** Computationally intensive tools like ColabFold and GROMACS are executed within Docker containers. Celery tasks in [`core/tasks.py`](core/tasks.py) dynamically manage these containers, ensuring that the tools run in isolated, consistent, and dependency-managed environments.
* **Database-Driven State Management:** The status, parameters, and results of each stage in the pipeline are meticulously tracked and stored in the PostgreSQL database. Django models such as [`JobQueue`](core/models.py), [`Prediction`](core/models.py), [`ValidationMetric`](core/models.py), and [`MLRanking`](core/models.py) provide a structured way to manage and query this data, enabling robust workflow orchestration and data provenance.
* **Modular Django Application:** The project follows Django's app-based structure, with the primary functionality encapsulated within the `core` application. This promotes modularity and maintainability.

## Core Components

* **`proteus/` Project Directory ([`proteus/`](proteus/)):**
  * [`settings.py`](proteus/settings.py): Django project settings, database configuration, installed apps, middleware, etc.
  * [`urls.py`](proteus/urls.py): Root URL configurations, routing requests to the appropriate views.
  * [`celery.py`](proteus/celery.py): Celery application definition and configuration.
* **`core/` Application Directory ([`core/`](core/)):**
  * [`models.py`](core/models.py): Defines the database schema for all entities (User, Role, ProteinSequence, Prediction, ValidationMetric, MLRanking, JobQueue, Log, SystemMetric).
  * [`views.py`](core/views.py): Contains the logic for handling HTTP requests, interacting with models, and rendering HTML templates. Implements views for sequence submission, listing, detail views, etc.
  * [`forms.py`](core/forms.py): Defines Django forms for user input, such as protein sequence submission.
  * [`tasks.py`](core/tasks.py): Contains Celery shared tasks for executing ColabFold predictions and GROMACS simulations, including Docker management, file handling, and database updates.
  * [`admin.py`](core/admin.py): Configures the Django admin interface for managing application data.
  * [`templates/core/`](core/templates/core/): HTML templates that define the user interface.
  * [`static/`](core/static/): Static files (CSS, JavaScript, images - Mol\*Star integration would be here).
  * [`migrations/`](core/migrations/): Database migration files generated by Django.
* **[`requirements.txt`](requirements.txt):** Specifies all Python package dependencies for the project.

## Setup and Installation

1. **Prerequisites:**
    * Python 3.12
    * PostgreSQL server
    * RabbitMQ server (or another Celery-compatible broker)
    * Docker
2. **Clone the repository:**

    ```bash
    git clone https://github.com/OtoYuki/proteus.git
    cd proteus
    ```

3. **Set up Python environment and install dependencies:**
    * It is highly recommended to use a virtual environment (e.g., `venv` or `conda`).
    * Install all project dependencies using pip and the provided `requirements.txt` file:

        ```bash
        pip install -r requirements.txt
        ```

4. **Configure Django settings:**
    * Update `proteus/settings.py` with your database credentials, secret key, Celery broker URL, and other environment-specific settings.
5. **Apply database migrations:**

    ```bash
    python manage.py migrate
    ```

6. **Create a superuser (for admin access):**

    ```bash
    python manage.py createsuperuser
    ```

7. **Start Celery worker(s):**

    ```bash
    celery -A proteus worker -l info
    ```

    (Ensure RabbitMQ is running and accessible as configured in `proteus/settings.py` or `proteus/celery.py`)
8. **Start the Django development server:**

    ```bash
    python manage.py runserver
    ```

9. **Ensure Docker is running and can pull necessary images** (e.g., `ghcr.io/sokrypton/colabfold:1.5.5`, a GROMACS image). The specific images and tags are defined within the execution logic in [`core/tasks.py`](core/tasks.py).

## Usage Workflow

1. **User Authentication:** Users register and log in to the Proteus platform.
2. **Sequence Submission:** Users submit protein sequences via a web form ([`ProteinSequenceForm`](core/forms.py) rendered by [`ProteinSequenceCreateView`](core/views.py)).
3. **Job Queuing:** Upon submission, a job is created in the [`JobQueue`](core/models.py) and picked up by a Celery worker for processing.
4. **Prediction:** The `run_colabfold` task in [`core/tasks.py`](core/tasks.py) executes ColabFold in a Docker container to predict the protein structure. Results (PDB file, scores) are saved to the [`Prediction`](core/models.py) model.
5. **Validation (Simulation):** If configured, the `run_gromacs_simulation` task in [`core/tasks.py`](core/tasks.py) runs GROMACS simulations on the predicted structure, also within Docker. Metrics (RMSD, Rg, energy) are stored in the [`ValidationMetric`](core/models.py) model.
6. **ML Ranking:** An ML ranking task (triggered after prediction) evaluates the prediction and simulation data, storing scores in the [`MLRanking`](core/models.py) model.
7. **Status Tracking & Results:** Users can monitor the status of their jobs and view detailed results, including 3D structure visualizations, prediction scores, and simulation metrics, through the web interface ([`ProteinSequenceListView`](core/views.py), [`ProteinSequenceDetailView`](core/views.py)).

## Challenges Overcome

The development of Proteus addressed several significant challenges inherent in computational biology workflows:

* **Managing Long-Running Computations:** Solved by implementing an asynchronous task queue with Celery, preventing the web server from blocking and ensuring a responsive user experience.
* **Complex Dependency Management:** Addressed by containerizing ColabFold and GROMACS using Docker, which encapsulates dependencies and ensures consistent execution environments.
* **Computational Resource Management:** Basic resource monitoring is implemented using `psutil` within tasks, and GROMACS tasks can be configured with `nice` for CPU priority. Docker resource limits are also utilized.
* **Workflow Orchestration:** Achieved through carefully designed Celery tasks in [`core/tasks.py`](core/tasks.py) that manage sequential execution, data handoff between steps, and persistent state tracking in the database.
* **Error Handling and Logging:** Robust `try-except` blocks and detailed logging to both the console and the [`Log`](core/models.py) database model are implemented throughout [`core/tasks.py`](core/tasks.py) to ensure issues are captured and diagnosable.

## Impact and Outcomes

Proteus is expected to deliver significant benefits:

* **A Functional, Web-Accessible Platform:** Providing a user-friendly application for the entire synthetic protein design pipeline.
* **An Automated Computational Pipeline:** Reducing manual effort and the potential for errors in complex multi-step analyses.
* **Centralized and Structured Data Storage:** Ensuring data provenance and facilitating easier analysis and comparison of results.
* **Accelerated Research:** By streamlining the design-predict-validate-rank cycle, Proteus can significantly speed up the process of identifying promising protein candidates.
* **Increased Accessibility:** Making advanced computational tools available to a wider range of researchers, including those who may lack deep command-line or infrastructure management expertise.
* **Improved Decision Making:** Providing integrated data from prediction, simulation, and ML-ranking stages to help researchers make more informed decisions about which protein designs to pursue experimentally.

## Future Directions

* Integration of more sophisticated and customizable Machine Learning models for ranking.
* Advanced 3D visualization options and comparative analysis tools.
* Expansion of the supported computational tools (e.g., different docking software, alternative simulation engines).
* Enhanced system monitoring, resource allocation, and job prioritization features.
* User collaboration features and project management capabilities.
* Comprehensive API development for programmatic access to Proteus functionalities.

---

This README aims to provide a comprehensive overview of the Proteus project. For more specific details, please refer to the source code and any accompanying documentation.
