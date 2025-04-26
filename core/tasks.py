from datetime import timezone as tz  # Renamed to avoid conflict
import datetime
from celery import shared_task
import subprocess
from .models import JobQueue, Log, MLRanking, Prediction, SystemMetric, ValidationMetric
import os
import logging
import time
import glob
import socket
import json
import psutil  # Ensuring psutil is imported directly
import shutil  # For file operations
import tempfile  # For creating temporary directories

logger = logging.getLogger(__name__)

# Resource limits for computation
MAX_CPU_PERCENT = 85.0
MAX_MEMORY_PERCENT = 85.0


# ColabFold prediction task
@shared_task
def run_colabfold(prediction_id):
    """
    Run a ColabFold prediction for a given prediction ID.
    """
    logger.info(f"Starting ColabFold prediction for prediction_id: {prediction_id}")

    # Get or create a job entry for this prediction
    job = None
    try:
        # First, create or update the job entry
        try:
            job = JobQueue.objects.filter(
                job_parameters__prediction_id=str(prediction_id),
                job_type="colabfold_prediction",
            ).first()

            if not job:
                prediction = Prediction.objects.get(prediction_id=prediction_id)
                user = prediction.sequence.user
                job = JobQueue.objects.create(
                    user=user,
                    job_type="colabfold_prediction",
                    status="running",
                    started_at=tz.now(),
                    job_parameters={
                        "prediction_id": str(prediction_id),
                        "component": "prediction_engine",
                        "priority": 1,
                    },
                    priority=1,  # ColabFold predictions get high priority
                )
                logger.info(f"Created new job for prediction: {prediction_id}")
            else:
                job.status = "running"
                job.started_at = tz.now()
                job.save()
                logger.info(f"Updated existing job for prediction: {prediction_id}")

            # Create initial system metric entry
            try:
                SystemMetric.objects.create(
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    disk_usage=psutil.disk_usage("/").percent,
                    active_jobs=JobQueue.objects.filter(status="running").count(),
                    status="normal",
                    performance_metrics={
                        "prediction_id": str(prediction_id),
                        "action": "prediction_started",
                        "job_id": str(job.job_id),
                    },
                )
            except Exception as e:
                logger.warning(f"Could not create system metrics: {e}")

            # Create initial log entry
            Log.objects.create(
                user=user,
                action="colabfold_prediction_started",
                details=f"Starting ColabFold prediction {prediction_id}",
                status="info",
                component="prediction_engine",
                session_id=job.job_id,
            )

        except Exception as e:
            logger.error(f"Error in job management: {e}")

        # Check if Celery worker is running, start one if not
        try:
            # Check for running Celery workers
            ps_cmd = "ps aux | grep 'celery worker' | grep -v grep | wc -l"
            ps_result = subprocess.run(
                ps_cmd, shell=True, capture_output=True, text=True
            )
            worker_count = int(ps_result.stdout.strip())

            if worker_count == 0:
                logger.warning("No Celery workers detected. Starting a worker...")
                # Start Celery worker as a background process
                worker_cmd = (
                    "cd /home/sire/proteus && "
                    "nohup celery -A proteus worker --loglevel=info "
                    "> /home/sire/celery_worker.log 2>&1 &"
                )
                subprocess.run(worker_cmd, shell=True)
                logger.info("Started new Celery worker in background")
                time.sleep(5)  # Give it a moment to start up
        except Exception as e:
            logger.error(f"Error checking/starting Celery worker: {e}")

        # Get or create a job entry for this prediction
        job = None

        try:
            # First, create or update the job entry
            try:
                # Find existing job or create new one
                job = JobQueue.objects.filter(
                    job_parameters__prediction_id=str(prediction_id),
                    job_type="colabfold_prediction",
                ).first()

                if not job:
                    # Get prediction to find the user
                    prediction = Prediction.objects.get(prediction_id=prediction_id)
                    user = prediction.sequence.user

                    # Create new job entry
                    job = JobQueue.objects.create(
                        user=user,
                        job_type="colabfold_prediction",
                        status="running",
                        started_at=tz.now(),
                        job_parameters={"prediction_id": str(prediction_id)},
                    )
                    logger.info(f"Created new job for prediction: {prediction_id}")
                else:
                    # Update existing job
                    job.status = "running"
                    job.started_at = tz.now()
                    job.save()
                    logger.info(f"Updated existing job for prediction: {prediction_id}")
            except Exception as e:
                logger.error(f"Error creating/updating job entry: {e}")
                # Continue with prediction even if job tracking fails

            # Create a log entry for the prediction start
            try:
                prediction = Prediction.objects.get(prediction_id=prediction_id)
                Log.objects.create(
                    user=prediction.sequence.user,
                    action="colabfold_prediction_started",
                    details=f"Started ColabFold prediction for {prediction.sequence.sequence_name}",
                    status="success",
                    component="prediction_engine",
                )
            except Exception as e:
                logger.error(f"Error creating log entry: {e}")

            # Record system metrics (if possible)
            try:
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage("/").percent
                active_jobs = JobQueue.objects.filter(status="running").count()

                SystemMetric.objects.create(
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    disk_usage=disk_usage,
                    active_jobs=active_jobs,
                    status="normal",
                    performance_metrics={
                        "prediction_id": str(prediction_id),
                        "action": "prediction_started",
                    },
                )
                logger.info(f"Recorded system metrics at prediction start")
            except Exception as e:
                logger.error(f"Error recording system metrics: {e}")

            # Retrieve prediction object
            prediction = Prediction.objects.get(prediction_id=prediction_id)
            sequence = prediction.sequence
            logger.info(f"Retrieved sequence: {sequence.sequence_name}")

            # Define paths
            host_data_dir = "/home/sire/colabfold_data"
            host_fasta_path = f"{host_data_dir}/predictions/{prediction_id}/input.fasta"
            host_output_dir = f"{host_data_dir}/predictions/{prediction_id}/output"
            container_data_dir = "/data"
            container_fasta_path = (
                f"{container_data_dir}/predictions/{prediction_id}/input.fasta"
            )
            container_output_dir = (
                f"{container_data_dir}/predictions/{prediction_id}/output"
            )

            logger.info(
                f"Host FASTA path: {host_fasta_path}, Host output dir: {host_output_dir}"
            )
            logger.info(
                f"Container FASTA path: {container_fasta_path}, Container output dir: {container_output_dir}"
            )

            # Create directories
            os.makedirs(os.path.dirname(host_fasta_path), exist_ok=True)
            os.makedirs(host_output_dir, exist_ok=True)
            logger.info("Directories created or already exist")

            # Write FASTA file
            with open(host_fasta_path, "w") as f:
                f.write(f">{sequence.sequence_name}\n{sequence.sequence_fasta}")
            logger.info("FASTA file written")

            # Update status to running
            prediction.status = "running"
            sequence.status = "running"  # Update sequence status to running
            prediction.save()
            sequence.save()
            logger.info("Prediction and sequence status updated to running")

            # Check and manage ColabFold container
            check_exists_cmd = "docker ps -aq -f name=colabfold"
            exists_result = subprocess.run(
                check_exists_cmd, shell=True, capture_output=True, text=True
            )
            container_id = exists_result.stdout.strip()

            if container_id:
                check_running_cmd = "docker ps -q -f name=colabfold"
                running_result = subprocess.run(
                    check_running_cmd, shell=True, capture_output=True, text=True
                )
                if not running_result.stdout.strip():
                    logger.info(
                        "ColabFold container exists but is not running. Starting it..."
                    )
                    subprocess.run(
                        f"docker start {container_id}", shell=True, check=True
                    )
                    logger.info("ColabFold container started")
                    time.sleep(5)  # Wait for container to stabilize
            else:
                logger.info(
                    "ColabFold container does not exist. Creating it with more memory..."
                )
                start_cmd = (
                    "docker run -d --name colabfold --gpus all "
                    "--restart unless-stopped --memory=16g --memory-swap=20g --shm-size=8g "
                    "-v /home/sire/colabfold_data:/data ghcr.io/sokrypton/colabfold:1.5.5"
                )
                subprocess.run(start_cmd, shell=True, check=True)
                logger.info("Started ColabFold container with increased memory")
                time.sleep(5)

            # Run ColabFold
            start_time = time.time()
            cmd = (
                f"docker exec -t colabfold /bin/bash -c "
                f"'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 && "
                f"source /opt/conda/etc/profile.d/conda.sh && "
                f"conda activate /localcolabfold/colabfold-conda && "
                f"colabfold_batch --num-models 1 --max-seq 256 --max-extra-seq 512 "
                f"--disable-unified-memory --use-gpu-relax "
                f"{container_fasta_path} {container_output_dir}'"
            )
            logger.info(f"Running command: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(
                f"ColabFold prediction completed in {execution_time:.2f} seconds"
            )

            # Extra sleep to allow file system sync (especially on mounted volumes)
            time.sleep(10)

            # Find specific output files
            pdb_files = glob.glob(
                os.path.join(host_output_dir, "*unrelaxed_rank_001*seed_000.pdb")
            )
            if not pdb_files:
                # Fallback to any PDB
                pdb_files = glob.glob(os.path.join(host_output_dir, "*.pdb"))

            pae_files = glob.glob(
                os.path.join(host_output_dir, "*predicted_aligned_error*.json")
            )
            scores_files = glob.glob(
                os.path.join(host_output_dir, "*scores_rank_001*.json")
            )
            config_file = os.path.join(host_output_dir, "config.json")

            logger.info(f"Found PDB files: {pdb_files}")
            logger.info(f"Found PAE files: {pae_files}")
            logger.info(f"Found scores files: {scores_files}")

            pdb_file_path = pdb_files[0] if pdb_files else None
            pae_file_path = pae_files[0] if pae_files else None
            scores_file_path = scores_files[0] if scores_files else None

            # Extract data from scores file
            plddt_score = None
            confidence_score = None
            metadata = {}

            if scores_file_path and os.path.exists(scores_file_path):
                try:
                    with open(scores_file_path, "r") as f:
                        scores_data = json.load(f)
                        logger.info(f"Scores data: {scores_data}")

                        # Extract pLDDT score (mean plddt)
                        if "plddt" in scores_data:
                            plddt_values = scores_data["plddt"]
                            if isinstance(plddt_values, list):
                                plddt_score = sum(plddt_values) / len(plddt_values)
                            else:
                                plddt_score = plddt_values
                            logger.info(f"Extracted pLDDT score: {plddt_score}")

                        # Use plddt as confidence score as well
                        confidence_score = plddt_score

                        # Store all scores as metadata
                        metadata.update({"scores": scores_data})
                except Exception as e:
                    logger.error(f"Error extracting scores: {e}")

            # Extract config data
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config_data = json.load(f)
                        logger.info(f"Config data: {config_data}")

                        # Extract model version
                        model_version = config_data.get("model_name", "colabfold-1.5.5")

                        # Add to metadata
                        metadata.update({"config": config_data})
                except Exception as e:
                    logger.error(f"Error extracting config: {e}")
                    model_version = "colabfold-1.5.5"  # Default if extraction fails
            else:
                model_version = "colabfold-1.5.5"  # Default if file doesn't exist

            # If we couldn't find PDB files through glob, try the previous approach
            if not pdb_file_path:
                # Poll for the PDB file using glob for a more robust file search
                timeout = 120  # seconds
                poll_interval = 5  # seconds
                elapsed = 0

                while elapsed < timeout:
                    pdb_files = glob.glob(os.path.join(host_output_dir, "*.pdb"))
                    logger.info(f"Polling at {elapsed}s, found pdb_files: {pdb_files}")
                    if pdb_files:
                        pdb_file_path = pdb_files[0]
                        logger.info(f"Found PDB file: {pdb_file_path}")
                        break
                    time.sleep(poll_interval)
                    elapsed += poll_interval

                if not pdb_file_path:
                    # Fallback: use the find command
                    find_cmd = f"find {host_output_dir} -type f -name '*.pdb'"
                    logger.info(f"Trying find command: {find_cmd}")
                    find_result = subprocess.run(
                        find_cmd, shell=True, capture_output=True, text=True
                    )
                    logger.info(f"Find command output: '{find_result.stdout}'")
                    found_files = [
                        f for f in find_result.stdout.strip().split("\n") if f
                    ]
                    if found_files:
                        pdb_file_path = found_files[0]
                        logger.info(
                            f"Found PDB file using find command: {pdb_file_path}"
                        )

            if pdb_file_path:
                # Update all fields in the prediction
                prediction.pdb_file_path = pdb_file_path
                prediction.status = "completed"

                if pae_file_path:
                    prediction.pae_file_path = pae_file_path
                    logger.info(f"Updated PAE file path: {pae_file_path}")

                if plddt_score is not None:
                    prediction.plddt_score = plddt_score
                    logger.info(f"Updated pLDDT score: {plddt_score}")

                if confidence_score is not None:
                    prediction.confidence_score = confidence_score
                    logger.info(f"Updated confidence score: {confidence_score}")

                prediction.model_version = model_version
                logger.info(f"Updated model version: {model_version}")

                # Add additional metadata
                metadata.update(
                    {
                        "sequence_name": sequence.sequence_name,
                        "sequence_length": len(
                            sequence.sequence_fasta.replace("\n", "")
                        ),
                        "completion_time": datetime.datetime.now().isoformat(),
                        "execution_time_seconds": execution_time,
                        "host_machine": (
                            socket.gethostname()
                            if hasattr(socket, "gethostname")
                            else "unknown"
                        ),
                    }
                )
                prediction.prediction_metadata = metadata
                logger.info(f"Updated prediction metadata")

                prediction.save()

                # IMPORTANT FIX: Update ProteinSequence status to completed
                sequence.status = "completed"
                sequence.save()
                logger.info(
                    f"Updated sequence {sequence.sequence_id} status to completed"
                )

                logger.info(
                    f"Prediction status updated to completed with all fields populated. PDB path: {pdb_file_path}"
                )

                # Update job status
                if job:
                    job.status = "completed"
                    job.completed_at = tz.now()
                    job.save()
                    logger.info(f"Updated job status to completed")

                # Add basic validation metrics based on PDB structure
                try:
                    # Simple PDB parser to count atoms and residues
                    atom_count = 0
                    residue_count = 0

                    with open(pdb_file_path, "r") as pdb_file:
                        prev_res_num = None
                        for line in pdb_file:
                            if line.startswith("ATOM"):
                                atom_count += 1
                                # Extract residue number (positions 22-26 in PDB format)
                                try:
                                    res_num = int(line[22:26].strip())
                                    if prev_res_num != res_num:
                                        residue_count += 1
                                        prev_res_num = res_num
                                except (ValueError, IndexError):
                                    pass

                    # Calculate approximate radius of gyration (very simplistic estimation)
                    approx_rg = 1.2 * pow(
                        residue_count, 1 / 3
                    )  # Very rough approximation

                    # Create validation entry with basic structure info
                    ValidationMetric.objects.create(
                        prediction=prediction,
                        status="initial",
                        rg=approx_rg,
                        stability_score=plddt_score if plddt_score else None,
                        validation_notes=f"Initial structure metrics: {atom_count} atoms, {residue_count} residues",
                        simulation_parameters={
                            "atom_count": atom_count,
                            "residue_count": residue_count,
                        },
                    )
                    logger.info(
                        f"Created initial validation metrics with {atom_count} atoms and {residue_count} residues"
                    )
                except Exception as e:
                    logger.error(f"Error creating initial validation metrics: {e}")
                    # Non-critical error, continue execution

                # Record final system metrics
                try:
                    SystemMetric.objects.create(
                        cpu_usage=psutil.cpu_percent(),
                        memory_usage=psutil.virtual_memory().percent,
                        disk_usage=psutil.disk_usage("/").percent,
                        active_jobs=JobQueue.objects.filter(status="running").count(),
                        status="normal",
                        performance_metrics={
                            "prediction_id": str(prediction_id),
                            "action": "prediction_completed",
                            "execution_time": execution_time,
                        },
                    )
                    logger.info(f"Recorded final system metrics")
                except Exception as e:
                    logger.error(f"Error recording final system metrics: {e}")

                # Create a log entry for successful completion
                try:
                    Log.objects.create(
                        user=prediction.sequence.user,
                        action="colabfold_prediction_completed",
                        details=f"Completed ColabFold prediction for {prediction.sequence.sequence_name}"
                        + (
                            f" with pLDDT {plddt_score:.2f}"
                            if plddt_score is not None
                            else ""
                        ),
                        status="success",
                        component="prediction_engine",
                    )
                    logger.info(f"Created success log entry")
                except Exception as e:
                    logger.error(f"Error creating completion log entry: {e}")

                # After successful prediction, initiate ML ranking
                if prediction.status == "completed":
                    try:
                        # Create ML Ranking job
                        ranking_job = JobQueue.objects.create(
                            user=prediction.sequence.user,
                            job_type="ml_ranking",
                            status="pending",
                            job_parameters={
                                "prediction_id": str(prediction_id),
                                "priority": 2,
                            },
                            priority=2,  # Lower priority than predictions
                        )

                        # Create initial MLRanking entry
                        MLRanking.objects.create(
                            prediction=prediction,
                            model_version="1.0",  # Update with your actual ML model version
                            ranking_notes="Initiated after successful prediction",
                        )

                        logger.info(
                            f"Created ML ranking job for prediction {prediction_id}"
                        )
                    except Exception as e:
                        logger.error(f"Error creating ML ranking job: {e}")

                return f"Prediction {prediction_id} completed successfully with enhanced data"
            else:
                # Handle failure due to missing PDB file
                error_msg = f"No PDB files found in {host_output_dir} after waiting for {timeout} seconds"

                prediction.status = "failed"
                prediction.save()

                # IMPORTANT FIX: Update ProteinSequence status to failed
                sequence.status = "failed"
                sequence.save()
                logger.info(f"Updated sequence {sequence.sequence_id} status to failed")

                if job:
                    job.status = "failed"
                    job.completed_at = tz.now()
                    job.save()

                # Log the failure
                try:
                    Log.objects.create(
                        user=prediction.sequence.user,
                        action="colabfold_prediction_failed",
                        details=error_msg,
                        status="error",
                        component="prediction_engine",
                    )
                except Exception as e:
                    logger.error(f"Error creating failure log entry: {e}")

                raise Exception(error_msg)

        except Prediction.DoesNotExist:
            error_msg = f"Prediction with id {prediction_id} does not exist"
            logger.error(error_msg)

            # Update job status if exists
            if job:
                job.status = "failed"
                job.completed_at = tz.now()
                job.save()

            # Create error log
            try:
                Log.objects.create(
                    action="colabfold_prediction_error",
                    details=error_msg,
                    status="error",
                    component="prediction_engine",
                )
            except Exception as e:
                logger.error(f"Error creating log entry: {e}")

            return f"Error: {error_msg}"

        except subprocess.CalledProcessError as e:
            error_msg = f"Command execution failed: {e}"
            logger.error(error_msg)

            try:
                prediction = Prediction.objects.get(prediction_id=prediction_id)
                prediction.status = "failed"
                prediction.save()

                # IMPORTANT FIX: Update ProteinSequence status to failed
                sequence = prediction.sequence
                sequence.status = "failed"
                sequence.save()
                logger.info(f"Updated sequence {sequence.sequence_id} status to failed")

                logger.info(
                    "Prediction status updated to failed due to subprocess error"
                )

                # Update job status
                if job:
                    job.status = "failed"
                    job.completed_at = tz.now()
                    job.save()

                # Log the failure
                Log.objects.create(
                    user=prediction.sequence.user,
                    action="colabfold_prediction_failed",
                    details=f"Command execution failed: {e.stderr if hasattr(e, 'stderr') else str(e)}",
                    status="error",
                    component="prediction_engine",
                )
            except Exception as inner_e:
                logger.error(f"Failed to update status records: {inner_e}")

            return f"Error: Command execution failed for prediction {prediction_id}"

        except Exception as e:
            error_msg = f"Error in run_colabfold: {e}"
            logger.error(error_msg)

            try:
                prediction = Prediction.objects.get(prediction_id=prediction_id)
                prediction.status = "failed"
                prediction.save()

                # IMPORTANT FIX: Update ProteinSequence status to failed
                sequence = prediction.sequence
                sequence.status = "failed"
                sequence.save()
                logger.info(f"Updated sequence {sequence.sequence_id} status to failed")

                logger.info("Prediction status updated to failed")

                # Update job status
                if job:
                    job.status = "failed"
                    job.completed_at = tz.now()
                    job.save()

                # Log the failure
                Log.objects.create(
                    user=prediction.sequence.user,
                    action="colabfold_prediction_failed",
                    details=f"Error: {str(e)}",
                    status="error",
                    component="prediction_engine",
                )
            except Exception as inner_e:
                logger.error(f"Failed to update status records: {inner_e}")

            return f"Error: {str(e)}"

    except Exception as e:
        error_msg = f"Error in prediction task: {str(e)}"
        logger.error(error_msg)

        if job:
            job.status = "failed"
            job.completed_at = tz.now()
            job.save()

        try:
            Log.objects.create(
                user=prediction.sequence.user if prediction else None,
                action="colabfold_prediction_failed",
                details=error_msg,
                status="error",
                component="prediction_engine",
                session_id=job.job_id if job else None,
            )
        except Exception as log_error:
            logger.error(f"Error creating failure log: {log_error}")

        raise Exception(error_msg)


# GROMACS Task
@shared_task
def run_gromacs_simulation(prediction_id):
    """
    Run a GROMACS simulation for a given prediction derived from ColabFold.
    This improved implementation:
    - Ensures GPU acceleration via CUDA when available
    - Manages resource monitoring to stay within 85% CPU/memory limits
    - Implements proper database logging and updates
    - Creates clean working directories and maintains trajectory files
    - Calculates essential structural metrics (RMSD, Rg, energy)
    """
    job = None
    prediction = None
    validation_metric = None
    sim_dir = None

    try:
        logger.info(f"Starting GROMACS simulation for prediction: {prediction_id}")

        # Get prediction object first to avoid repeated lookups
        try:
            prediction = Prediction.objects.get(prediction_id=prediction_id)
        except Prediction.DoesNotExist:
            error_msg = f"Prediction with ID {prediction_id} does not exist"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Check if input PDB exists
        if not prediction.pdb_file_path or not os.path.exists(prediction.pdb_file_path):
            error_msg = f"PDB file does not exist at {prediction.pdb_file_path}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Create or update job entry
        try:
            # Find existing job or create new one
            job = JobQueue.objects.filter(
                job_parameters__prediction_id=str(prediction_id),
                job_type="gromacs_simulation",
            ).first()

            if not job:
                job = JobQueue.objects.create(
                    user=prediction.sequence.user,
                    job_type="gromacs_simulation",
                    status="running",
                    started_at=datetime.datetime.now(tz.utc),
                    job_parameters={
                        "prediction_id": str(prediction_id),
                        "component": "simulation_engine",
                        "priority": 3,  # Lower priority than predictions and rankings
                    },
                    priority=3,
                )
                logger.info(
                    f"Created new simulation job for prediction: {prediction_id}"
                )
            else:
                job.status = "running"
                job.started_at = datetime.datetime.now(tz.utc)
                job.save()
                logger.info(
                    f"Updated existing simulation job for prediction: {prediction_id}"
                )

            # Record initial system metrics
            try:
                SystemMetric.objects.create(
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    disk_usage=psutil.disk_usage("/").percent,
                    active_jobs=JobQueue.objects.filter(status="running").count(),
                    status="normal",
                    performance_metrics={
                        "prediction_id": str(prediction_id),
                        "action": "simulation_started",
                        "job_id": str(job.job_id),
                    },
                )
            except Exception as e:
                logger.warning(f"Could not create system metrics: {e}")

            # Create initial log entry
            Log.objects.create(
                user=prediction.sequence.user,
                action="gromacs_simulation_started",
                details=f"Starting GROMACS simulation for prediction {prediction_id}",
                status="info",
                component="simulation_engine",
                session_id=job.job_id,
            )
        except Exception as e:
            logger.error(f"Error in job management: {e}")

        # Check existing simulations to avoid duplicates
        existing_sim = ValidationMetric.objects.filter(
            prediction_id=prediction_id, status__in=["running", "pending"]
        ).first()

        if existing_sim:
            # Check for stalled simulations
            time_threshold = datetime.datetime.now(tz.utc) - datetime.timedelta(
                minutes=30
            )
            if existing_sim.validation_date < time_threshold:
                existing_sim.status = "failed"
                existing_sim.validation_notes = f"{existing_sim.validation_notes or ''}\nSimulation timed out after 30 minutes of inactivity"
                existing_sim.save()
                logger.warning(
                    f"Found stale simulation for {prediction_id}, marking as failed"
                )
            else:
                logger.warning(f"A simulation is already running for {prediction_id}")
                return f"A simulation is already running for prediction {prediction_id}"

        # Create a ValidationMetric entry for this simulation
        validation_metric = ValidationMetric.objects.create(
            prediction=prediction,
            status="pending",
            validation_notes="Preparing GROMACS simulation environment",
        )
        logger.info(f"Created validation metric entry {validation_metric.metric_id}")

        # Check for GPU availability by checking nvidia-smi command
        has_gpu = False
        try:
            gpu_check = subprocess.run(
                "nvidia-smi", shell=True, capture_output=True, text=True
            )
            if gpu_check.returncode == 0:
                has_gpu = True
                logger.info("NVIDIA GPU detected, will use GPU acceleration")
            else:
                logger.warning(
                    "No NVIDIA GPU detected, will proceed with CPU-only mode"
                )
        except:
            logger.warning("Could not check for GPU, assuming CPU-only mode")

        # Check if GROMACS Docker container is available
        try:
            # Check for CUDA-enabled GROMACS container if GPU is available
            container_name = "gromacs_gpu" if has_gpu else "gromacs"
            check_gromacs_cmd = (
                f"docker ps -a | grep {container_name} || echo 'not_found'"
            )
            container_result = subprocess.run(
                check_gromacs_cmd, shell=True, capture_output=True, text=True
            )

            if "not_found" in container_result.stdout:
                logger.info(
                    f"GROMACS container not found, creating {container_name} container..."
                )
                if has_gpu:
                    # Create CUDA-enabled GROMACS container
                    create_container_cmd = (
                        "docker run -d --name gromacs_gpu --gpus all "
                        "--restart unless-stopped "
                        "-v /home/sire/gromacs_data:/data "
                        "nvcr.io/hpc/gromacs:2023"
                    )
                else:
                    # Create CPU-only GROMACS container
                    create_container_cmd = (
                        "docker run -d --name gromacs "
                        "--restart unless-stopped "
                        "-v /home/sire/gromacs_data:/data "
                        "gromacs/gromacs:latest"
                    )
                subprocess.run(create_container_cmd, shell=True, check=True)
                logger.info(f"Created {container_name} container")
            elif "Exited" in container_result.stdout:
                logger.info(
                    f"{container_name} container exists but is not running, starting it..."
                )
                start_container_cmd = f"docker start {container_name}"
                subprocess.run(start_container_cmd, shell=True, check=True)
                logger.info(f"Started {container_name} container")
        except Exception as e:
            logger.error(f"Error setting up GROMACS container: {e}")
            validation_metric.status = "failed"
            validation_metric.validation_notes = (
                f"Failed to set up GROMACS container: {str(e)}"
            )
            validation_metric.save()

            if job:
                job.status = "failed"
                job.completed_at = datetime.datetime.now(tz.utc)
                job.save()

            raise Exception(f"Failed to set up GROMACS container: {str(e)}")

        # Create a unique working directory for this simulation
        protein_name = f"protein_{prediction_id}"[:25].replace("-", "_")
        sim_dir = f"/home/sire/gromacs_data/{protein_name}"
        container_sim_dir = f"/data/{protein_name}"

        # Create directories and copy PDB file
        os.makedirs(sim_dir, exist_ok=True)
        pdb_destination = f"{sim_dir}/{protein_name}.pdb"
        shutil.copy2(prediction.pdb_file_path, pdb_destination)
        logger.info(f"Copied PDB file to {pdb_destination}")

        # Update validation metric status
        validation_metric.status = "running"
        validation_metric.validation_notes = "Setting up GROMACS simulation"
        validation_metric.save()

        logger.info("Preparing GROMACS simulation with controlled resource usage")

        # Define the GROMACS workflow steps with resource control and GPU acceleration
        # Each step is a shell command that will be executed in sequence
        nice_prefix = "nice -n 10"  # Lower CPU priority to avoid resource exhaustion
        container_name = "gromacs_gpu" if has_gpu else "gromacs"

        # GPU flag for mdrun
        gpu_flag = "-update gpu -bonded gpu -nb gpu" if has_gpu else ""

        steps = [
            # Step 1: Generate topology
            f"{nice_prefix} docker exec {container_name} bash -c 'cd {container_sim_dir} && "
            f"gmx pdb2gmx -f {protein_name}.pdb -o {protein_name}_processed.gro "
            f"-water spc -ter -ignh -ff amber03 -p {protein_name}.top <<< 0'",
            # Step 2: Define simulation box
            f"{nice_prefix} docker exec {container_name} bash -c 'cd {container_sim_dir} && "
            f"gmx editconf -f {protein_name}_processed.gro -o {protein_name}_box.gro "
            f"-c -d 1.0 -bt dodecahedron'",
            # Step 3: Solvate the box
            f"{nice_prefix} docker exec {container_name} bash -c 'cd {container_sim_dir} && "
            f"gmx solvate -cp {protein_name}_box.gro -cs spc216.gro "
            f"-o {protein_name}_solvated.gro -p {protein_name}.top'",
            # Step 4: Add ions for neutrality
            f"{nice_prefix} docker exec {container_name} bash -c 'cd {container_sim_dir} && "
            f"gmx grompp -f /data/ions.mdp -c {protein_name}_solvated.gro "
            f"-p {protein_name}.top -o {protein_name}_ions.tpr -maxwarn 2 && "
            f'echo "SOL" | gmx genion -s {protein_name}_ions.tpr -o {protein_name}_solv_ions.gro '
            f"-p {protein_name}.top -pname NA -nname CL -neutral'",
            # Step 5: Energy minimization (CPU-only as it's more stable for this step)
            f"{nice_prefix} docker exec {container_name} bash -c 'cd {container_sim_dir} && "
            f"gmx grompp -f /data/em.mdp -c {protein_name}_solv_ions.gro "
            f"-p {protein_name}.top -o {protein_name}_em.tpr -maxwarn 2 && "
            f"gmx mdrun -v -s {protein_name}_em.tpr -deffnm {protein_name}_em'",
            # Step 6: NVT equilibration (with GPU if available)
            f"{nice_prefix} docker exec {container_name} bash -c 'cd {container_sim_dir} && "
            f"gmx grompp -f /data/nvt.mdp -c {protein_name}_em.gro "
            f"-r {protein_name}_em.gro -p {protein_name}.top -o {protein_name}_nvt.tpr -maxwarn 2 && "
            f"gmx mdrun -v {gpu_flag} -s {protein_name}_nvt.tpr -deffnm {protein_name}_nvt'",
            # Step 7: NPT equilibration (with GPU if available)
            f"{nice_prefix} docker exec {container_name} bash -c 'cd {container_sim_dir} && "
            f"gmx grompp -f /data/npt.mdp -c {protein_name}_nvt.gro "
            f"-r {protein_name}_nvt.gro -t {protein_name}_nvt.cpt "
            f"-p {protein_name}.top -o {protein_name}_npt.tpr -maxwarn 2 && "
            f"gmx mdrun -v {gpu_flag} -s {protein_name}_npt.tpr -deffnm {protein_name}_npt'",
            # Step 8: Production MD (with GPU if available)
            f"{nice_prefix} docker exec {container_name} bash -c 'cd {container_sim_dir} && "
            f"gmx grompp -f /data/md.mdp -c {protein_name}_npt.gro "
            f"-t {protein_name}_npt.cpt -p {protein_name}.top -o {protein_name}_md.tpr -maxwarn 2 && "
            f"gmx mdrun -v {gpu_flag} -s {protein_name}_md.tpr -deffnm {protein_name}_md'",
        ]

        # Create MDP files with parameters optimized for GPU when available
        mdp_templates = {
            "ions.mdp": "; ions.mdp - for adding ions\nintegrator  = steep\nemtol = 1000.0\nnsteps = 50000",
            "em.mdp": """; em.mdp - energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstenergy   = 1000
; Bond constraints
constraints = h-bonds
constraint-algorithm = lincs
; Neighbor searching
cutoff-scheme = Verlet
nstlist     = 20
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2
pbc         = xyz
""",
            "nvt.mdp": f"""; nvt.mdp - NVT equilibration
define      = -DPOSRES
; Run control
integrator  = md
dt          = 0.002
nsteps      = 50000
nstenergy   = 500
; Output coordinates
nstxout     = 500
nstvout     = 500
nstfout     = 0
; Bond constraints
constraints = h-bonds
constraint-algorithm = lincs
; Neighbor searching
cutoff-scheme = Verlet
nstlist     = 20
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2
pbc         = xyz
; Temperature coupling
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300
; Pressure coupling
pcoupl      = no
; Velocity generation
gen_vel     = yes
gen_temp    = 300
gen_seed    = -1
""",
            "npt.mdp": f"""; npt.mdp - NPT equilibration
define      = -DPOSRES
; Run control
integrator  = md
dt          = 0.002
nsteps      = 50000
nstenergy   = 500
; Output control
nstxout     = 500
nstvout     = 500
nstfout     = 0
; Bond constraints
constraints = h-bonds
constraint-algorithm = lincs
; Neighbor searching
cutoff-scheme = Verlet
nstlist     = 20
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2
pbc         = xyz
; Temperature coupling
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300
; Pressure coupling
pcoupl      = Berendsen
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
""",
            "md.mdp": f"""; md.mdp - Production MD
; Run control
integrator  = md
dt          = 0.002
nsteps      = 100000
nstenergy   = 1000
; Output control
nstxout     = 1000
nstvout     = 1000
nstfout     = 0
; Bond constraints
constraints = h-bonds
constraint-algorithm = lincs
lincs-iter  = 1
lincs-order = 4
; Neighbor searching
cutoff-scheme = Verlet
nstlist     = 20
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2
pbc         = xyz
; Temperature coupling
tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300
; Pressure coupling
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
""",
        }

        # Create MDP files in simulation directory
        for mdp_name, mdp_content in mdp_templates.items():
            with open(f"{sim_dir}/{mdp_name}", "w") as mdp_file:
                mdp_file.write(mdp_content)

        logger.info(
            f"Created MDP template files for GROMACS simulation using {'GPU' if has_gpu else 'CPU'} mode"
        )

        # After each simulation step, update the validation metric with progress
        for i, step_cmd in enumerate(steps):
            # Check resource usage before each step
            current_cpu = psutil.cpu_percent()
            current_memory = psutil.virtual_memory().percent

            # If resource usage is high, pause briefly
            if current_cpu > MAX_CPU_PERCENT or current_memory > MAX_MEMORY_PERCENT:
                logger.warning(
                    f"Resource usage high before step {i+1}: CPU {current_cpu}%, "
                    f"Memory {current_memory}%. Pausing for 30 seconds."
                )
                time.sleep(30)  # Wait for resources to free up

            logger.info(
                f"Running GROMACS step {i+1}/{len(steps)} (CPU: {current_cpu}%, Memory: {current_memory}%)"
            )

            # Update validation metric with current step
            validation_metric.validation_notes = f"Running step {i+1}/{len(steps)}"
            validation_metric.modified_date = datetime.datetime.now(tz.utc)
            validation_metric.save()

            try:
                result = subprocess.run(
                    step_cmd, shell=True, capture_output=True, text=True, check=True
                )
                logger.info(f"Step {i+1} completed successfully")

                # Check for GPU usage in output if applicable
                if (
                    has_gpu
                    and "mdrun" in step_cmd
                    and "Using" in result.stdout
                    and "GPU" in result.stdout
                ):
                    logger.info("Confirmed GPU usage for this step")

                # Update validation metric with progress
                validation_metric.validation_notes = f"Completed step {i+1}/{len(steps)}: {result.stdout[-150:] if result.stdout else ''}"
                validation_metric.modified_date = datetime.datetime.now(tz.utc)
                validation_metric.save()

                # Create progress log
                Log.objects.create(
                    user=prediction.sequence.user,
                    action="gromacs_simulation_progress",
                    details=f"Completed step {i+1}/{len(steps)}",
                    status="info",
                    component="simulation_engine",
                    session_id=job.job_id,
                )

            except subprocess.CalledProcessError as e:
                error_msg = f"GROMACS step {i+1} failed: {e.stderr[:500] if e.stderr else str(e)}"
                logger.error(error_msg)

                validation_metric.status = "failed"
                validation_metric.validation_notes = (
                    f"{validation_metric.validation_notes}\n{error_msg}"
                )
                validation_metric.save()

                if job:
                    job.status = "failed"
                    job.completed_at = datetime.datetime.now(tz.utc)
                    job.save()

                Log.objects.create(
                    user=prediction.sequence.user,
                    action="gromacs_simulation_failed",
                    details=error_msg,
                    status="error",
                    component="simulation_engine",
                    session_id=job.job_id,
                )

                return f"Error in GROMACS simulation: {error_msg}"

        # Process and record final results
        try:
            logger.info("All GROMACS steps completed, processing results")
            trajectory_path = f"{sim_dir}/{protein_name}_md.xtc"

            if os.path.exists(trajectory_path):
                # Calculate final metrics
                metrics = {}
                try:
                    # Extract RMSD from trajectory
                    rmsd_cmd = f"""docker exec {container_name} bash -c '
                        cd {container_sim_dir}
                        echo "Protein" | gmx rms -s {protein_name}_md.tpr -f {protein_name}_md.xtc -o rmsd.xvg
                        chmod -R 755 .
                    '"""
                    subprocess.run(
                        rmsd_cmd, shell=True, check=True, capture_output=True
                    )

                    # Calculate radius of gyration
                    rg_cmd = f"""docker exec {container_name} bash -c '
                        cd {container_sim_dir}
                        echo "Protein" | gmx gyrate -s {protein_name}_md.tpr -f {protein_name}_md.xtc -o rg.xvg
                        chmod -R 755 .
                    '"""
                    subprocess.run(rg_cmd, shell=True, check=True, capture_output=True)

                    # Calculate potential energy
                    energy_cmd = f"""docker exec {container_name} bash -c '
                        cd {container_sim_dir}
                        echo "Potential" | gmx energy -s {protein_name}_md.tpr -f {protein_name}_md.edr -o energy.xvg
                        chmod -R 755 .
                    '"""
                    subprocess.run(
                        energy_cmd, shell=True, check=True, capture_output=True
                    )

                    # Read the last RMSD value
                    if os.path.exists(f"{sim_dir}/rmsd.xvg"):
                        with open(f"{sim_dir}/rmsd.xvg", "r") as f:
                            lines = [
                                line
                                for line in f.readlines()
                                if not line.startswith(("#", "@"))
                            ]
                            if lines:
                                last_rmsd = float(lines[-1].split()[1])
                                metrics["rmsd"] = last_rmsd
                                validation_metric.rmsd = last_rmsd

                    # Read the last Rg value
                    if os.path.exists(f"{sim_dir}/rg.xvg"):
                        with open(f"{sim_dir}/rg.xvg", "r") as f:
                            lines = [
                                line
                                for line in f.readlines()
                                if not line.startswith(("#", "@"))
                            ]
                            if lines:
                                last_rg = float(lines[-1].split()[1])
                                metrics["rg"] = last_rg
                                validation_metric.rg = last_rg

                    # Read the last energy value
                    if os.path.exists(f"{sim_dir}/energy.xvg"):
                        with open(f"{sim_dir}/energy.xvg", "r") as f:
                            lines = [
                                line
                                for line in f.readlines()
                                if not line.startswith(("#", "@"))
                            ]
                            if lines:
                                last_energy = float(lines[-1].split()[1])
                                metrics["energy"] = last_energy
                                validation_metric.energy = last_energy

                    # Calculate stability score (simple metric based on RMSD)
                    if "rmsd" in metrics:
                        stability_score = max(
                            0, 100 - (metrics["rmsd"] * 50)
                        )  # Lower RMSD = higher stability
                        validation_metric.stability_score = stability_score
                        metrics["stability_score"] = stability_score

                    logger.info(f"Calculated metrics: {metrics}")

                except Exception as e:
                    logger.error(f"Error calculating metrics: {e}")

                # Store simulation parameters
                validation_metric.simulation_parameters = {
                    "protein_name": protein_name,
                    "simulation_dir": sim_dir,
                    "trajectory_path": trajectory_path,
                    "metrics": metrics,
                    "simulation_length": "100 ps",  # Based on md.mdp parameters
                    "completion_time": datetime.datetime.now().isoformat(),
                    "gpu_accelerated": has_gpu,
                }

                validation_metric.trajectory_path = trajectory_path
                validation_metric.status = "completed"
                validation_metric.validation_notes = (
                    f"GROMACS simulation completed successfully using {'GPU' if has_gpu else 'CPU'} acceleration. "
                    f"Metrics: {str(metrics)}"
                )
                validation_metric.save()

                # Update job status
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.datetime.now(tz.utc)
                    job.save()

                # Record final system metrics
                try:
                    SystemMetric.objects.create(
                        cpu_usage=psutil.cpu_percent(),
                        memory_usage=psutil.virtual_memory().percent,
                        disk_usage=psutil.disk_usage("/").percent,
                        active_jobs=JobQueue.objects.filter(status="running").count(),
                        status="normal",
                        performance_metrics={
                            "prediction_id": str(prediction_id),
                            "action": "simulation_completed",
                            "job_id": str(job.job_id),
                            "metrics": metrics,
                            "gpu_used": has_gpu,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Could not create final system metrics: {e}")

                # Create completion log
                Log.objects.create(
                    user=prediction.sequence.user,
                    action="gromacs_simulation_completed",
                    details=(
                        f"GROMACS simulation completed successfully using {'GPU' if has_gpu else 'CPU'} acceleration. "
                        f"RMSD: {metrics.get('rmsd', 'N/A'):.3f} nm, "
                        f"Stability: {metrics.get('stability_score', 'N/A'):.1f}%"
                    ),
                    status="success",
                    component="simulation_engine",
                    session_id=job.job_id,
                )

                # Clean up unneeded files to save space but keep important results
                try:
                    # List of essential files to keep
                    essential_files = [
                        f"{protein_name}.pdb",
                        f"{protein_name}_md.xtc",  # Trajectory
                        f"{protein_name}_md.tpr",  # Run input
                        f"{protein_name}_md.gro",  # Final structure
                        "rmsd.xvg",
                        "rg.xvg",
                        "energy.xvg",
                    ]

                    # Move essential files to a results directory
                    results_dir = f"{sim_dir}/results"
                    os.makedirs(results_dir, exist_ok=True)

                    for filename in essential_files:
                        filepath = f"{sim_dir}/{filename}"
                        if os.path.exists(filepath):
                            shutil.copy2(filepath, f"{results_dir}/{filename}")

                    # Update validation metric to point to the saved trajectory
                    validation_metric.trajectory_path = (
                        f"{results_dir}/{protein_name}_md.xtc"
                    )
                    validation_metric.save()

                    logger.info(f"Saved essential simulation files to {results_dir}")

                except Exception as e:
                    logger.warning(f"Error during clean-up: {e}")

                return f"GROMACS simulation for prediction {prediction_id} completed successfully with calculated metrics using {'GPU' if has_gpu else 'CPU'} acceleration"

            else:
                error_msg = "Simulation completed but no trajectory file was generated"
                logger.warning(error_msg)
                validation_metric.status = "failed"
                validation_metric.validation_notes = error_msg
                validation_metric.save()

                if job:
                    job.status = "failed"
                    job.completed_at = datetime.datetime.now(tz.utc)
                    job.save()

                Log.objects.create(
                    user=prediction.sequence.user,
                    action="gromacs_simulation_failed",
                    details=error_msg,
                    status="error",
                    component="simulation_engine",
                    session_id=job.job_id,
                )

                return f"Error: {error_msg}"

        except Exception as e:
            error_msg = f"Error finalizing simulation results: {str(e)}"
            logger.error(error_msg)

            if validation_metric:
                validation_metric.status = "failed"
                validation_metric.validation_notes = (
                    f"{validation_metric.validation_notes or ''}\n{error_msg}"
                )
                validation_metric.save()

            if job:
                job.status = "failed"
                job.completed_at = datetime.datetime.now(tz.utc)
                job.save()

            Log.objects.create(
                user=prediction.sequence.user,
                action="gromacs_simulation_error",
                details=error_msg,
                status="error",
                component="simulation_engine",
                session_id=job.job_id if job else None,
            )

            return f"Error: {error_msg}"

    except Exception as e:
        error_msg = f"Error in GROMACS simulation: {str(e)}"
        logger.error(error_msg)

        # Handle metrics database updates on error
        try:
            if validation_metric:
                validation_metric.status = "failed"
                validation_metric.validation_notes = error_msg
                validation_metric.save()

            if job:
                job.status = "failed"
                job.completed_at = datetime.datetime.now(tz.utc)
                job.save()

            # Create an error log if possible
            if prediction:
                Log.objects.create(
                    user=prediction.sequence.user,
                    action="gromacs_simulation_error",
                    details=error_msg,
                    status="error",
                    component="simulation_engine",
                    session_id=job.job_id if job else None,
                )
        except Exception as log_error:
            logger.error(f"Error creating error log: {log_error}")

        # Clean up working directory on critical error
        if sim_dir and os.path.exists(sim_dir):
            try:
                # Create a minimal error log in the directory
                with open(f"{sim_dir}/error.log", "w") as f:
                    f.write(f"Error in simulation: {error_msg}")
            except:
                pass

        raise Exception(error_msg)
