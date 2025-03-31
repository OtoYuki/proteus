from datetime import timezone
from celery import shared_task
import subprocess
from .models import JobQueue, Log, MLRanking, Prediction, SystemMetric, ValidationMetric
import os
import logging
import time
import glob
import socket

logger = logging.getLogger(__name__)


# ColabFold prediction task
@shared_task
def run_colabfold(prediction_id):
    """
    Run a ColabFold prediction for a given prediction ID.
    """
    logger.info(f"Starting ColabFold prediction for prediction_id: {prediction_id}")

    # Check if Celery worker is running, start one if not
    try:
        # Check for running Celery workers
        ps_cmd = "ps aux | grep 'celery worker' | grep -v grep | wc -l"
        ps_result = subprocess.run(ps_cmd, shell=True, capture_output=True, text=True)
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
                    started_at=timezone.now(),
                    job_parameters={"prediction_id": str(prediction_id)},
                )
                logger.info(f"Created new job for prediction: {prediction_id}")
            else:
                # Update existing job
                job.status = "running"
                job.started_at = timezone.now()
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
            import psutil

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
        except ImportError:
            logger.warning("psutil not installed, skipping system metrics")
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
                subprocess.run(f"docker start {container_id}", shell=True, check=True)
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
        logger.info(f"ColabFold prediction completed in {execution_time:.2f} seconds")

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
                import json

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
                import json

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
                found_files = [f for f in find_result.stdout.strip().split("\n") if f]
                if found_files:
                    pdb_file_path = found_files[0]
                    logger.info(f"Found PDB file using find command: {pdb_file_path}")

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
            import datetime

            metadata.update(
                {
                    "sequence_name": sequence.sequence_name,
                    "sequence_length": len(sequence.sequence_fasta.replace("\n", "")),
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
            logger.info(f"Updated sequence {sequence.sequence_id} status to completed")

            logger.info(
                f"Prediction status updated to completed with all fields populated. PDB path: {pdb_file_path}"
            )

            # Update job status
            if job:
                job.status = "completed"
                job.completed_at = timezone.now()
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
                approx_rg = 1.2 * pow(residue_count, 1 / 3)  # Very rough approximation

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
                import psutil

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

            return (
                f"Prediction {prediction_id} completed successfully with enhanced data"
            )
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
                job.completed_at = timezone.now()
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
            job.completed_at = timezone.now()
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

            logger.info("Prediction status updated to failed due to subprocess error")

            # Update job status
            if job:
                job.status = "failed"
                job.completed_at = timezone.now()
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
                job.completed_at = timezone.now()
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


# GROMACS Task
@shared_task
def run_gromacs_simulation(prediction_id):
    """
    Run a GROMACS simulation for a given prediction derived from ColabFold.
    """
    # Check if a simulation is already running for this prediction

    existing_sim = ValidationMetric.objects.filter(
        prediction_id=prediction_id, status="running"
    ).first()

    if existing_sim:
        # If it's been running for more than 30 minutes, assume it's stalled
        time_threshold = timezone.now() - timezone.timedelta(minutes=30)
        if existing_sim.created_at < time_threshold:
            existing_sim.status = "failed"
            existing_sim.validation_notes = "Simulation timed out after 30 minutes"
            existing_sim.save()
            logger.warning(
                f"Found stale simulation for {prediction_id}, marking as failed"
            )
        else:
            logger.warning(f"A simulation is already running for {prediction_id}")
            return f"A simulation is already running for prediction {prediction_id}"
    logger.info(f"Starting GROMACS simulation for prediction_id: {prediction_id}")

    try:
        # Getting the Prediction Object
        prediction = Prediction.objects.get(prediction_id=prediction_id)
        logger.info(f"Retrieved prediction object: {prediction}")

        if not prediction.pdb_file_path or not os.path.exists(prediction.pdb_file_path):
            raise Exception(
                "PDB file not found. Ensure the ColabFold prediction is complete."
            )
        logger.info(f"PDB file path: {prediction.pdb_file_path}")

        # Create a ValidationMetric entry to track the simulation
        validation_metric = ValidationMetric(
            prediction=prediction,
            status="running",
            simulation_parameters={
                "forcefield": "amber99sb-ildn",
                "water_model": "spce",
                "simulation_time": "100ps",
                "temperature": 300,
            },
            validation_notes="GROMACS simulation started",
        )
        validation_metric.save()

        # Setting up the Simulation Directory
        host_data_dir = "/home/sire/colabfold_data"
        sim_dir = f"{host_data_dir}/simulations/{prediction_id}"
        os.makedirs(sim_dir, exist_ok=True)

        # Copy the PDF file to the simulation directory
        pdb_basename = os.path.basename(prediction.pdb_file_path)
        sim_pdb_path = os.path.join(sim_dir, pdb_basename)
        subprocess.run(
            f"cp {prediction.pdb_file_path} {sim_pdb_path}", shell=True, check=True
        )

        # Checking the status of the GROMACS container and starting it if necessary
        check_exists_cmd = "docker ps -aq -f name=gromacs"
        exists_result = subprocess.run(
            check_exists_cmd, shell=True, capture_output=True, text=True
        )
        container_id = exists_result.stdout.strip()

        if container_id:
            check_running_cmd = "docker ps -q -f name=gromacs"
            running_result = subprocess.run(
                check_running_cmd, shell=True, capture_output=True, text=True
            )
            if not running_result.stdout.strip():
                logger.info(
                    "GROMACS container exists but is not running. Starting it..."
                )
                subprocess.run(f"docker start {container_id}", shell=True, check=True)

        else:
            logger.info("GROMACS container does not exist. Creating it...")
            start_cmd = (
                "docker run -d --name gromacs --gpus all "
                + "-v /home/sire/colabfold_data:/data gromacs/gromacs:latest "
                + "tail -f /dev/null"
            )
            subprocess.run(start_cmd, shell=True, check=True)
            logger.info("Started GROMACS container")
            time.sleep(5)  # Wait for the container to stabilize first

        # Container Paths
        container_sim_dir = sim_dir.replace("/home/sire/colabfold_data", "/data")
        container_pdb_path = os.path.join(container_sim_dir, pdb_basename)
        logger.info(f"Container PDB path: {container_pdb_path}")
        protein_name = os.path.splitext(pdb_basename)[0]
        logger.info(f"Protein name: {protein_name}")

        # Creating the GROMACS simulation environment with parameter files
        mdp_files = {
            "ions.mdp": """
; ions.mdp - used as input into grompp to generate ions.tpr
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = 0.01          ; Minimization step size
nsteps      = 50000         ; Maximum number of (minimization) steps to perform

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist     = 10            ; Frequency to update the neighbor list and long range forces
cutoff-scheme = Verlet
ns_type     = grid          ; Method to determine neighbor list (simple, grid)
coulombtype = PME           ; Treatment of long range electrostatic interactions
rcoulomb    = 1.0           ; Short-range electrostatic cut-off
rvdw        = 1.0           ; Short-range Van der Waals cut-off
pbc         = xyz           ; Periodic Boundary Conditions in all 3 dimensions
    """,
            "minim.mdp": """
; minim.mdp - used as input into grompp to generate em.tpr
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = 0.01          ; Minimization step size
nsteps      = 50000         ; Maximum number of (minimization) steps to perform

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist         = 10        ; Frequency to update the neighbor list and long range forces
cutoff-scheme   = Verlet    ; Buffered neighbor searching
nstype          = grid      ; Method to determine neighbor list (simple, grid)
coulombtype     = PME       ; Treatment of long range electrostatic interactions
rcoulomb        = 1.0       ; Short-range electrostatic cut-off
rvdw            = 1.0       ; Short-range Van der Waals cut-off
pbc             = xyz       ; Periodic Boundary Conditions
    """,
            "md.mdp": """
; md.mdp - used as input into grompp to generate md.tpr
title       = Protein MD simulation 
; Run parameters
integrator  = md        ; leap-frog integrator
nsteps      = 5000      ; 2 * 5000 = 10 ps
dt          = 0.002     ; 2 fs
; Output control
nstxout     = 500       ; save coordinates every 1.0 ps
nstvout     = 500       ; save velocities every 1.0 ps
nstenergy   = 500       ; save energies every 1.0 ps
nstlog      = 500       ; update log file every 1.0 ps
nstxtcout   = 500       ; save compressed trajectory every 1.0 ps
xtc-precision = 1000    ; precision with which to write xtc file
; Bond parameters
continuation    = no    ; first dynamics run
constraint_algorithm = lincs     ; holonomic constraints 
constraints     = h-bonds       ; bonds involving H are constrained 
; Neighbor searching
cutoff-scheme   = Verlet
ns_type         = grid          ; search neighboring grid cells
nstlist         = 10            ; 20 fs
rcoulomb        = 1.0           ; short-range electrostatic cutoff (in nm)
rvdw            = 1.0           ; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype     = PME           ; Particle Mesh Ewald for long-range electrostatics
; Temperature coupling
tcoupl      = V-rescale         ; modified Berendsen thermostat
tc-grps     = Protein Non-Protein   ; two coupling groups - more accurate
tau_t       = 0.1   0.1         ; time constant, in ps
ref_t       = 300   300         ; reference temperature, one for each group, in K
; Pressure coupling
pcoupl      = no                ; no pressure coupling in NVT
; Periodic boundary conditions
pbc         = xyz               ; 3-D PBC
; Dispersion correction
DispCorr    = EnerPres          ; account for cut-off vdW scheme
; Velocity generation
gen_vel     = yes               ; assign velocities from Maxwell distribution
gen_temp    = 300               ; temperature for Maxwell distribution
gen_seed    = -1                ; generate a random seed
""",
        }

        for mdp_name, content in mdp_files.items():
            with open(os.path.join(sim_dir, mdp_name), "w") as f:
                f.write(content)

        setup_cmd = f"""docker exec gromacs bash -c '
        # Make sure the simulation directory exists and has proper permissions
        mkdir -p {container_sim_dir}
        chmod -R 777 {container_sim_dir}
        echo "Directory prepared with proper permissions"
        '"""

        subprocess.run(setup_cmd, shell=True, check=True)
        # Running the GROMACS commands

        steps = [
            # 1. Generate topology - Now with proper directory context
            f"""docker exec gromacs bash -c '
            set -e  # Exit on any error
            cd {container_sim_dir}
            echo "Step 1: Generating topology"
            gmx pdb2gmx -f {container_pdb_path} -o {protein_name}_processed.gro -p topol.top -water spce -ff amber99sb-ildn -ignh
            chmod -R 777 .  # Ensure all files are writable
            '""",
            # 2. Define simulation box
            f"""docker exec gromacs bash -c '
            set -e
            cd {container_sim_dir}
            echo "Step 2: Defining simulation box"
            gmx editconf -f {protein_name}_processed.gro -o {protein_name}_box.gro -c -d 1.0 -bt cubic
            chmod -R 777 .
            '""",
            # 3. Solvate the box - with fallback options for water model
            f"""docker exec gromacs bash -c '
            set -e
            cd {container_sim_dir}
            echo "Step 3: Solvating the box"

            # Find all available water models
            echo "Available water models:"
            find /gromacs -name "*.gro" 2>/dev/null | grep -i "spc\\|tip" || echo "No water models found with find"

            # Try with built-in spc216
            if gmx solvate -cp {protein_name}_box.gro -cs spc216 -o {protein_name}_solv.gro -p topol.top; then
                echo "Solvation with spc216 successful"
            elif gmx solvate -cp {protein_name}_box.gro -cs tip3p -o {protein_name}_solv.gro -p topol.top; then
                echo "Fallback to tip3p successful"
            else
                # Create a minimal water box if all else fails
                echo "Creating minimal water box"
                gmx solvate -cp {protein_name}_box.gro -o {protein_name}_solv.gro -p topol.top
            fi

            chmod -R 777 .
            '""",
            # 4. Add ions (prepare) - FIXED: removed redundant cp command
            f"""docker exec gromacs bash -c '
            set -e
            cd {container_sim_dir}
            echo "Step 4: Preparing for ion addition"
            # File is already there, no need to copy
            gmx grompp -f ions.mdp -c {protein_name}_solv.gro -p topol.top -o {protein_name}_ions.tpr -maxwarn 2
            chmod -R 777 .
            '""",
            # 5. Add ions (execute) - with proper file handling
            f"""docker exec gromacs bash -c '
            set -e
            cd {container_sim_dir}
            echo "Step 5: Adding ions"
            # Make a backup of topol.top first
            cp topol.top topol.top.bak
            echo SOL | gmx genion -s {protein_name}_ions.tpr -o {protein_name}_solv_ions.gro -p topol.top -pname NA -nname CL -neutral

            # Verify the file was modified correctly
            if [ ! -s topol.top ]; then
                echo "Error: topol.top is empty after genion, restoring from backup"
                cp topol.top.bak topol.top
            fi

            chmod -R 777 .
            '""",
            # 6. Energy minimization (prepare) - FIXED: removed redundant cp command
            f"""docker exec gromacs bash -c '
            set -e
            cd {container_sim_dir}
            echo "Step 6: Preparing for energy minimization"
            # File is already there, no need to copy
            gmx grompp -f minim.mdp -c {protein_name}_solv_ions.gro -p topol.top -o {protein_name}_em.tpr -maxwarn 2
            chmod -R 777 .
            '""",
            # 7. Energy minimization (run)
            f"""docker exec gromacs bash -c '
            set -e
            cd {container_sim_dir}
            echo "Step 7: Running energy minimization"
            gmx mdrun -v -deffnm {protein_name}_em
            chmod -R 777 .
            '""",
            # 8. Brief MD simulation (prepare) - FIXED: removed redundant cp command
            f"""docker exec gromacs bash -c '
            set -e
            cd {container_sim_dir}
            echo "Step 8: Preparing for MD simulation"
            # File is already there, no need to copy
            gmx grompp -f md.mdp -c {protein_name}_em.gro -p topol.top -o {protein_name}_md.tpr -maxwarn 2
            chmod -R 777 .
            '""",
            # 9. Brief MD simulation (run)
            f"""docker exec gromacs bash -c "cd {container_sim_dir} && \
            echo Step 9: Running MD simulation && \
            gmx mdrun -v -deffnm {protein_name}_md && \
            echo MD simulation completed, checking for output files && \
            ls -la {protein_name}_md* && \
            chmod -R 777 ."
            """,
        ]

        # Executing Each Step
        for i, step_cmd in enumerate(steps):
            logger.info(f"Running GROMACS step {i+1}/{len(steps)}")
            try:
                result = subprocess.run(
                    step_cmd, shell=True, capture_output=True, text=True, check=True
                )
                logger.info(f"Step {i+1} completed successfully")
                logger.debug(f"Output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"GROMACS step {i+1} failed with error code {e.returncode}"
                )
                logger.error(f"Command output: {e.stdout}")
                logger.error(f"Command error: {e.stderr}")
                validation_metric.status = "failed"
                validation_metric.validation_notes = (
                    f"Step {i+1} failed: {e} - {e.stderr}"
                )
                validation_metric.save()
                return f"Error in GROMACS simulation: Step {i+1} failed - {e.stderr}"

        # Add this right before the final ValidationMetric status update
        try:
            # Update the ValidationMetric with results.
            trajectory_path = f"{sim_dir}/{protein_name}_md.xtc"
            if os.path.exists(trajectory_path):
                validation_metric.trajectory_path = trajectory_path
                validation_metric.status = "completed"
                validation_metric.validation_notes = (
                    "GROMACS simulation completed successfully"
                )
            else:
                logger.warning(f"Trajectory file not found at {trajectory_path}")
                validation_metric.status = "failed"
                validation_metric.validation_notes = (
                    "Simulation completed but no trajectory file was generated"
                )
            validation_metric.save()

            logger.info(
                f"GROMACS simulation completed for prediction_id: {prediction_id}"
            )
            return f"GROMACS simulation completed for prediction_id: {prediction_id}"

        except Exception as e:
            logger.error(f"Error finalizing GROMACS simulation: {e}")
            validation_metric.status = "failed"
            validation_metric.validation_notes = f"Error finalizing results: {str(e)}"
            validation_metric.save()
            return f"Error finalizing results: {str(e)}"

    except Exception as e:
        logger.error(f"Error in GROMACS simulation: {e}")
        try:
            validation_metric = ValidationMetric.objects.get(
                prediction_id=prediction_id, status="running"
            )
            validation_metric.status = "failed"
            validation_metric.validation_notes = f"Error: {str(e)}"
            validation_metric.save()
        except:
            pass
        return f"Error: {str(e)}"
