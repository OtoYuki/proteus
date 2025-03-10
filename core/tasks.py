from celery import shared_task
import subprocess
from .models import Prediction
import os
import logging
import glob
import time

logger = logging.getLogger(__name__)


@shared_task
def run_colabfold(prediction_id):
    logger.info(f"Starting ColabFold prediction for prediction_id: {prediction_id}")

    try:
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
        prediction.save()
        logger.info("Prediction status updated to running")

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
        logger.info("ColabFold prediction completed")

        # Update prediction with actual PDB file
        pdb_files = glob.glob(f"{host_output_dir}/rank_*.pdb")
        if pdb_files:
            prediction.pdb_file_path = pdb_files[0]
            logger.info(f"Set PDB file path to: {prediction.pdb_file_path}")
        else:
            raise Exception(f"No PDB files found in {host_output_dir}")
        prediction.status = "completed"
        prediction.save()
        logger.info("Prediction status updated to completed")

        return f"Prediction {prediction_id} completed successfully"

    except Prediction.DoesNotExist:
        logger.error(f"Prediction with id {prediction_id} does not exist")
        return f"Error: Prediction with id {prediction_id} does not exist"
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed: {e}")
        try:
            prediction = Prediction.objects.get(prediction_id=prediction_id)
            prediction.status = "failed"
            prediction.save()
            logger.info("Prediction status updated to failed due to subprocess error")
        except Exception as inner_e:
            logger.error(f"Failed to update prediction status: {inner_e}")
        return f"Error: Command execution failed for prediction {prediction_id}"
    except Exception as e:
        logger.error(f"Error in run_colabfold: {e}")
        try:
            prediction = Prediction.objects.get(prediction_id=prediction_id)
            prediction.status = "failed"
            prediction.save()
            logger.info("Prediction status updated to failed")
        except Exception as inner_e:
            logger.error(f"Failed to update prediction status: {inner_e}")
        return f"Error: {str(e)}"
