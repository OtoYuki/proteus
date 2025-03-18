from django.shortcuts import render, redirect, get_object_or_404
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from .forms import SequenceForm
from .models import ProteinSequence, Prediction, ValidationMetric
from .tasks import run_colabfold, run_gromacs_simulation  # Importing the Tasks
import os
import logging

logger = logging.getLogger(__name__)


# Create your views here.


def home(request):
    """
    View for the home page.
    """
    return render(request, "home.html")


def submit_sequence(request):
    """
    View to handle the submission of protein sequences.
    """
    if request.method == "POST":
        form = SequenceForm(request.POST)
        if form.is_valid():
            sequence = form.save(commit=False)
            sequence.user = (
                request.user
            )  # Associate the sequence with the logged-in user
            sequence.status = "pending"
            sequence.sequence_length = len(sequence.sequence_fasta)
            sequence.save()

            # Create a Prediction object
            prediction = Prediction(
                sequence=sequence, status="pending", model_version="ColabFold"
            )
            prediction.save()

            # Run the Celery task to process the prediction
            run_colabfold.delay(prediction.prediction_id)

            # Redirect to a Status Page
            return redirect(
                "home"
            )  # Temporary redirect to home until prediction_status is implemented
    else:
        form = SequenceForm()

    return render(request, "submit_sequence.html", {"form": form})


def prediction_list(request):
    """
    View to list all predictions (from the current user).
    This view will be used to display the predictions made by the user.
    """
    predictions = Prediction.objects.all().order_by("-prediction_date")
    return render(request, "prediction_list.html", {"predictions": predictions})


def prediction_detail(request, prediction_id):
    """
    View to display a single prediction with 3D visualization.
    """
    prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

    # Check if the PDB file exists
    pdb_file_exists = False
    if prediction.pdb_file_path and os.path.exists(prediction.pdb_file_path):
        pdb_file_exists = True

    return render(
        request,
        "prediction_detail.html",
        {
            "prediction": prediction,
            "pdb_file_exists": pdb_file_exists,
        },
    )


def serve_pdb(request, prediction_id):
    """
    View to serve PDB files directly.
    """
    prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

    if not prediction.pdb_file_path or not os.path.exists(prediction.pdb_file_path):
        raise Http404("PDB file not found")

    return FileResponse(
        open(prediction.pdb_file_path, "rb"), content_type="chemical/x-pdb"
    )


@require_POST
@csrf_exempt
def start_gromacs_simulation(request, prediction_id):
    """
    View to start a GROMACS simulation for a protein structure.
    Automatically cleans up stalled/failed simulations.
    """
    try:
        prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

        # Check if PDB file exists
        if not prediction.pdb_file_path or not os.path.exists(prediction.pdb_file_path):
            return JsonResponse(
                {"status": "error", "message": "PDB file not found"}, status=404
            )

        # AUTO-CLEANUP: Find simulations that are stuck or stalled
        # 1. Find "running" simulations that haven't been updated in 10+ minutes
        stalled_time = timezone.now() - timezone.timedelta(minutes=10)
        stalled_sims = ValidationMetric.objects.filter(
            prediction=prediction,
            status__in=["running", "pending"],
            modified_date__lt=stalled_time,
        )

        # 2. Mark these as failed with a note
        if stalled_sims.exists():
            for sim in stalled_sims:
                sim.status = "failed"
                sim.validation_notes = f"{sim.validation_notes or ''}\nAutomatically marked as failed due to lack of progress."
                sim.save()
                logger.info(f"Auto-marked stalled simulation {sim.metric_id} as failed")

        # 3. Check for any remaining active simulations after cleanup
        active_sims = ValidationMetric.objects.filter(
            prediction=prediction, status__in=["running", "pending"]
        )

        if active_sims.exists():
            # There are still active simulations that aren't stalled
            return JsonResponse(
                {
                    "status": "error",
                    "message": f"A simulation is already {active_sims[0].status}",
                    "simulation_id": str(active_sims[0].metric_id),
                },
                status=409,
            )

        # Start a new simulation task
        task = run_gromacs_simulation.delay(prediction_id)

        # Create a ValidationMetric entry
        validation_metric = ValidationMetric(
            prediction=prediction,
            status="pending",
            validation_notes=f"Simulation queued (task_id: {task.id})",
        )
        validation_metric.save()

        return JsonResponse(
            {
                "status": "success",
                "message": "GROMACS simulation started",
                "task_id": task.id,
            }
        )

    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


# A view to check the status of the GROMACS simulation
def simulation_status(request, prediction_id):
    """
    View to check the status of the GROMACS simulation.
    """
    try:
        prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

        # Get the latest simulation for this prediction
        latest_sim = (
            ValidationMetric.objects.filter(prediction=prediction)
            .order_by("-validation_date")
            .first()
        )

        if not latest_sim:
            return JsonResponse(
                {
                    "status": "not_found",
                    "message": "No simulation found for this prediction",
                }
            )

        return JsonResponse(
            {
                "status": "success",
                "simulation_status": latest_sim.status,
                "date": latest_sim.validation_date,
                "trajectory_path": latest_sim.trajectory_path,
                "notes": latest_sim.validation_notes,
            }
        )

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


# Add this function to your views.py file


def download_trajectory(request, prediction_id):
    """
    View to download a simulation trajectory file.
    """
    try:
        prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

        # Get the latest completed simulation
        latest_sim = (
            ValidationMetric.objects.filter(prediction=prediction, status="completed")
            .order_by("-validation_date")
            .first()
        )

        if (
            not latest_sim
            or not latest_sim.trajectory_path
            or not os.path.exists(latest_sim.trajectory_path)
        ):
            raise Http404("Trajectory file not found")

        # Get filename from path
        filename = os.path.basename(latest_sim.trajectory_path)

        return FileResponse(
            open(latest_sim.trajectory_path, "rb"),
            content_type="application/octet-stream",
            as_attachment=True,
            filename=filename,
        )

    except Exception as e:
        logger.error(f"Error serving trajectory file: {e}")
        raise Http404(f"Error: {str(e)}")
