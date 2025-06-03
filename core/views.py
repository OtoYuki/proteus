from django.shortcuts import render, redirect, get_object_or_404
from django.http import FileResponse, Http404, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from .forms import SequenceForm, SignupForm, ProfileUpdateForm, CustomPasswordChangeForm
from .models import ProteinSequence, Prediction, ValidationMetric, JobQueue
from .tasks import run_colabfold, run_gromacs_simulation  # Importing the Tasks
import os
import logging
import zipfile
import tempfile
from django.contrib.auth import login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import datetime


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
    View to list predictions separated into 'Your Predictions' and 'Public Predictions'.
    Since this is an open platform, all predictions are public.
    """
    if request.user.is_authenticated:
        # Get user's own predictions
        user_predictions = Prediction.objects.filter(
            sequence__user=request.user
        ).order_by("-prediction_date")

        # Get all other users' predictions (public predictions)
        public_predictions = Prediction.objects.exclude(
            sequence__user=request.user
        ).order_by("-prediction_date")
    else:
        # For anonymous users, show no personal predictions and all as public
        user_predictions = Prediction.objects.none()
        public_predictions = Prediction.objects.all().order_by("-prediction_date")

    return render(
        request,
        "prediction_list.html",
        {
            "user_predictions": user_predictions,
            "public_predictions": public_predictions,
            "user_predictions_count": user_predictions.count(),
            "public_predictions_count": public_predictions.count(),
        },
    )


def prediction_detail(request, prediction_id):
    """
    View to display a single prediction with 3D visualization.
    """
    prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

    # Check if the PDB file exists and get additional info
    pdb_file_exists = False
    pdb_file_size = None
    pdb_file_date = None

    if prediction.pdb_file_path and os.path.exists(prediction.pdb_file_path):
        pdb_file_exists = True
        pdb_file_size = os.path.getsize(prediction.pdb_file_path) / 1024  # Size in KB
        pdb_file_date = datetime.datetime.fromtimestamp(
            os.path.getmtime(prediction.pdb_file_path)
        )

    return render(
        request,
        "prediction_detail.html",
        {
            "prediction": prediction,
            "pdb_file_exists": pdb_file_exists,
            "pdb_file_size": pdb_file_size,
            "pdb_file_date": pdb_file_date,
        },
    )


def serve_pdb(request, prediction_id):
    """
    View to serve PDB files directly.
    """
    prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

    # Check if file exists
    if not prediction.pdb_file_path or not os.path.exists(prediction.pdb_file_path):
        raise Http404("PDB file not found")

    # Serve the file with proper Content-Type
    response = FileResponse(open(prediction.pdb_file_path, "rb"))
    response["Content-Type"] = "chemical/x-pdb"  # Proper MIME type for PDB files
    response["Content-Disposition"] = f'inline; filename="{prediction_id}.pdb"'
    return response


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

        # 2.5. Also clean up any stuck JobQueue entries
        stuck_jobs = JobQueue.objects.filter(
            job_type="gromacs_simulation",
            status__in=["running", "pending"],
            started_at__lt=stalled_time,
        )
        for job in stuck_jobs:
            job.status = "failed"
            job.save()
            logger.info(f"Auto-marked stuck job {job.job_id} as failed")

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
    View to check the status of the GROMACS simulation with enhanced synchronization.
    """
    try:
        prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

        # Use the enhanced status management from ValidationMetric
        status_info = ValidationMetric.get_status_for_prediction(prediction_id)

        if status_info["status"] == "not_found":
            return JsonResponse(
                {
                    "status": "not_found",
                    "message": "No simulation found for this prediction",
                }
            )

        # Clean up any stale simulations
        latest_vm = (
            ValidationMetric.objects.filter(prediction=prediction)
            .order_by("-validation_date")
            .first()
        )

        if latest_vm:
            latest_vm.cleanup_stale_simulations()
            # Re-fetch status after cleanup
            status_info = ValidationMetric.get_status_for_prediction(prediction_id)

        return JsonResponse(status_info)

    except Exception as e:
        logger.error(f"Error in simulation_status view: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def simulation_status_realtime(request, prediction_id):
    """
    Enhanced real-time simulation status endpoint with automatic cleanup and synchronization.
    """
    try:
        prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

        # First, clean up any stale simulations
        ValidationMetric.objects.filter(
            prediction=prediction, status__in=["pending", "running"]
        ).update(
            modified_date=timezone.now()
        )  # Touch to update modified_date

        # Get enhanced status
        status_info = ValidationMetric.get_status_for_prediction(prediction_id)

        # Add additional metadata for frontend
        if status_info["status"] == "success":
            # Check if there are any recent logs
            from .models import Log

            recent_logs = Log.objects.filter(
                details__icontains=str(prediction_id), component="simulation_engine"
            ).order_by("-timestamp")[:3]

            status_info["recent_logs"] = [
                f"{log.timestamp.strftime('%H:%M:%S')}: {log.action} - {log.details[:150]}"
                for log in recent_logs
            ]

            # Add progress information if available
            if status_info["simulation_status"] == "running":
                latest_vm = (
                    ValidationMetric.objects.filter(prediction=prediction)
                    .order_by("-validation_date")
                    .first()
                )

                if latest_vm and latest_vm.validation_notes:
                    # Extract step information from validation notes
                    notes = latest_vm.validation_notes
                    if "step" in notes.lower():
                        status_info["progress_info"] = notes.split("\n")[
                            -1
                        ]  # Last line usually has progress

        return JsonResponse(status_info)

    except Exception as e:
        logger.error(f"Error in simulation_status_realtime view: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def download_trajectory(request, prediction_id):
    """View to serve both trajectory and structure files as a ZIP archive for PyMOL visualization."""
    try:
        prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

        # Get the latest completed simulation
        latest_sim = (
            ValidationMetric.objects.filter(
                prediction_id=prediction_id, status="completed"
            )
            .order_by("-validation_date")
            .first()
        )

        if not latest_sim:
            raise Http404("No completed simulation found")

        # Check if required files exist
        files_to_include = []

        # Add trajectory file (.trr or .xtc)
        if latest_sim.trajectory_path and os.path.exists(latest_sim.trajectory_path):
            trajectory_filename = os.path.basename(latest_sim.trajectory_path)
            # Change extension to .trr for PyMOL compatibility if it's .xtc
            if trajectory_filename.endswith(".xtc"):
                # Look for a .trr file in the same directory
                trr_path = latest_sim.trajectory_path.replace(".xtc", ".trr")
                if os.path.exists(trr_path):
                    files_to_include.append((trr_path, "trajectory.trr"))
                    logger.info(
                        f"Using .trr file for better PyMOL compatibility: {trr_path}"
                    )
                else:
                    files_to_include.append(
                        (latest_sim.trajectory_path, "trajectory.xtc")
                    )
                    logger.info(f"Using .xtc file: {latest_sim.trajectory_path}")
            else:
                files_to_include.append(
                    (latest_sim.trajectory_path, trajectory_filename)
                )
        else:
            logger.warning(f"Trajectory file not found: {latest_sim.trajectory_path}")

        # Add structure file (.gro)
        if latest_sim.structure_path and os.path.exists(latest_sim.structure_path):
            files_to_include.append((latest_sim.structure_path, "structure.gro"))
            logger.info(f"Including structure file: {latest_sim.structure_path}")
        else:
            logger.warning(f"Structure file not found: {latest_sim.structure_path}")

        if not files_to_include:
            raise Http404("No simulation files found")

        # Create a temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")

        try:
            with zipfile.ZipFile(temp_zip.name, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path, archive_name in files_to_include:
                    zipf.write(file_path, archive_name)
                    logger.info(f"Added {archive_name} to ZIP archive")

                # Add a README with instructions
                readme_content = """PyMOL Simulation Files
======================

This archive contains:
- structure.gro: Final MD structure with all atoms in correct order (required for PyMOL)
- trajectory.trr/.xtc: Full precision MD trajectory data

To visualize in PyMOL:
1. Load structure.gro first: File > Open > structure.gro
2. Then load trajectory: File > Open > trajectory.trr (or trajectory.xtc)
3. Use the trajectory controls to play the animation

For best visualization:
- Use trajectory.trr if available (higher precision)
- The structure.gro file ensures proper atom ordering
"""
                zipf.writestr("README.txt", readme_content)

            # Serve the ZIP file
            temp_zip.seek(0)
            response = FileResponse(
                open(temp_zip.name, "rb"),
                content_type="application/zip",
                as_attachment=True,
                filename=f"simulation_files_{prediction_id}.zip",
            )

            response["Content-Disposition"] = (
                f'attachment; filename="simulation_files_{prediction_id}.zip"'
            )

            # Clean up temp file after response (Django will handle this)
            def cleanup():
                try:
                    os.unlink(temp_zip.name)
                except OSError:
                    pass

            # Store cleanup function for later execution
            response._cleanup = cleanup

            logger.info(
                f"Serving ZIP archive with {len(files_to_include)} files for prediction {prediction_id}"
            )
            return response

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_zip.name)
            except OSError:
                pass
            raise e

    except Exception as e:
        logger.error(f"Error serving simulation files: {e}")
        raise Http404(f"Error: {str(e)}")


# Add this to views.py
def serve_trajectory_frame(request, prediction_id, frame_number):
    """Serve individual PDB frames from a trajectory"""
    try:
        prediction = get_object_or_404(Prediction, prediction_id=prediction_id)

        # Get the latest completed simulation
        latest_sim = (
            ValidationMetric.objects.filter(
                prediction_id=prediction_id, status="completed"
            )
            .order_by("-validation_date")
            .first()
        )

        if not latest_sim or not latest_sim.trajectory_path:
            raise Http404("Trajectory not found")

        # Convert XTC to PDB frames using MDAnalysis or similar
        # This is where you'd extract frame 'frame_number' from the XTC file
        # For now, just return the original PDB for any frame request

        return FileResponse(
            open(prediction.pdb_file_path, "rb"), content_type="chemical/x-pdb"
        )

    except Exception as e:
        logger.error(f"Error serving trajectory frame: {e}")
        raise Http404(f"Error: {str(e)}")


# Add this function to your views.py


def signup_view(request):
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Log the user in
            login(request, user)
            messages.success(
                request,
                f"Welcome to Proteus, {user.first_name or user.email.split('@')[0]}! Your account has been created successfully.",
            )
            # Redirect to a dedicated success page
            return redirect("signup_success")
    else:
        form = SignupForm()

    return render(request, "registration/signup.html", {"form": form})


# Simple view for the signup success page
@login_required
def signup_success_view(request):
    return render(request, "registration/signup_success.html")


@login_required
def profile_view(request):
    """
    View for user profile management.
    """
    if request.method == "POST":
        if "update_profile" in request.POST:
            # Handle profile update
            profile_form = ProfileUpdateForm(
                request.POST, instance=request.user, user=request.user
            )
            password_form = CustomPasswordChangeForm(request.user)

            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, "Your profile has been updated successfully!")
                return redirect("profile")
            else:
                messages.error(request, "Please correct the errors below.")

        elif "change_password" in request.POST:
            # Handle password change
            profile_form = ProfileUpdateForm(instance=request.user, user=request.user)
            password_form = CustomPasswordChangeForm(request.user, request.POST)

            if password_form.is_valid():
                password_form.save()
                messages.success(
                    request, "Your password has been changed successfully!"
                )
                return redirect("profile")
            else:
                messages.error(request, "Please correct the password errors below.")
    else:
        profile_form = ProfileUpdateForm(instance=request.user, user=request.user)
        password_form = CustomPasswordChangeForm(request.user)

    # Get user statistics
    user_predictions = Prediction.objects.filter(sequence__user=request.user)
    stats = {
        "total_predictions": user_predictions.count(),
        "completed_predictions": user_predictions.filter(status="completed").count(),
        "pending_predictions": user_predictions.filter(status="pending").count(),
        "running_predictions": user_predictions.filter(status="running").count(),
        "failed_predictions": user_predictions.filter(status="failed").count(),
    }

    # Get recent predictions
    recent_predictions = user_predictions.order_by("-prediction_date")[:5]

    context = {
        "profile_form": profile_form,
        "password_form": password_form,
        "stats": stats,
        "recent_predictions": recent_predictions,
    }

    return render(request, "profile.html", context)
