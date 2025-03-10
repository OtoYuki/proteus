from django.shortcuts import render, redirect, get_object_or_404
from django.http import FileResponse, Http404
from .forms import SequenceForm
from .models import ProteinSequence, Prediction
from .tasks import run_colabfold  # Importing the Celery task
import os


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
