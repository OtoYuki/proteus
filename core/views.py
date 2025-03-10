from django.shortcuts import render, redirect
from .forms import SequenceForm
from .models import ProteinSequence, Prediction
from .tasks import run_colabfold  # Importing the Celery task


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
