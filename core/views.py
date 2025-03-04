from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView, CreateView
from django.urls import reverse_lazy
from django.contrib import messages
from django.utils import timezone

from .models import ProteinSequence, Prediction, JobQueue
from .forms import ProteinSequenceForm


def home(request):
    """Home page view."""
    return render(request, "core/home.html")


class ProteinSequenceCreateView(LoginRequiredMixin, CreateView):
    model = ProteinSequence
    form_class = ProteinSequenceForm
    template_name = "core/sequence_create.html"
    success_url = reverse_lazy("sequence_list")

    def form_valid(self, form):
        # Set user and additional fields
        form.instance.user = self.request.user
        form.instance.status = "uploaded"
        form.instance.sequence_length = form.sequence_length

        # Save the sequence
        response = super().form_valid(form)

        # Create a job for prediction
        JobQueue.objects.create(
            user=self.request.user,
            job_type="protein_prediction",
            status="pending",
            job_parameters={
                "sequence_id": str(self.object.sequence_id),
                "sequence_name": self.object.sequence_name,
            },
            priority=0,
        )

        messages.success(
            self.request,
            f"Sequence '{self.object.sequence_name}' uploaded successfully and queued for prediction.",
        )
        return response


class ProteinSequenceListView(LoginRequiredMixin, ListView):
    model = ProteinSequence
    template_name = "core/sequence_list.html"
    context_object_name = "sequences"

    def get_queryset(self):
        return ProteinSequence.objects.filter(user=self.request.user).order_by(
            "-upload_date"
        )


class ProteinSequenceDetailView(LoginRequiredMixin, DetailView):
    model = ProteinSequence
    template_name = "core/sequence_detail.html"
    context_object_name = "sequence"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["predictions"] = self.object.predictions.all().order_by(
            "-prediction_date"
        )
        return context
