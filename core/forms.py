from django import forms
from .models import ProteinSequence


class SequenceForm(forms.ModelForm):
    """
    The form for uploading protein sequences.
    """

    class Meta:
        model = ProteinSequence
        fields = [
            "sequence_name",
            "sequence_fasta",
            "description",
            "organism",
            "source",
        ]
