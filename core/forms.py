from django import forms
from .models import ProteinSequence


class ProteinSequenceForm(forms.ModelForm):
    """Form for uploading protein sequences."""

    class Meta:
        model = ProteinSequence
        fields = [
            "sequence_name",
            "sequence_fasta",
            "description",
            "organism",
            "source",
        ]
        widgets = {
            "sequence_name": forms.TextInput(attrs={"class": "form-control"}),
            "sequence_fasta": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 6,
                    "placeholder": ">Sequence name\nMETVYIASLLLLGAGLAVAKVSLNSPPTNVHISNPERIYDKNNIKFAGEW...",
                }
            ),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
            "organism": forms.TextInput(attrs={"class": "form-control"}),
            "source": forms.TextInput(attrs={"class": "form-control"}),
        }

    def clean_sequence_fasta(self):
        """Validate and clean the FASTA sequence."""
        fasta = self.cleaned_data.get("sequence_fasta")

        if not fasta:
            raise forms.ValidationError("Protein sequence is required")

        # Basic FASTA format validation
        lines = fasta.strip().split("\n")
        if not lines or not lines[0].startswith(">"):
            raise forms.ValidationError(
                "Invalid FASTA format. Sequence should start with '>'"
            )

        # Extract just the sequence data (excluding header)
        sequence = "".join(lines[1:]).replace(" ", "").upper()

        # Calculate sequence length
        self.sequence_length = len(sequence)

        return fasta
