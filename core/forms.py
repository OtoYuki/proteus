from django import forms
from .models import ProteinSequence, Role
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import get_user_model


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


User = get_user_model()


class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    role = forms.ModelChoiceField(
        queryset=Role.objects.all(), required=True, empty_label="Select a role"
    )

    class Meta:
        model = User
        fields = ("email", "password1", "password2", "role")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user
