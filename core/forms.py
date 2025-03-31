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


class SignupForm(UserCreationForm):
    email = forms.EmailField(
        required=True, widget=forms.EmailInput(attrs={"class": "input input-bordered"})
    )
    role = forms.ModelChoiceField(
        queryset=Role.objects.all(),
        widget=forms.Select(attrs={"class": "select select-bordered w-full"}),
    )
    first_name = forms.CharField(
        required=False, widget=forms.TextInput(attrs={"class": "input input-bordered"})
    )
    last_name = forms.CharField(
        required=False, widget=forms.TextInput(attrs={"class": "input input-bordered"})
    )

    class Meta:
        model = User
        fields = ("email", "first_name", "last_name", "role", "password1", "password2")

    def clean_email(self):
        email = self.cleaned_data.get("email")
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email address is already in use.")
        return email
