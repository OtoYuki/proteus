from django import forms
from .models import ProteinSequence, Role, User  # Ensure User is imported
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


class SignupForm(UserCreationForm):
    email = forms.EmailField(
        required=True, widget=forms.EmailInput(attrs={"class": "input input-bordered"})
    )
    first_name = forms.CharField(
        required=False, widget=forms.TextInput(attrs={"class": "input input-bordered"})
    )
    last_name = forms.CharField(
        required=False, widget=forms.TextInput(attrs={"class": "input input-bordered"})
    )

    class Meta:
        model = User
        # Password fields are handled by UserCreationForm, we only need email and names here
        fields = ("email", "first_name", "last_name")

    def clean_email(self):
        email = self.cleaned_data.get("email")
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email address is already in use.")
        return email

    def save(self, commit=True):
        # Override the default save to use our custom manager's create_user method
        # This ensures the default role assignment logic in the manager is executed.
        user = User.objects.create_user(
            email=self.cleaned_data["email"],
            password=self.cleaned_data[
                "password2"
            ],  # Use password2 (confirmed password) from UserCreationForm
            first_name=self.cleaned_data.get("first_name"),
            last_name=self.cleaned_data.get("last_name"),
            # The UserManager.create_user method will handle assigning the default 'User' role
        )
        # The create_user method already saves the user, so we don't need to call user.save() again
        # or worry about the commit flag here.
        return user
