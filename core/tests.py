from django.test import (
    TestCase,
    Client,
    LiveServerTestCase,
)
from django.urls import reverse
from unittest.mock import patch, MagicMock
import time


from django.contrib.auth import get_user_model
from .models import ProteinSequence, Prediction, ValidationMetric, MLRanking
from django.core.exceptions import ValidationError

User = get_user_model()


class SequenceModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="testpass123"
        )

    def test_sequence_model_validation(self):  # UT1
        """Test ProteinSequence model validates FASTA sequences correctly"""
        valid_seq = ProteinSequence(
            user=self.user,
            sequence_name="Valid",
            sequence_fasta=">seq1\nMKT",
            status="pending",
            sequence_length=3,
        )
        try:
            valid_seq.full_clean()
            valid_seq.save()
        except ValidationError:
            self.fail("Valid FASTA sequence should not raise ValidationError")

        invalid_seq = ProteinSequence(
            user=self.user,
            sequence_name="Invalid",
            sequence_fasta=">seq1\nXYZ",  # Invalid amino acids
            status="pending",
            sequence_length=3,
        )
        with self.assertRaises(ValidationError):
            invalid_seq.full_clean()

    def test_user_model_relationship(self):  # UT2
        """Test cascade deletion behaves correctly"""
        seq = ProteinSequence.objects.create(
            user=self.user,
            sequence_name="Test",
            sequence_fasta=">seq1\nMKT",
            status="pending",
            sequence_length=3,
        )
        seq_id = seq.sequence_id
        self.user.delete()
        self.assertFalse(ProteinSequence.objects.filter(sequence_id=seq_id).exists())


class SubmissionFormTests(TestCase):
    def test_submission_form_valid_input(self):  # UT3
        """Test form accepts valid input"""
        form_data = {
            "sequence_name": "Test Protein",
            "sequence_fasta": ">seq1\nMKTESTSEQUENCE",
            "description": "Test description",
            "organism": "E. coli",
            "source": "UniProt",
        }
        from .forms import SequenceForm

        form = SequenceForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_submission_form_invalid_input(self):  # UT4
        """Test form rejects invalid input"""
        form_data = {
            "sequence_name": "",  # Required field
            "sequence_fasta": "Invalid FASTA",  # Invalid format
            "description": "Test",
        }
        from .forms import SequenceForm

        form = SequenceForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("sequence_name", form.errors)


class ViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="testpass123"
        )
        self.client.login(email="test@example.com", password="testpass123")

    def test_home_view_get_request(self):  # UT5
        """Test home view returns correct response"""
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "home.html")

    def test_login_view_authentication(self):  # UT6
        """Test login view handles authentication correctly"""
        self.client.logout()
        # Test valid credentials
        response = self.client.post(
            reverse("login"),
            {"username": "test@example.com", "password": "testpass123"},
        )
        self.assertIn(response.status_code, [200, 302])  # Success or redirect

        # Test invalid credentials
        response = self.client.post(
            reverse("login"), {"username": "test@example.com", "password": "wrongpass"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "registration/login.html")

    def test_result_view_template(self):  # UT7
        """Test result view displays prediction data correctly"""
        seq = ProteinSequence.objects.create(
            user=self.user,
            sequence_name="Test",
            sequence_fasta=">seq1\nMKT",
            status="completed",
            sequence_length=3,
        )
        pred = Prediction.objects.create(
            sequence=seq,
            status="completed",
            model_version="ColabFold",
            pdb_file_path="/test/path.pdb",
        )
        response = self.client.get(
            reverse("prediction_detail", args=[pred.prediction_id])
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "prediction_detail.html")
        self.assertContains(response, "Test")  # Should contain sequence name


class TaskTests(TestCase):
    @patch("core.tasks.subprocess.run")
    def test_prediction_task_success(self, mock_subprocess):  # UT8
        """Test successful prediction task execution"""
        mock_subprocess.return_value = MagicMock(returncode=0)

        user = User.objects.create_user(
            email="test@example.com", password="testpass123"
        )
        seq = ProteinSequence.objects.create(
            user=user,
            sequence_name="Test",
            sequence_fasta=">seq1\nMKT",
            status="pending",
            sequence_length=3,
        )
        pred = Prediction.objects.create(
            sequence=seq, status="pending", model_version="ColabFold"
        )

        from .tasks import run_colabfold

        result = run_colabfold(pred.prediction_id)
        self.assertNotIn("error", str(result).lower())

    @patch("core.tasks.subprocess.run", side_effect=Exception("Docker error"))
    def test_prediction_task_failure(self, mock_subprocess):  # UT9
        """Test prediction task failure handling"""
        user = User.objects.create_user(
            email="test@example.com", password="testpass123"
        )
        seq = ProteinSequence.objects.create(
            user=user,
            sequence_name="Test",
            sequence_fasta=">seq1\nMKT",
            status="pending",
            sequence_length=3,
        )
        pred = Prediction.objects.create(
            sequence=seq, status="pending", model_version="ColabFold"
        )

        from .tasks import run_colabfold

        result = run_colabfold(pred.prediction_id)
        self.assertIn("error", str(result).lower())

        # Verify the prediction status was updated
        pred.refresh_from_db()
        self.assertEqual(pred.status, "failed")


class UtilityTests(TestCase):
    def test_parse_fasta_utility(self):  # UT10
        """Test FASTA parsing utility"""
        valid_fasta = ">seq1\nMKT"
        invalid_fasta = "Invalid format"

        from core.utils import parse_fasta

        # Test valid FASTA
        result = parse_fasta(valid_fasta)
        self.assertEqual(result["id"], "seq1")
        self.assertEqual(result["sequence"], "MKT")

        # Test invalid FASTA
        with self.assertRaises(Exception):
            parse_fasta(invalid_fasta)
