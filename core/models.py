# Description: This file contains the models for the core app.
from django.db import models
import uuid
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver


class Role(models.Model):
    """Role model for the user."""

    role_id = models.AutoField(primary_key=True)
    role_name = models.CharField(max_length=20, unique=True, db_index=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "role"
        verbose_name = "Role"
        verbose_name_plural = "Roles"
        indexes = [models.Index(fields=["role_name"])]

    def __str__(self):
        return self.role_name


class UserManager(BaseUserManager):
    """Custom manager for User model with email as the identifier."""

    def _create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email field is a mandatory requirement.")
        email = self.normalize_email(email)

        # Role should absolutely be in extra_fields when called from create_user/create_superuser
        if "role" not in extra_fields or extra_fields["role"] is None:
            # This indicates a programming error if hit from create_user/superuser
            print("ERROR: _create_user called without a valid role in extra_fields.")
            # Attempt to recover with default, but this shouldn't be necessary
            try:
                default_role, _ = Role.objects.get_or_create(role_name="User")
                extra_fields["role"] = default_role
                print("Warning: _create_user recovered by assigning the default role.")
            except Exception as e:
                raise ValueError(
                    f"Could not assign default role in _create_user recovery: {e}"
                )

        print(f"_create_user: Role in extra_fields: {extra_fields.get('role')}")
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        print(f"_create_user: User object created with role: {user.role}")
        user.save(using=self._db)
        print(f"_create_user: User saved with ID: {user.pk}")
        return user

    def create_user(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)

        # Ensure the default 'User' role is assigned *before* calling _create_user
        if "role" not in extra_fields:
            print("create_user: Assigning default 'User' role.")
            try:
                user_role, created = Role.objects.get_or_create(role_name="User")
                if created:
                    print(f"create_user: Default role 'User' created.")
                extra_fields["role"] = user_role
                print(f"create_user: Assigned role: {user_role}")
            except Exception as e:
                # Make failure explicit and stop user creation
                print(f"ERROR: Failed to get or create the default 'User' role: {e}")
                raise ValueError(
                    f"System configuration error: Default 'User' role is missing or inaccessible. {e}"
                )
        else:
            print(f"create_user: Role provided in extra_fields: {extra_fields['role']}")

        # Ensure role is not None before proceeding
        if extra_fields.get("role") is None:
            print("ERROR: Role became None unexpectedly in create_user.")
            raise ValueError("System error: Role assignment failed unexpectedly.")

        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        # Attempt to set the Admin role explicitly
        print("create_superuser: Assigning 'Admin' role.")
        try:
            admin_role, created = Role.objects.get_or_create(role_name="Admin")
            if created:
                print(f"create_superuser: Admin role created.")
            extra_fields["role"] = admin_role
            print(f"create_superuser: Assigned role: {admin_role}")
        except Exception as e:
            print(f"ERROR: Could not assign Admin role during superuser creation: {e}")
            # Raise error to prevent superuser creation without Admin role
            raise ValueError(
                f"System configuration error: 'Admin' role is missing or inaccessible. {e}"
            )

        return self._create_user(email, password, **extra_fields)


class User(AbstractUser):
    """Custom User model with email as the primary identifier."""

    user_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Allow null/blank temporarily for the save logic to handle default assignment
    role = models.ForeignKey(
        Role,
        on_delete=models.SET_NULL,  # Keep SET_NULL for flexibility, but manager should prevent null for new users
        related_name="users",
        null=True,
        blank=True,
    )
    email = models.EmailField(
        unique=True,
        db_index=True,
        db_collation="und-x-icu",  # Case-insensitive collation
    )
    username = None
    created_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    last_login = models.DateTimeField(null=True, blank=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = UserManager()

    def save(self, *args, **kwargs):
        print(f"User.save: Called for user {self.email}. Current role: {self.role}")
        # Role assignment is now handled entirely by the UserManager before save is called.
        # We only need to sync is_staff and handle superuser role enforcement.

        # 1. Sync is_staff based on role (if role is set)
        if self.role:
            print(f"User.save: Syncing is_staff based on role: {self.role.role_name}")
            if self.role.role_name == "Admin":
                self.is_staff = True
            else:
                # Ensure non-admins are not staff, unless they are superusers
                if not self.is_superuser:
                    self.is_staff = False
            print(f"User.save: is_staff set to {self.is_staff}")
        else:
            # This case should ideally not happen for new users if UserManager works.
            # Could happen if role is manually set to None later.
            print("User.save: Role is None. Setting is_staff based on is_superuser.")
            if not self.is_superuser:
                self.is_staff = False
            print(f"User.save: is_staff set to {self.is_staff}")

        # 2. Ensure Superusers always have 'Admin' role and are staff
        if self.is_superuser:
            print("User.save: Ensuring superuser settings.")
            self.is_staff = True  # Superusers MUST be staff
            try:
                admin_role, _ = Role.objects.get_or_create(role_name="Admin")
                if self.role != admin_role:
                    print(
                        f"User.save: Assigning/Correcting Admin role for superuser {self.email}."
                    )
                    self.role = admin_role
            except Exception as e:
                # Log this critical error
                print(
                    f"CRITICAL ERROR: Could not ensure 'Admin' role for superuser {self.email} in save(): {e}"
                )
                # Consider implications: superuser might lack Admin role if DB issue persists
            print(
                f"User.save: Superuser final role: {self.role}, is_staff: {self.is_staff}"
            )

        print(f"User.save: Calling super().save() for user {self.email}")
        super().save(*args, **kwargs)
        print(f"User.save: Finished saving user {self.email}")

    class Meta:
        db_table = "users"
        verbose_name = "User"
        verbose_name_plural = "Users"
        indexes = [
            models.Index(fields=["email", "created_at"]),
        ]

    def __str__(self):
        return self.email


class ProteinSequence(models.Model):
    """Model to store protein sequences uploaded by the user."""

    sequence_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="protein_sequences"
    )
    sequence_name = models.CharField(max_length=100, db_index=True)
    sequence_fasta = models.TextField()
    upload_date = models.DateTimeField(auto_now_add=True, db_index=True)
    status = models.CharField(max_length=20, db_index=True)
    description = models.TextField(blank=True, null=True)
    sequence_length = models.IntegerField(db_index=True)
    organism = models.CharField(max_length=100, blank=True, null=True, db_index=True)
    source = models.CharField(max_length=100, blank=True, null=True)
    search_vector = SearchVectorField(null=True)

    class Meta:
        db_table = "protein_sequences"
        verbose_name = "Protein Sequence"
        verbose_name_plural = "Protein Sequences"
        indexes = [
            models.Index(fields=["sequence_name", "upload_date"]),
            models.Index(fields=["organism", "sequence_length"]),
            GinIndex(fields=["search_vector"]),
        ]

    def __str__(self):
        return self.sequence_name


class Prediction(models.Model):
    """Model to store the predictions generated by the model."""

    prediction_id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    sequence = models.ForeignKey(
        ProteinSequence, on_delete=models.CASCADE, related_name="predictions"
    )
    pdb_file_path = models.CharField(max_length=255)
    pae_file_path = models.CharField(max_length=255, blank=True, null=True)
    plddt_score = models.FloatField(blank=True, null=True, db_index=True)
    prediction_date = models.DateTimeField(auto_now_add=True, db_index=True)
    status = models.CharField(max_length=20, db_index=True)
    confidence_score = models.FloatField(blank=True, null=True, db_index=True)
    model_version = models.CharField(max_length=50)
    prediction_metadata = models.JSONField(blank=True, null=True)

    class Meta:
        db_table = "predictions"
        verbose_name = "Prediction"
        verbose_name_plural = "Predictions"
        indexes = [
            models.Index(fields=["prediction_date", "status"]),
            models.Index(fields=["plddt_score", "confidence_score"]),
        ]

    def __str__(self):
        return f"Prediction for {self.sequence.sequence_name}"


class ValidationMetric(models.Model):
    """Model to store the validation metrics for the predictions."""

    metric_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prediction = models.ForeignKey(
        Prediction, on_delete=models.CASCADE, related_name="validation_metrics"
    )
    rmsd = models.FloatField(blank=True, null=True, db_index=True)
    rg = models.FloatField(blank=True, null=True, db_index=True)
    energy = models.FloatField(blank=True, null=True, db_index=True)
    trajectory_path = models.CharField(max_length=255, blank=True, null=True)
    structure_path = models.CharField(
        max_length=255, blank=True, null=True
    )  # .gro file path
    validation_date = models.DateTimeField(auto_now_add=True, db_index=True)
    status = models.CharField(max_length=20, db_index=True)
    simulation_parameters = models.JSONField(blank=True, null=True)
    stability_score = models.FloatField(blank=True, null=True, db_index=True)
    validation_notes = models.TextField(blank=True, null=True)
    modified_date = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "validation_metrics"
        verbose_name = "Validation Metric"
        verbose_name_plural = "Validation Metrics"
        indexes = [
            models.Index(fields=["validation_date", "status"]),
            models.Index(fields=["rmsd", "rg", "energy"]),
        ]

    def __str__(self):
        return f"Validation for {self.prediction}"

    @classmethod
    def get_status_for_prediction(cls, prediction_id):
        """
        Get the current simulation status for a prediction.
        This method provides a unified status that considers both ValidationMetric and JobQueue.
        """
        # Get the latest ValidationMetric for this prediction
        latest_vm = (
            cls.objects.filter(prediction_id=prediction_id)
            .order_by("-validation_date")
            .first()
        )

        if not latest_vm:
            return {
                "status": "not_found",
                "simulation_status": None,
                "message": "No simulation found for this prediction",
            }

        # Check for associated JobQueue entries
        job = (
            JobQueue.objects.filter(
                job_parameters__prediction_id=str(prediction_id),
                job_type="gromacs_simulation",
            )
            .order_by("-created_at")
            .first()
        )

        # Determine real status based on ValidationMetric and JobQueue state
        real_status = cls._determine_real_status(latest_vm, job)

        return {
            "status": "success",
            "simulation_status": real_status,
            "date": latest_vm.validation_date,
            "trajectory_path": latest_vm.trajectory_path,
            "notes": latest_vm.validation_notes,
            "metric_id": str(latest_vm.metric_id),
            "job_id": str(job.job_id) if job else None,
            "rmsd": latest_vm.rmsd,
            "rg": latest_vm.rg,
            "energy": latest_vm.energy,
            "stability_score": latest_vm.stability_score,
        }

    @classmethod
    def _determine_real_status(cls, validation_metric, job):
        """
        Determine the real status considering both ValidationMetric and JobQueue states.
        """
        if not validation_metric:
            return "not_found"

        vm_status = validation_metric.status

        # If no job exists but VM exists, trust VM status unless it's inconsistent
        if not job:
            # If VM shows pending/running but no job exists, mark as failed
            if vm_status in ["pending", "running"]:
                validation_metric.status = "failed"
                validation_metric.validation_notes = f"{validation_metric.validation_notes or ''}\nMarked as failed: No associated job found"
                validation_metric.save()
                return "failed"
            return vm_status

        job_status = job.status

        # If job is failed/completed, trust that over VM status
        if job_status == "failed":
            if vm_status not in ["failed"]:
                validation_metric.status = "failed"
                validation_metric.validation_notes = f"{validation_metric.validation_notes or ''}\nUpdated status based on job failure"
                validation_metric.save()
            return "failed"

        if job_status == "completed":
            if vm_status not in ["completed"]:
                validation_metric.status = "completed"
                validation_metric.save()
            return "completed"

        # If job is running, VM should be running too
        if job_status == "running":
            if vm_status not in ["running", "pending"]:
                validation_metric.status = "running"
                validation_metric.save()
            return "running"

        # For any other cases, trust VM status
        return vm_status

    def sync_with_job_queue(self):
        """
        Synchronize this ValidationMetric status with its associated JobQueue entry.
        """
        job = (
            JobQueue.objects.filter(
                job_parameters__prediction_id=str(self.prediction.prediction_id),
                job_type="gromacs_simulation",
            )
            .order_by("-created_at")
            .first()
        )

        if job:
            real_status = self._determine_real_status(self, job)
            if real_status != self.status:
                self.status = real_status
                self.save()
                return True
        return False

    def cleanup_stale_simulations(self):
        """
        Clean up stale simulations that have been running too long without updates.
        """
        from django.utils import timezone
        from datetime import timedelta

        stale_threshold = timezone.now() - timedelta(minutes=15)

        if (
            self.status in ["pending", "running"]
            and self.modified_date < stale_threshold
        ):

            self.status = "failed"
            self.validation_notes = f"{self.validation_notes or ''}\nMarked as failed due to inactivity (stale simulation cleanup)"
            self.save()

            # Also update associated job if it exists
            job = JobQueue.objects.filter(
                job_parameters__prediction_id=str(self.prediction.prediction_id),
                job_type="gromacs_simulation",
            ).first()

            if job and job.status in ["pending", "running"]:
                job.status = "failed"
                job.completed_at = timezone.now()
                job.save()

            return True
        return False


class MLRanking(models.Model):
    """Model to store the ML rankings for the predictions."""

    ranking_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prediction = models.ForeignKey(
        Prediction, on_delete=models.CASCADE, related_name="ml_rankings"
    )
    stability_score = models.FloatField(blank=True, null=True, db_index=True)
    solubility_score = models.FloatField(blank=True, null=True, db_index=True)
    binding_efficiency_score = models.FloatField(blank=True, null=True, db_index=True)
    ranking_date = models.DateTimeField(auto_now_add=True, db_index=True)
    overall_score = models.FloatField(blank=True, null=True, db_index=True)
    feature_importance = models.JSONField(blank=True, null=True)
    model_version = models.CharField(max_length=50)
    ranking_notes = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "ml_ranking"
        verbose_name = "ML Ranking"
        verbose_name_plural = "ML Rankings"
        indexes = [
            models.Index(fields=["ranking_date"]),
            models.Index(
                fields=[
                    "stability_score",
                    "solubility_score",
                    "binding_efficiency_score",
                ]
            ),
        ]

    def __str__(self):
        return f"Ranking for {self.prediction}"


class Log(models.Model):
    """Model to store the logs generated by the system."""

    log_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="logs", null=True, blank=True
    )
    action = models.CharField(max_length=100, db_index=True)
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    details = models.TextField(blank=True, null=True)
    ip_address = models.CharField(max_length=45, blank=True, null=True)
    session_id = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(max_length=20, db_index=True)
    component = models.CharField(max_length=50, db_index=True)

    class Meta:
        db_table = "logs"
        verbose_name = "Log"
        verbose_name_plural = "Logs"
        indexes = [
            models.Index(fields=["timestamp", "action"]),
            models.Index(fields=["status", "component"]),
        ]

    def __str__(self):
        return f"{self.action} by {self.user if self.user else 'System'}"


class SystemMetric(models.Model):
    """Model to store the system metrics."""

    metric_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    cpu_usage = models.FloatField(db_index=True)
    memory_usage = models.FloatField(db_index=True)
    disk_usage = models.FloatField(db_index=True)
    active_jobs = models.IntegerField(db_index=True)
    performance_metrics = models.JSONField(blank=True, null=True)
    status = models.CharField(max_length=20, db_index=True)

    class Meta:
        db_table = "system_metrics"
        verbose_name = "System Metric"
        verbose_name_plural = "System Metrics"
        indexes = [
            models.Index(fields=["timestamp"]),
            models.Index(fields=["cpu_usage", "memory_usage", "disk_usage"]),
        ]

    def __str__(self):
        return f"System Metric at {self.timestamp}"


class JobQueue(models.Model):
    """Model to store the jobs in the job queue."""

    job_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="jobs")
    job_type = models.CharField(max_length=50, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, db_index=True)
    job_parameters = models.JSONField(blank=True, null=True)
    priority = models.IntegerField(default=0, db_index=True)

    class Meta:
        db_table = "job_queue"
        verbose_name = "Job"
        verbose_name_plural = "Jobs"
        indexes = [
            models.Index(fields=["created_at", "status"]),
            models.Index(fields=["job_type", "priority"]),
        ]

    def __str__(self):
        return f"{self.job_type} - {self.status}"


# Signal handlers for ValidationMetric/JobQueue synchronization
@receiver(post_delete, sender=JobQueue)
def handle_job_deletion(sender, instance, **kwargs):
    """
    When a JobQueue entry is deleted, update associated ValidationMetric status.
    """
    if instance.job_type == "gromacs_simulation":
        prediction_id = instance.job_parameters.get("prediction_id")
        if prediction_id:
            try:
                vm = (
                    ValidationMetric.objects.filter(
                        prediction_id=prediction_id, status__in=["pending", "running"]
                    )
                    .order_by("-validation_date")
                    .first()
                )

                if vm:
                    vm.status = "failed"
                    vm.validation_notes = f"{vm.validation_notes or ''}\nSimulation failed: Associated job was deleted"
                    vm.save()
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error updating ValidationMetric after job deletion: {e}")


@receiver(post_save, sender=JobQueue)
def handle_job_status_change(sender, instance, created, **kwargs):
    """
    When a JobQueue status changes, synchronize ValidationMetric status.
    """
    if instance.job_type == "gromacs_simulation":
        prediction_id = instance.job_parameters.get("prediction_id")
        if prediction_id:
            try:
                vm = (
                    ValidationMetric.objects.filter(prediction_id=prediction_id)
                    .order_by("-validation_date")
                    .first()
                )

                if vm:
                    status_changed = False

                    if instance.status == "failed" and vm.status not in [
                        "failed",
                        "completed",
                    ]:
                        vm.status = "failed"
                        vm.validation_notes = (
                            f"{vm.validation_notes or ''}\nJob status updated to failed"
                        )
                        status_changed = True

                    elif instance.status == "running" and vm.status == "pending":
                        vm.status = "running"
                        vm.validation_notes = (
                            f"{vm.validation_notes or ''}\nSimulation started"
                        )
                        status_changed = True

                    if status_changed:
                        vm.save()

            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(
                    f"Error synchronizing ValidationMetric with job status: {e}"
                )
