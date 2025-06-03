from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from core.models import ValidationMetric, JobQueue
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Clean up stale simulations and synchronize ValidationMetric with JobQueue status"

    def add_arguments(self, parser):
        parser.add_argument(
            "--stale-minutes",
            type=int,
            default=5,
            help="Minutes to consider a simulation stale (default: 5)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be cleaned up without actually doing it",
        )

    def handle(self, *args, **options):
        stale_minutes = options["stale_minutes"]
        dry_run = options["dry_run"]

        self.stdout.write(
            f"Cleaning up simulations stale for more than {stale_minutes} minutes..."
        )

        if dry_run:
            self.stdout.write(
                self.style.WARNING("DRY RUN MODE - No changes will be made")
            )

        # Find stale ValidationMetric entries
        stale_threshold = timezone.now() - timedelta(minutes=stale_minutes)
        stale_vms = ValidationMetric.objects.filter(
            status__in=["pending", "running"], modified_date__lt=stale_threshold
        )

        cleaned_count = 0
        synced_count = 0

        for vm in stale_vms:
            if dry_run:
                self.stdout.write(
                    f"Would mark as failed: {vm.metric_id} (prediction: {vm.prediction.prediction_id})"
                )
            else:
                if vm.cleanup_stale_simulations():
                    cleaned_count += 1
                    self.stdout.write(f"Marked as failed: {vm.metric_id}")

        # Synchronize all active ValidationMetrics with JobQueue
        active_vms = ValidationMetric.objects.filter(status__in=["pending", "running"])

        for vm in active_vms:
            if dry_run:
                # Check what would be synchronized
                job = (
                    JobQueue.objects.filter(
                        job_parameters__prediction_id=str(vm.prediction.prediction_id),
                        job_type="gromacs_simulation",
                    )
                    .order_by("-created_at")
                    .first()
                )

                if job:
                    real_status = ValidationMetric._determine_real_status(vm, job)
                    if real_status != vm.status:
                        self.stdout.write(
                            f"Would sync status: {vm.metric_id} from {vm.status} to {real_status}"
                        )
            else:
                if vm.sync_with_job_queue():
                    synced_count += 1
                    self.stdout.write(f"Synchronized: {vm.metric_id}")

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry run completed"))
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Cleanup completed: {cleaned_count} stale simulations marked as failed, "
                    f"{synced_count} simulations synchronized"
                )
            )
