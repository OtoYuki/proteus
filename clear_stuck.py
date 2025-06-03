#!/usr/bin/env python
import os
import sys
import django

# Setup Django
sys.path.append("/home/sire/proteus")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "proteus.settings")
django.setup()

from core.models import ValidationMetric


def main():
    print("=== Clearing Stuck Simulations ===")

    # Find all stuck simulations
    stuck_sims = ValidationMetric.objects.filter(status__in=["running", "pending"])
    total_stuck = stuck_sims.count()

    print(f"Found {total_stuck} stuck simulations")

    if total_stuck > 0:
        # Show details before clearing
        for sim in stuck_sims:
            print(
                f"  - {sim.metric_id}: {sim.status} (prediction: {sim.prediction.prediction_id})"
            )

        # Clear them one by one to see progress
        cleared_count = 0
        for sim in stuck_sims:
            sim.status = "failed"
            sim.validation_notes = "Cleared stuck simulation to allow new runs"
            sim.save()
            cleared_count += 1
            print(f"  ✓ Cleared: {sim.metric_id}")

        print(f"\nCleared {cleared_count} stuck simulations")
    else:
        print("No stuck simulations found")

    # Verify
    remaining = ValidationMetric.objects.filter(
        status__in=["running", "pending"]
    ).count()
    print(f"Remaining running/pending simulations: {remaining}")

    if remaining == 0:
        print("✅ All clear! You should now be able to run new simulations.")
    else:
        print("⚠️  Some simulations are still stuck")


if __name__ == "__main__":
    main()
