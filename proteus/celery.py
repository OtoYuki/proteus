import os
from celery import Celery


# Setting the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "proteus.settings")

# Creating a Celery instance and configuring it with the Django settings.
app = Celery("proteus")

# Loacing the settings from the Django settings module. with the namespace 'CELERY'.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Autodiscovering tasks from all registered Django app configs.
app.autodiscover_tasks()
