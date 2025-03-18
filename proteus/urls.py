"""
URL configuration for proteus project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from core import views
from core.views import (
    submit_sequence,
    home,
    prediction_list,
    prediction_detail,
    serve_pdb,
    start_gromacs_simulation,
    simulation_status,
)

urlpatterns = [
    path("", home, name="home"),  # Root URL
    path("admin/", admin.site.urls),
    # Include the URLs from the core app
    path("submit/", submit_sequence, name="submit_sequence"),
    path("predictions/", prediction_list, name="prediction_list"),
    path(
        "predictions/<uuid:prediction_id>/", prediction_detail, name="prediction_detail"
    ),
    path("pdb/<uuid:prediction_id>/", serve_pdb, name="serve_pdb"),
    # GROMACS simulation URLs
    path(
        "predictions/<uuid:prediction_id>/simulate/",
        start_gromacs_simulation,
        name="start_simulation",
    ),
    path(
        "predictions/<uuid:prediction_id>/simulation_status/",
        simulation_status,
        name="simulation_status",
    ),
    # Download Trajectory URL
    path(
        "download_trajectory/<uuid:prediction_id>/",
        views.download_trajectory,
        name="download_trajectory",
    ),
]
