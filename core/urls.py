from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("sequences/", views.ProteinSequenceListView.as_view(), name="sequence_list"),
    path(
        "sequences/new/",
        views.ProteinSequenceCreateView.as_view(),
        name="sequence_create",
    ),
    path(
        "sequences/<uuid:pk>/",
        views.ProteinSequenceDetailView.as_view(),
        name="sequence_detail",
    ),
]
