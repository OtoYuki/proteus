from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _
from .models import (
    Role,
    User,
    ProteinSequence,
    Prediction,
    ValidationMetric,
    MLRanking,
    Log,
    SystemMetric,
    JobQueue,
)


# Register your models here.
class UserAdmin(BaseUserAdmin):
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        (_("Personal info"), {"fields": ("first_name", "last_name", "role")}),
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                ),
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined", "created_at")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "password1", "password2", "role"),
            },
        ),
    )
    list_display = ("email", "first_name", "last_name", "role", "is_staff")
    search_fields = ("email", "first_name", "last_name")
    ordering = ("email",)
    list_filter = ("is_staff", "is_superuser", "is_active", "groups", "role")


# Register all models
admin.site.register(User, UserAdmin)
admin.site.register(Role)
admin.site.register(ProteinSequence)
admin.site.register(Prediction)
admin.site.register(ValidationMetric)
admin.site.register(MLRanking)
admin.site.register(Log)
admin.site.register(SystemMetric)
admin.site.register(JobQueue)
