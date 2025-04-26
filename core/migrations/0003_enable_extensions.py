from django.contrib.postgres.operations import CreateExtension
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0002_validationmetric_modified_date"),
    ]

    operations = [
        CreateExtension("citext"),
        CreateExtension("btree_gin"),
    ]
