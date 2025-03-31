from django import template
import json

register = template.Library()


@register.filter
def pprint(value):
    """Pretty print JSON data in templates"""
    try:
        return json.dumps(value, indent=2)
    except:
        return value
