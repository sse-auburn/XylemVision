import json
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(is_safe=True)
def tojson(value):
    """Serialize a Python value to a JSON string safe for use in <script> tags."""
    return mark_safe(json.dumps(value))
