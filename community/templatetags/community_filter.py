from django import template

register = template.Library()

@register.filter
def sub(value, arg):
    return value - arg

@register.filter(name='get_item')
def get_item(list, index):
    try:
        return list[index]
    except IndexError:
        return None

@register.filter(name='to_float')
def to_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0
