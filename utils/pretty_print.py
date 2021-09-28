from typing import Iterable


def is_float(v):
    """if v is a scalar"""
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def is_iterable(v):
    """if v is an iterable, except str"""
    if isinstance(v, str):
        return False
    return isinstance(v, (list, tuple, dict, set))


def _float2str(v):
    """convert a scalar to float, in order to display"""
    v = float(v)
    if abs(float(v)) < 0.01 or abs(float(v)) >= 99:
        return f"{v:.2e}"
    return f"{v:.3f}"


def _leafitem2str(v):
    if is_float(v):
        return _float2str(v)
    return f"{v}"


def _generate_pair(k, v):
    """generate str for non iterable k v"""
    return f"{k}:{_leafitem2str(v)}"


def _dict2str(dictionary: dict):
    def create_substring(k, v):
        if not is_iterable(v):
            return _generate_pair(k, v)
        else:
            return f"{k}:[" + item2str(v) + "]"

    strings = [create_substring(k, v) for k, v in dictionary.items()]
    return ", ".join(strings)


def _iter2str(item: Iterable):
    """A list or a tuple"""
    return ", ".join(
        [_leafitem2str(x) if not is_iterable(x) else item2str(x) for x in item]
    )


def item2str(item):
    """convert item to string in a pretty way.
        @param item: list, dictionary, set and tuple
        @return: pretty string
    """
    if isinstance(item, dict):
        return _dict2str(item)
    return _iter2str(item)