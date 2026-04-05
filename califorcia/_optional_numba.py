try:
    from numba import njit as _njit
except ImportError:
    _njit = None


def optional_njit(*args, **kwargs):
    if _njit is None:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    return _njit(*args, **kwargs)
