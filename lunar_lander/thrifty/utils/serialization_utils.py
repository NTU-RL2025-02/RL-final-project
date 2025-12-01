import json
import numpy as np
import logging


def convert_json(obj, _visited=None):
    """
    Convert obj to something JSON-serializable.

    - Handles numpy scalars/arrays
    - Avoids infinite recursion via cycle detection
    - Treats logging.Logger specially
    """
    if _visited is None:
        _visited = set()

    obj_id = id(obj)
    if obj_id in _visited:
        # break circular references (env -> logger -> env ...)
        return f"<circular-ref:{type(obj).__name__}>"
    _visited.add(obj_id)

    # Already JSON-native: leave it alone
    if is_json_serializable(obj):
        return obj

    # Numpy types
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # logging.Logger is nasty to introspect => summarize instead
    if isinstance(obj, logging.Logger):
        return {"logger_name": obj.name, "logger_level": obj.level}

    # dict -> dict
    if isinstance(obj, dict):
        return {
            convert_json(k, _visited): convert_json(v, _visited) for k, v in obj.items()
        }

    # tuple -> list (JSON has no tuple)
    if isinstance(obj, tuple):
        return [convert_json(x, _visited) for x in obj]

    # list -> list
    if isinstance(obj, list):
        return [convert_json(x, _visited) for x in obj]

    # Named objects (functions, classes) → use their name
    if hasattr(obj, "__name__") and "lambda" not in obj.__name__:
        return convert_json(obj.__name__, _visited)

    # Generic objects with a __dict__: serialize their attributes
    if hasattr(obj, "__dict__") and obj.__dict__:
        obj_dict = {
            convert_json(k, _visited): convert_json(v, _visited)
            for k, v in obj.__dict__.items()
        }
        # avoid calling str(obj), which can recurse (e.g. in logging)
        return {type(obj).__name__: obj_dict}

    # Fallback: repr
    try:
        return repr(obj)
    except Exception:
        return f"<unreprable:{type(obj).__name__}>"


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False
