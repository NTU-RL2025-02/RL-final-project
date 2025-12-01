import json
import numpy as np


def convert_json(obj):
    """
    Best-effort conversion to something JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: convert_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
