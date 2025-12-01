import os
import time
from pathlib import Path


def setup_logger_kwargs(exp_name, seed=0, data_dir=None):
    """
    Create a logging directory like data_dir/exp_name/exp_name_s{seed}/
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(data_dir or "./data")
    output_dir = base / exp_name / f"{exp_name}_s{seed}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return dict(output_dir=str(output_dir), exp_name=exp_name)
