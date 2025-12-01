import os
import os.path as osp
import time
import json
import atexit

from .serialization_utils import convert_json


class EpochLogger:
    """
    Lightweight logger similar to the one in spinup.
    Stores tabular metrics to progress.txt inside an output directory.
    """

    def __init__(self, output_dir=None, output_fname="progress.txt", exp_name=None):
        self.output_dir = output_dir or f"/tmp/experiments/{int(time.time())}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = open(osp.join(self.output_dir, output_fname), "w")
        atexit.register(self.output_file.close)
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        elif key not in self.log_headers:
            self.log_headers.append(key)
        self.log_current_row[key] = val

    def save_config(self, config):
        cfg = convert_json(config)
        if self.exp_name is not None:
            cfg["exp_name"] = self.exp_name
        with open(osp.join(self.output_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def save_state(self, state_dict, itr=None):
        fname = osp.join(self.output_dir, "vars.pkl" if itr is None else f"vars{itr}.pkl")
        import pickle

        with open(fname, "wb") as f:
            pickle.dump(state_dict, f)

    def dump_tabular(self):
        vals = [self.log_current_row.get(k, "") for k in self.log_headers]
        if self.first_row:
            self.output_file.write("\t".join(self.log_headers) + "\n")
        self.output_file.write("\t".join(map(str, vals)) + "\n")
        self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

    # shim to keep compatibility with original API
    def setup_pytorch_saver(self, _):
        pass
