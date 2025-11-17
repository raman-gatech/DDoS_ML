
import os
import json
import random
import numpy as np

from config import CFG

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
