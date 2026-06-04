import os
import re
import torch

def load_shapes_as_torch_size(path):
    import json

    if not os.path.exists(path):
        return {}   # or return {}
    
    with open(path, "r") as f:
        data = json.load(f)

    shapes = {
        key: torch.Size(shape)   # convert list -> torch.Size
        for key, shape in data.items()
    }

    return shapes