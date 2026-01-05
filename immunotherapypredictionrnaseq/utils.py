import random
import numpy as np
import torch

def fixseed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_params_and_gradients(model):
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError("Bad parameter:", name)

    answers = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                answers.append(f"NaN/Inf in gradient of {name}")
    if len(answers) > 0:
        raise RuntimeError("\n".join(answers))
