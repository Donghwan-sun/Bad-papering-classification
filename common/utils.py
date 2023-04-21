import torch
def device():
    result = "cuda" if torch.cuda.is_available() else "cpu"
    return result