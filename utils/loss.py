import torch.nn as nn

def make_loss(cfg: dict):
    if cfg["type"] == "bce":
        return nn.BCEWithLogitsLoss()
    elif cfg["type"] == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {cfg['type']}")

