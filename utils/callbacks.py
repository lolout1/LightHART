import os
import torch

class EarlyStopping:
    def __init__(self, patience, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best = 0, float("inf")
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best - self.min_delta:
            self.best, self.counter = loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class ModelCheckpoint:
    def __init__(self, dirpath, monitor="val_loss"):
        self.dirpath, self.monitor = dirpath, monitor
        self.best, self.best_model = float("inf"), None

    def __call__(self, epoch, metric, model):
        if metric < self.best:
            self.best = metric
            path = os.path.join(self.dirpath, f"best_epoch{epoch}.pth")
            torch.save(model.state_dict(), path)
            self.best_model = model

def make_callbacks(workdir, cfg):
    cbs = []
    cbs.append(ModelCheckpoint(workdir, monitor=cfg["monitor"]))
    cbs.append(EarlyStopping(patience=cfg["patience"]))
    return cbs

