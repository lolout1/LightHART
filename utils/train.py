import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

def run_training(
    model, train_dl, valid_dl,
    criterion, metrics, callbacks,
    optim_cfg, sched_cfg, device, epochs
):
    opt    = AdamW(model.parameters(), **optim_cfg)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **sched_cfg) if sched_cfg else None
    scaler = GradScaler(enabled=device.startswith("cuda"))

    best_val = float("inf")
    for epoch in range(1, epochs+1):
        # ——— Train
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            with autocast(enabled=bool(scaler)):
                preds = model(xb)
                loss  = criterion(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()

        # ——— Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(valid_dl)

        # Callbacks
        for cb in callbacks:
            cb.on_epoch_end(epoch, val_loss, model)

        if sched:
            sched.step(val_loss)

    return callbacks.best_model

