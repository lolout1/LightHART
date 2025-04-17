import torch

def evaluate_model(model, dataloader, metrics, device):
    model.eval()
    results = {m.__name__: [] for m in metrics}
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            for m in metrics:
                results[m.__name__].append(m(preds, yb))
    # Print averages
    print({k: sum(v)/len(v) for k,v in results.items()})

