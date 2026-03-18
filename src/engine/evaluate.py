import torch
from ..loss.metrics import get_metrics

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_loss    += loss.item() * images.size(0)
        total_samples += images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / total_samples
    metrics = get_metrics(y_true, y_pred)
    metrics['loss'] = avg_loss

    print(f"Eval  loss: {avg_loss:.4f}  "
          f"acc: {metrics['accuracy']:.2f}%  "
          f"macro_f1: {metrics['macro_f1']:.4f}  "
          f"balanced_acc: {metrics['balanced_acc']:.4f}")
    return metrics