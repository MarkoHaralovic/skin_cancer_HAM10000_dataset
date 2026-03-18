import torch


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, log_interval=100):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_iters = len(data_loader)

    for i, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss    += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if i % log_interval == 0 or i == num_iters:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(f'[epoch {epoch + 1}], [iter {i} / {num_iters}], '
                  f'[train loss {avg_loss:.5f}], [train acc {accuracy:.5f}]')

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return {'loss': avg_loss, 'accuracy': accuracy * 100}