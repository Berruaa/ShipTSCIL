import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    model.encoder.eval()  # frozen encoder

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_mask, batch_y in tqdm(
        dataloader,
        total=len(dataloader),
        desc="Train",
        leave=False,
        disable=True,
    ):
        batch_x = batch_x.to(device).float()
        batch_mask = batch_mask.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        logits, _ = model(batch_x, batch_mask)
        loss = F.cross_entropy(logits, batch_y)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return {
        "loss": avg_loss,
        "acc": avg_acc,
    }


@torch.no_grad()
def evaluate(model, dataloader, device="cpu"):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_mask, batch_y in tqdm(
        dataloader,
        total=len(dataloader),
        desc="Eval",
        leave=False,
        disable=True,
    ):
        batch_x = batch_x.to(device).float()
        batch_mask = batch_mask.to(device)
        batch_y = batch_y.to(device)

        logits, _ = model(batch_x, batch_mask)
        loss = F.cross_entropy(logits, batch_y)

        preds = torch.argmax(logits, dim=1)

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return {
        "loss": avg_loss,
        "acc": avg_acc,
    }