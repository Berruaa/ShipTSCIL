import torch
import torch.nn.functional as F
from tqdm import tqdm


def _unpack_batch(batch, model, device, allow_grad=False):
    """Return (embeddings, labels) regardless of batch format.

    Accepts either precomputed 2-tuples ``(embedding, y)`` or raw
    3-tuples ``(x, mask, y)`` (the encoder is called on-the-fly for the
    latter, though this path should rarely be hit after precomputation).

    When *allow_grad* is True the encoder forward pass retains the
    computation graph so that gradients can reach trainable encoder
    parameters (e.g. LoRA adapters).
    """
    if len(batch) == 2:
        emb, y = batch
        return emb.to(device).float(), y.to(device)

    x, mask, y = batch
    x = x.to(device).float()
    mask = mask.to(device)
    if allow_grad:
        emb = model.encoder(x, mask)
    else:
        with torch.no_grad():
            emb = model.encoder(x, mask)
    return emb, y.to(device)


def train_one_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    model.encoder.eval()

    encoder_trainable = any(p.requires_grad for p in model.encoder.parameters())

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(
        dataloader,
        total=len(dataloader),
        desc="Train",
        leave=False,
        disable=True,
    ):
        emb, batch_y = _unpack_batch(batch, model, device, allow_grad=encoder_trainable)

        optimizer.zero_grad()
        logits = model.head(emb)
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
    return {"loss": avg_loss, "acc": avg_acc}


@torch.no_grad()
def evaluate(model, dataloader, device="cpu"):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(
        dataloader,
        total=len(dataloader),
        desc="Eval",
        leave=False,
        disable=True,
    ):
        emb, batch_y = _unpack_batch(batch, model, device)

        logits = model.head(emb)
        loss = F.cross_entropy(logits, batch_y)

        preds = torch.argmax(logits, dim=1)
        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return {"loss": avg_loss, "acc": avg_acc}
