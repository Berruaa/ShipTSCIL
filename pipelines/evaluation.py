import numpy as np
import torch

from pipelines.data import build_loader, make_class_subset


@torch.no_grad()
def collect_predictions_torch_model(model, dataloader, device):
    model.eval()

    preds_list = []
    targets_list = []

    for batch_x, batch_mask, batch_y in dataloader:
        batch_x = batch_x.to(device).float()
        batch_mask = batch_mask.to(device)

        logits, _ = model(batch_x, batch_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        preds_list.append(preds)
        targets_list.append(batch_y.numpy())

    y_pred = np.concatenate(preds_list)
    y_true = np.concatenate(targets_list)
    return y_true, y_pred


def collect_predictions(method, dataloader, device):
    if hasattr(method, "predict"):
        return method.predict(dataloader)

    if hasattr(method, "model"):
        return collect_predictions_torch_model(method.model, dataloader, device=device)

    raise ValueError("Method does not support prediction.")


def evaluate_on_seen_classes(method, test_dataset, seen_classes, config):
    eval_subset = make_class_subset(test_dataset, seen_classes)
    eval_loader = build_loader(
        dataset=eval_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    return method.evaluate(eval_loader), eval_loader
