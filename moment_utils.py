import numpy as np
import torch
from tqdm import tqdm
from momentfm import MOMENTPipeline


def load_moment_model(model_name="AutonLab/MOMENT-1-base", device="cpu"):
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.to(device).float()
    model.eval()
    return model


def get_embeddings(model, dataloader, device="cpu"):
    embeddings, labels = [], []

    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to(device).float()
            batch_masks = batch_masks.to(device)

            output = model(x_enc=batch_x, input_mask=batch_masks)
            embeddings.append(output.embeddings.detach().cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(embeddings), np.concatenate(labels)