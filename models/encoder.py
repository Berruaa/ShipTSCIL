import torch
import torch.nn as nn
from momentfm import MOMENTPipeline


class FrozenMomentEncoder(nn.Module):
    def __init__(self, model_name="AutonLab/MOMENT-1-base"):
        super().__init__()

        self.model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"},
        )
        self.model.init()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward(self, x, input_mask):
        with torch.no_grad():
            output = self.model(x_enc=x, input_mask=input_mask)
            embeddings = output.embeddings
        return embeddings