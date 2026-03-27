import torch.nn as nn


class MomentModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x, input_mask):
        embeddings = self.encoder(x, input_mask)
        logits = self.head(embeddings)
        return logits, embeddings