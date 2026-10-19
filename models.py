"""Two character-level models for ABC notation."""
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, width=128, layers=2, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.lstm = nn.LSTM(width, width, layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        hidden, _ = self.lstm(self.embedding(tokens))
        return self.output(hidden)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, width=128, layers=2, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(max_length, width)
        # Separate layers get separate initial weights.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=2*width,
                dropout=0.1, batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        # A position can use earlier characters, but not future characters.
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                    device=tokens.device), diagonal=1)
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask)
        return self.output(self.norm(hidden))


def build_model(kind, vocab_size, width):
    model_class = {'lstm': LSTMModel, 'transformer': TransformerModel}[kind]
    return model_class(vocab_size, width=width)
