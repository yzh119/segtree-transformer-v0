import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    "Position Encoding module"
    def __init__(self, dim_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = th.zeros(max_len, dim_model, dtype=th.float)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, dim_model, 2, dtype=th.float) *
                             -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Not a parameter but should be in state_dict

    def forward(self, pos):
        return th.index_select(self.pe, 1, pos).squeeze(0)


class Embedding(nn.Module):
    "Word Embedding module"
    def __init__(self, vocab_size, dim_model, scale=False):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, dim_model)
        self.dim_model = dim_model
        self.scale = scale

    def forward(self, x):
        if self.scale:
            return self.lut(x) * np.sqrt(self.dim_model)
        else:
            return self.lut(x) 


class Embed2Hidden(nn.Module):
    def __init__(self, embed_size, dim_model):
        super(Embed2Hidden, self).__init__()
        self.affine = nn.Linear(embed_size, dim_model)

    def forward(self, x):
        return self.affine(x)
