import torch.nn as nn
from torch import Tensor
import math
import torch

# Embed a numericalized set of sentences
# Inputs: numericalized sentence Tensor, dimension of embeddings (d_model), len(vocab)
def embedding(sentence_ids, d_model, vocab_size):
  embedding_layer = torch.nn.Embedding(vocab_size, d_model)
  return embedding_layer(sentence_ids)

# Class for creating positional encodings to be applied to input embeddings (respectively)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)