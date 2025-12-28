import torch.nn as nn
from torch import Tensor
import math
import torch

# Embed a numericalized set of sentences
# Inputs: numericalized sentence Tensor, dimension of embeddings (d_model), len(vocab)
# def embedding(sentence_ids, d_model, vocab_size):
#   embedding_layer = torch.nn.Embedding(vocab_size, d_model)
#   return embedding_layer(sentence_ids)

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
    
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, vocab_size_src, vocab_size_tgt, dropout=0.1):
        super(Transformer, self).__init__()

        # Transformer model definition
        # Takes variables for d_model, number of attention heads, num of encoder and decoder layers, 
        # dimenson of feedforward network (d_ff), and dropout percentage
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        

        self.embedding_src = nn.Embedding(vocab_size_src, d_model)
        self.embedding_tgt = nn.Embedding(vocab_size_tgt, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size_tgt)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src_emb = self.pos_encoder(self.embedding_src(src))
        print("SRC EMB SHAPE:", src_emb.shape)
        tgt_emb = self.pos_encoder(self.embedding_tgt(tgt))
        print("TGT EMB SHAPE:", tgt_emb.shape)
        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  memory_mask=memory_mask)
        output = self.fc_out(output)
        return output
    