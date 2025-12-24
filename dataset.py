from datasets import load_dataset
import pandas as pd
import spacy
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn
from torch import Tensor
import math

# Tokenize function -> Takes in raw text dataset(individual column) and string 'en' or 'es' to specify english or spanish
def tokenize(text, language):
  if language == 'en':
    spacy_en = spacy.load("en_core_web_sm")
    return [token.text for token in spacy_en.tokenizer(text)]
  elif language == 'es':
    spacy_es = spacy.load("es_core_news_sm")
  return [token.text for token in spacy_es.tokenizer(text)]


# Numericalize the all_tokens array
# Inputs: Vocab object (torchtext), list of tokenized sentences, size of each sentence
def numericalize_and_pad(vocab, list_of_token_lists, sentence_size):
  numericalized_unpadded = [vocab(tokens) for tokens in list_of_token_lists]
  # For each sentence, subtract max size from size of that token and create array of size difference full of padding index
  # Pad index is concatenated to numericalized sentence
  # Return tensor full of numericalized and padded sentences
  numericalized_sentences = []
  for sentence in numericalized_unpadded:
    if len(sentence) > sentence_size:
      sentence = sentence[:sentence_size]
      numericalized_sentences.append(sentence)
    else:
      padding_size = sentence_size - len(sentence)
      padding = [vocab["<pad>"] for _ in range(padding_size)]
      sentence.extend(padding)
      numericalized_sentences.append(sentence)


  return torch.tensor(numericalized_sentences)

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