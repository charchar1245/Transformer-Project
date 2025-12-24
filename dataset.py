from datasets import load_dataset
import pandas as pd
import spacy
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch import Tensor
import math

# Run this before running tokenize function
def load_tokenizer():
  spacy_en = spacy.load("en_core_web_sm")
  spacy_es = spacy.load("es_core_news_sm")

# Tokenize function -> Takes in raw text dataset(individual column) and string 'en' or 'es' to specify english or spanish
def tokenize(text, language):
  if language == 'en':
    return [token.text for token in spacy_en.tokenizer(text)]
  elif language == 'es':
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
