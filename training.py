from dataset import load_tokenizer, tokenize, numericalize_and_pad
from model import Transformer
import torch

from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
import pandas as pd

def testMethod():
    load_tokenizer()
    ds = load_dataset("okezieowen/english_to_spanish")
    selected_data = ds['train'].select(range(10))
    df = selected_data.to_pandas()
    
    # Tokenize all sentences in both English and Spanish columns
    all_tokens_english = [tokenize(sentence, 'en') for sentence in df['English'] if sentence is not None]
    all_tokens_spanish = [tokenize(sentence, 'es') for sentence in df['Spanish'] if sentence is not None]
    
    # Build vocabularies for both languages
    SPECIALS = ["<unk>", "<pad>", "<bos>", "<eos>"]

    vocab_english = build_vocab_from_iterator(
        all_tokens_english,
        min_freq=1,
        specials=SPECIALS,
        special_first=True
    )

    vocab_spanish = build_vocab_from_iterator(
        all_tokens_spanish,
        min_freq=1,
        specials=SPECIALS,
        special_first=True
    )

    # Numericalize and pad sentences
    numericalize_and_padded_english = numericalize_and_pad(vocab_english, all_tokens_english, sentence_size=100)
    numericalize_and_padded_spanish = numericalize_and_pad(vocab_spanish, all_tokens_spanish, sentence_size=100)

    # ---- MODEL.PY ---- #
    # Test transformer output 
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    vocab_size_src = len(vocab_english)
    vocab_size_tgt = len(vocab_spanish)
    dropout = 0.1

    model = Transformer(d_model, nhead, num_encoder_layers,
                        num_decoder_layers, dim_feedforward,
                        vocab_size_src, vocab_size_tgt, dropout)

    src = numericalize_and_padded_english
    tgt = numericalize_and_padded_spanish
    
    output = model(src, tgt)
    
    print("OUTPUT SHAPE:", output.shape)

    output_probabilities = torch.softmax(output, dim=-1)

    print("OUTPUT PROBABILITIES SHAPE:", output_probabilities.shape)
    print(output_probabilities[0])

    last_step_probs = output_probabilities[-1, -1, :]
    print("LAST STEP PROBABILITIES SHAPE:", last_step_probs.shape)

    # To show that the untrained model is trying to predict most likely next word probabilities,
    # I will show that it produces output probabilities when given input sentences 

    k = 10

    topk_probs, topk_indices = torch.topk(last_step_probs, k)
    vocab_spanish_itos = vocab_spanish.get_itos()
    topk_tokens = [vocab_spanish_itos[idx] for idx in topk_indices.tolist()]

    # Top k spanish word probabilities for the last word in the last sentence of the batch
    for token, prob in zip(topk_tokens, topk_probs.tolist()):
        print(f"Token: {token}, Probability: {prob}")
    
def train():
    pass

# TO DO:
# Making a training loop that iterates over batches of every sentence in the huggingface data.
# Objective of this: Train model to minimize loss and find way to measure accuracy / BLEU score over timesteps - base
# testing off of 'Attention Is All You Need'

# Run testing / validation method to verify that trained model is functional and BLEU score is consistent with unseen data

if __name__ == "__main__":
    testMethod()
