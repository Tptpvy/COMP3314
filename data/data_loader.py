import torch
import torchtext
from torchtext.data import Field, BucketIterator
import spacy
import random

def load_multi30k(batch_size=128, device='cuda'):
    """
    Load Multi30k dataset for German to English translation
    """
    # Define tokenizers
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    # Define fields
    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    
    # Load Multi30k dataset
    try:
        # For newer torchtext versions
        from torchtext.legacy import datasets
        train_data, valid_data, test_data = datasets.Multi30k.splits(
            exts=('.de', '.en'), fields=(SRC, TRG)
        )
    except:
        # For older torchtext versions
        train_data, valid_data, test_data = torchtext.datasets.Multi30k.splits(
            exts=('.de', '.en'), fields=(SRC, TRG)
        )
    
    # Build vocabulary
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    
    print(f"German vocabulary size: {len(SRC.vocab)}")
    print(f"English vocabulary size: {len(TRG.vocab)}")
    print(f"Number of training examples: {len(train_data)}")
    print(f"Number of validation examples: {len(valid_data)}")
    print(f"Number of test examples: {len(test_data)}")
    
    # Create iterators
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device
    )
    
    return train_iterator, valid_iterator, test_iterator, SRC, TRG

def tokenize_sentence(sentence, field, device):
    """
    Tokenize a single sentence for inference
    """
    tokens = [field.init_token] + field.tokenize(sentence.lower()) + [field.eos_token]
    indices = [field.vocab.stoi[token] for token in tokens]
    return torch.tensor(indices, device=device).unsqueeze(0)