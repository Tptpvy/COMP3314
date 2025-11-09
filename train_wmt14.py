# train_wmt14_hf.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
from tqdm import tqdm
import math
import time
from nltk.translate.bleu_score import corpus_bleu
from datasets import load_dataset
import os

from model.transformer import Transformer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizers
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except:
    print("Please install spaCy models:")
    print("python -m spacy download de_core_news_sm")
    print("python -m spacy download en_core_web_sm")
    raise

# Tokenization functions
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

# Define fields
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

# Load WMT14 dataset from Hugging Face
print("Loading WMT14 dataset from Hugging Face...")
try:
    # Load the dataset
    dataset = load_dataset("wmt14", "de-en")
    
    print("WMT14 dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    print(f"Test examples: {len(dataset['test'])}")
    
    # Convert to torchtext format by creating temporary files
    os.makedirs('.data/wmt14/wmt14_hf', exist_ok=True)
    
    # Write data to files for torchtext
    for split in ['train', 'validation', 'test']:
        # Convert Hugging Face split names to torchtext names
        torchtext_split = 'val' if split == 'validation' else split
        de_file = f'.data/wmt14/wmt14_hf/{torchtext_split}.de'
        en_file = f'.data/wmt14/wmt14_hf/{torchtext_split}.en'
        
        with open(de_file, 'w', encoding='utf-8') as f_de, \
            open(en_file, 'w', encoding='utf-8') as f_en:
            
            for example in dataset[split]:
                f_de.write(example['translation']['de'] + '\n')
                f_en.write(example['translation']['en'] + '\n')
    
    print("Converted Hugging Face dataset to torchtext format")
    
except Exception as e:
    print(f"Error loading WMT14 from Hugging Face: {e}")
    print("Please install datasets library: pip install datasets")
    exit(1)

# Now load with torchtext using the converted files
from torchtext.datasets import TranslationDataset

class WMT14Dataset(TranslationDataset):
    """Custom dataset class for WMT14"""
    
    urls = []
    name = 'wmt14'
    dirname = 'wmt14_hf'
    
    @classmethod
    def splits(cls, exts, fields, root='.data', **kwargs):
        return super().splits(exts, fields, root=root, **kwargs)

# Load the dataset
print("Loading WMT14 with torchtext...")
train_data, valid_data, test_data = WMT14Dataset.splits(
    exts=('.de', '.en'), 
    fields=(SRC, TRG),
    root='.data'
)

print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(valid_data)}")
print(f"Test examples: {len(test_data)}")

# Show some examples
print("\nSample examples from WMT14:")
for i in range(3):
    print(f"Example {i+1}:")
    print(f"  German: {train_data[i].src[:100]}...")
    print(f"  English: {train_data[i].trg[:100]}...")
    print()

# Build vocabulary with larger size for WMT14
print("Building vocabulary...")
SRC.build_vocab(train_data, min_freq=2, max_size=50000)
TRG.build_vocab(train_data, min_freq=2, max_size=50000)

print(f"Source vocabulary size: {len(SRC.vocab)}")
print(f"Target vocabulary size: {len(TRG.vocab)}")

# Model hyperparameters (larger for WMT14)
src_vocab_size = len(SRC.vocab)
trg_vocab_size = len(TRG.vocab)
src_pad_idx = SRC.vocab.stoi['<pad>']
trg_pad_idx = TRG.vocab.stoi['<pad>']
embed_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 6
dropout = 0.1
max_length = 300

print(f"Source vocab size: {src_vocab_size}")
print(f"Target vocab size: {trg_vocab_size}")

# Create model
model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_layers=num_layers,
    dropout=dropout,
    device=device,
    max_length=max_length
).to(device)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Data loaders with appropriate batch size for WMT14
BATCH_SIZE = 32  # Smaller batch size due to larger model and dataset

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# Learning rate scheduler
class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

scheduler = TransformerScheduler(optimizer, embed_dim)

def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(iterator, desc="Training"):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        
        optimizer.zero_grad()
        
        output = model(src, trg[:, :-1])
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            
            output = model(src, trg[:, :-1])
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def calculate_bleu(model, iterator, trg_field, max_len=50):
    model.eval()
    trg_vocab = trg_field.vocab
    trg_list = []
    output_list = []
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Calculating BLEU"):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            
            for i in range(trg.shape[0]):
                trg_sentence = []
                for j in range(1, trg.shape[1]):
                    token_idx = trg[i, j].item()
                    if token_idx == trg_vocab.stoi['<eos>']:
                        break
                    trg_sentence.append(trg_vocab.itos[token_idx])
                trg_list.append([trg_sentence])
            
            outputs = translate_batch(model, src, trg_vocab, max_len)
            output_list.extend(outputs)
    
    bleu_score = corpus_bleu(trg_list, output_list)
    return bleu_score

def translate_batch(model, src, trg_vocab, max_len=50):
    model.eval()
    batch_size = src.shape[0]
    trg_indexes = [[trg_vocab.stoi['<sos>']] for _ in range(batch_size)]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(src.device)
        
        with torch.no_grad():
            output = model(src, trg_tensor)
        
        next_tokens = output.argmax(2)[:, -1:]
        
        for j in range(batch_size):
            trg_indexes[j].append(next_tokens[j].item())
            
        if all(trg_vocab.stoi['<eos>'] in seq for seq in trg_indexes):
            break
    
    translations = []
    for seq in trg_indexes:
        translation = []
        for idx in seq[1:]:
            if idx == trg_vocab.stoi['<eos>']:
                break
            translation.append(trg_vocab.itos[idx])
        translations.append(translation)
    
    return translations

print("Checking for problematic sequences...")
max_length_to_check = 500  # Set a reasonable maximum

for i, batch in enumerate(train_iterator):
    if i >= 5:  # Check first 5 batches
        break
    
    src = batch.src.transpose(0, 1)
    trg = batch.trg.transpose(0, 1)
    
    print(f"Batch {i}: src_shape={src.shape}, trg_shape={trg.shape}")
    
    # Check for any indices that are out of vocabulary range
    src_max_idx = src.max().item()
    trg_max_idx = trg.max().item()
    
    print(f"  Source max index: {src_max_idx} (vocab size: {src_vocab_size})")
    print(f"  Target max index: {trg_max_idx} (vocab size: {trg_vocab_size})")
    
    if src_max_idx >= src_vocab_size:
        print(f"  ERROR: Source has index {src_max_idx} >= vocab size {src_vocab_size}")
    
    if trg_max_idx >= trg_vocab_size:
        print(f"  ERROR: Target has index {trg_max_idx} >= vocab size {trg_vocab_size}")
    
    # Check sequence lengths
    if src.shape[1] > max_length_to_check:
        print(f"  WARNING: Source sequence too long: {src.shape[1]}")
    
    if trg.shape[1] > max_length_to_check:
        print(f"  WARNING: Target sequence too long: {trg.shape[1]}")

# Training loop
N_EPOCHS = 10  # Start with fewer epochs for testing
CLIP = 1.0

best_valid_loss = float('inf')

print("Starting training on WMT14...")
for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate_epoch(model, valid_iterator, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    if (epoch + 1) % 2 == 0:
        bleu_score = calculate_bleu(model, valid_iterator, TRG)
        print(f'BLEU Score: {bleu_score:.4f}')
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-transformer-wmt14.pt')
        print(f"Saved new best model with validation loss: {valid_loss:.3f}")
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')

print("Training completed!")