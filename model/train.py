# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
from tqdm import tqdm
import math
import time
from nltk.translate.bleu_score import corpus_bleu

from transformer import Transformer

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

# Load Multi30k dataset
print("Loading Multi30k dataset...")
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

# Build vocabulary
print("Building vocabulary...")
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f"German vocabulary size: {len(SRC.vocab)}")
print(f"English vocabulary size: {len(TRG.vocab)}")

# Model hyperparameters (smaller than original paper due to resource constraints)
src_vocab_size = len(SRC.vocab)
trg_vocab_size = len(TRG.vocab)
src_pad_idx = SRC.vocab.stoi['<pad>']
trg_pad_idx = TRG.vocab.stoi['<pad>']
embed_dim = 256
num_heads = 8
ff_dim = 512
num_layers = 3
dropout = 0.1
max_length = 100
device = device

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

# Data loaders
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# Learning rate scheduler (from Attention is All You Need)
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
        src = batch.src.transpose(0, 1)  # [batch_size, src_len]
        trg = batch.trg.transpose(0, 1)  # [batch_size, trg_len]
        
        optimizer.zero_grad()
        
        # Remove last token from target for input, and first token for output
        output = model(src, trg[:, :-1])
        
        # Reshape for loss calculation
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
            
            # Store target sentences
            for i in range(trg.shape[0]):
                trg_sentence = []
                for j in range(1, trg.shape[1]):  # Skip <sos>
                    token_idx = trg[i, j].item()
                    if token_idx == trg_vocab.stoi['<eos>']:
                        break
                    trg_sentence.append(trg_vocab.itos[token_idx])
                trg_list.append([trg_sentence])  # Wrap in list for corpus_bleu
            
            # Generate translations
            outputs = translate_batch(model, src, trg_vocab, max_len)
            output_list.extend(outputs)
    
    # Calculate BLEU score
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
        
        # Get next token
        next_tokens = output.argmax(2)[:, -1:]
        
        for j in range(batch_size):
            trg_indexes[j].append(next_tokens[j].item())
            
        # Stop if all sequences have <eos>
        if all(trg_vocab.stoi['<eos>'] in seq for seq in trg_indexes):
            break
    
    # Convert to tokens
    translations = []
    for seq in trg_indexes:
        translation = []
        for idx in seq[1:]:  # Skip <sos>
            if idx == trg_vocab.stoi['<eos>']:
                break
            translation.append(trg_vocab.itos[idx])
        translations.append(translation)
    
    return translations

# Training loop
N_EPOCHS = 20
CLIP = 1.0

best_valid_loss = float('inf')

print("Starting training...")
for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate_epoch(model, valid_iterator, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    # Calculate BLEU every 5 epochs
    if (epoch + 1) % 5 == 0:
        bleu_score = calculate_bleu(model, valid_iterator, TRG)
        print(f'BLEU Score: {bleu_score:.4f}')
    
    # Save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-transformer-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')

# Final evaluation on test set
print("Evaluating on test set...")
model.load_state_dict(torch.load('best-transformer-model.pt'))
test_loss = evaluate_epoch(model, test_iterator, criterion)
test_bleu = calculate_bleu(model, test_iterator, TRG)

print(f'Final Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
print(f'Final Test BLEU Score: {test_bleu:.4f}')