# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ssl
import urllib.request
import os
import tarfile
from io import BytesIO

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Manual dataset download function
def download_and_extract_multi30k():
    """Download and extract Multi30k dataset using the working URLs"""
    urls = {
        'train': "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
        'valid': "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
        'test': "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt_task1_test2016.tar.gz"
    }
    
    datasets = {}
    
    for split, url in urls.items():
        print(f"Downloading {split} data from {url}...")
        
        # Download the tar.gz file
        response = urllib.request.urlopen(url)
        tar_data = response.read()
        
        # Extract from memory
        with tarfile.open(fileobj=BytesIO(tar_data), mode='r:gz') as tar:
            # Get the list of files in the archive
            file_list = tar.getnames()
            print(f"Files in {split} archive: {file_list}")
            
            # Extract files to memory
            extracted_files = {}
            for member in tar.getmembers():
                if member.isfile():
                    file_content = tar.extractfile(member).read()
                    # Try different encodings
                    encodings = ['utf-8', 'latin-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            decoded_content = file_content.decode(encoding)
                            extracted_files[member.name] = decoded_content
                            print(f"Successfully decoded {member.name} with {encoding}")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If all encodings fail, use replace strategy
                        extracted_files[member.name] = file_content.decode('utf-8', errors='replace')
                        print(f"Used replace strategy for {member.name}")
        
        # Process the extracted files based on split
        if split == 'train':
            de_content = extracted_files.get('train.de', extracted_files.get('training/train.de', ''))
            en_content = extracted_files.get('train.en', extracted_files.get('training/train.en', ''))
        elif split == 'valid':
            de_content = extracted_files.get('val.de', extracted_files.get('validation/val.de', ''))
            en_content = extracted_files.get('val.en', extracted_files.get('validation/val.en', ''))
        else:  # test
            de_content = extracted_files.get('test2016.de', extracted_files.get('mmt16_task1/test2016.de', ''))
            en_content = extracted_files.get('test2016.en', extracted_files.get('mmt16_task1/test2016.en', ''))
        
        # Split into sentences
        de_sentences = de_content.strip().split('\n') if de_content else []
        en_sentences = en_content.strip().split('\n') if en_content else []
        
        # Ensure same length
        min_len = min(len(de_sentences), len(en_sentences))
        de_sentences = de_sentences[:min_len]
        en_sentences = en_sentences[:min_len]
        
        datasets[split] = list(zip(de_sentences, en_sentences))
        print(f"Loaded {min_len} {split} samples")
    
    return datasets['train'], datasets['valid'], datasets['test']

# Download dataset first
print("Downloading Multi30k dataset...")
train_data_raw, valid_data_raw, test_data_raw = download_and_extract_multi30k()

# Now continue with torchtext 0.6 imports and setup
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
from tqdm import tqdm
import math
import time
from nltk.translate.bleu_score import corpus_bleu

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

# Load Multi30k dataset using torchtext
print("Loading Multi30k dataset with torchtext...")
try:
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    print("Successfully loaded dataset with torchtext")
except Exception as e:
    print(f"Failed to load with torchtext: {e}")
    print("Creating dataset from manually downloaded data...")
    
    # Create temporary files for torchtext to read
    os.makedirs('.data/multi30k', exist_ok=True)
    
    # Write train data
    with open('.data/multi30k/train.de', 'w', encoding='utf-8') as f:
        f.write('\n'.join([de for de, en in train_data_raw]))
    with open('.data/multi30k/train.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join([en for de, en in train_data_raw]))
    
    # Write validation data  
    with open('.data/multi30k/val.de', 'w', encoding='utf-8') as f:
        f.write('\n'.join([de for de, en in valid_data_raw]))
    with open('.data/multi30k/val.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join([en for de, en in valid_data_raw]))
    
    # Write test data
    with open('.data/multi30k/test2016.de', 'w', encoding='utf-8') as f:
        f.write('\n'.join([de for de, en in test_data_raw]))
    with open('.data/multi30k/test2016.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join([en for de, en in test_data_raw]))
    
    # Now try loading again
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

print(f"Source vocab size: {src_vocab_size}")
print(f"Target vocab size: {trg_vocab_size}")

# Create model
model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    d_model=embed_dim,
    num_heads=num_heads,
    d_ff=ff_dim,
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
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

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