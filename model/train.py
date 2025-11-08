import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import math
from model.transformer import Transformer
from data.data_loader import load_multi30k
from model.bleu_score import calculate_bleu

def train_model(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        # Remove last token from target for input, keep for output
        output = model(src, trg[:, :-1])
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            output = model(src, trg[:, :-1])
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    # Hyperparameters (matching "Attention Is All You Need" paper as closely as possible)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128
    N_EPOCHS = 20
    CLIP = 1.0
    LEARNING_RATE = 0.0005
    
    # Model parameters
    EMBED_DIM = 512
    N_HEADS = 8
    FF_DIM = 2048
    N_LAYERS = 6
    DROPOUT = 0.1
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_iterator, valid_iterator, test_iterator, SRC, TRG = load_multi30k(
        batch_size=BATCH_SIZE, device=DEVICE
    )
    
    # Initialize model
    model = Transformer(
        src_vocab_size=len(SRC.vocab),
        trg_vocab_size=len(TRG.vocab),
        src_pad_idx=SRC.vocab.stoi[SRC.pad_token],
        trg_pad_idx=TRG.vocab.stoi[TRG.pad_token],
        embed_dim=EMBED_DIM,
        num_heads=N_HEADS,
        ff_dim=FF_DIM,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        device=DEVICE,
        max_length=100
    ).to(DEVICE)
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])
    
    # Training loop
    best_valid_loss = float('inf')
    writer = SummaryWriter()
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train_model(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate_model(model, valid_iterator, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Calculate BLEU score on validation set every 5 epochs
        if (epoch + 1) % 5 == 0:
            bleu_score = calculate_bleu(model, valid_iterator, SRC, TRG, DEVICE)
            print(f'BLEU Score: {bleu_score:.2f}')
            writer.add_scalar('BLEU', bleu_score, epoch)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-transformer-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('PPL/train', math.exp(train_loss), epoch)
        writer.add_scalar('PPL/valid', math.exp(valid_loss), epoch)
    
    writer.close()
    
    # Final evaluation on test set
    model.load_state_dict(torch.load('best-transformer-model.pt'))
    test_loss = evaluate_model(model, test_iterator, criterion)
    test_bleu = calculate_bleu(model, test_iterator, SRC, TRG, DEVICE)
    
    print(f'Final Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
    print(f'Final Test BLEU Score: {test_bleu:.2f}')

if __name__ == '__main__':
    main()