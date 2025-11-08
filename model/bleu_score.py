import torch
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

def translate_sentence(model, sentence, src_field, trg_field, device, max_length=50):
    model.eval()
    
    if isinstance(sentence, str):
        tokens = [token.lower() for token in sentence.split()]
    else:
        tokens = [token.lower() for token in sentence]
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor, model.make_src_mask(src_tensor))
    
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.decoder(trg_tensor, encoder_outputs, 
                                 model.make_trg_mask(trg_tensor), 
                                 model.make_src_mask(src_tensor))
            output = model.fc_out(output)
        
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]  # Remove <sos>

def calculate_bleu(model, iterator, src_field, trg_field, device):
    targets = []
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="Calculating BLEU"):
            src = batch.src
            trg = batch.trg
            
            # Translate each sentence in batch
            for j in range(src.shape[0]):
                src_sentence = [src_field.vocab.itos[idx] for idx in src[j] 
                               if idx not in [src_field.vocab.stoi[src_field.pad_token]]]
                trg_sentence = [trg_field.vocab.itos[idx] for idx in trg[j] 
                               if idx not in [trg_field.vocab.stoi[trg_field.pad_token]]]
                
                # Remove <sos> and <eos> for target
                trg_sentence = trg_sentence[1:-1]
                
                prediction = translate_sentence(model, src_sentence, src_field, trg_field, device)
                prediction = prediction[:-1]  # Remove <eos>
                
                # Filter out empty sentences
                if len(prediction) > 0 and len(trg_sentence) > 0:
                    targets.append([trg_sentence])  # BLEU expects list of list of references
                    predictions.append(prediction)
    
    return bleu_score(predictions, targets) * 100