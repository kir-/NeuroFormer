# gpt2_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class GPT2EncoderLayer(nn.Module):
    def __init__(self, d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1):
        super(GPT2EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.layer1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
       

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.norm1(src)
        src_mask = generate_square_subsequent_mask(len(src)).to(src.device)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.layer1(src))))
        src = src + self.dropout2(src2)
        return src

class GPT2Encoder(nn.Module):
    def __init__(self, num_layers=12, d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1):
        super(GPT2Encoder, self).__init__()
        self.layers = nn.ModuleList([GPT2EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return src

class GPT2Model(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, num_layers=12, nhead=12, dim_feedforward=3072, dropout=0.1, max_seq_length=1024):
        super(GPT2Model, self).__init__()

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (we're going with learned embeddings here, similar to original GPT-2)
        self.positional_embedding = nn.Embedding(max_seq_length, d_model)

        # The main GPT-2 body (stacked layers of transformers)
        self.encoder = GPT2Encoder(num_layers, d_model, nhead, dim_feedforward, dropout)

        # To produce logits over the vocabulary
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # Input embedding
        input_emb = self.embedding(src)

        # Add positional embedding
        positions = torch.arange(len(src[0]), device=src.device).unsqueeze(0)
        position_emb = self.positional_embedding(positions)
        embeddings = input_emb + position_emb

        # Passing through the encoder
        transformer_output = self.encoder(embeddings)

        # Producing logits over the vocabulary
        logits = self.classifier(transformer_output)

        return logits
    
    def generate(self, input_ids, max_length=1024, temperature=1.0, pad_token_id=None):
        self.eval()
        generated = input_ids
        with torch.no_grad():
            for _ in range(max_length - len(input_ids[0])):
                logits = self(generated)[:, -1, :]
                probs = torch.exp(F.log_softmax(logits / temperature, dim=-1))
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)

                if pad_token_id is not None and next_token.item() == pad_token_id:
                    break
        return generated


