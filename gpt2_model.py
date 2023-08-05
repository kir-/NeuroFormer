# gpt2_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from transformers import GPT2Tokenizer

class OscillatoryAttention(nn.Module):
    def __init__(self, d_model=768, nhead=12, dropout=0.1):
        super(OscillatoryAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        q = query
        k = key
        v = value

        # Calculate sinusoidal weights
        pos_enc = torch.arange(0, self.d_model, 2, dtype=torch.float32, device=query.device)
        sin_weights = torch.sin(pos_enc / self.d_model)
        cos_weights = torch.cos(pos_enc / self.d_model)

        # Apply sinusoidal modification to the attention weights
        attention_weights = torch.einsum('bth, bsh -> bts', q + sin_weights, k + cos_weights)

        # Apply mask if available
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(1), float('-inf'))

        # Apply softmax to compute attention scores
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum using the modified attention scores
        output = torch.einsum('bts, bsh -> bth', attention_weights, v)

        return output

class GPT2EncoderLayer(nn.Module):
    def __init__(self, d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1, ltc=False, oscattention=False):
        super(GPT2EncoderLayer, self).__init__()
        if (oscattention== False):
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = OscillatoryAttention(d_model, nhead, dropout)

        if (ltc == False):
            self.layer1 = nn.Linear(d_model, dim_feedforward)
        else:
            wiring = AutoNCP(16, dim_feedforward)  # 16 units, 1 motor neuron
            self.layer1 = LTC(d_model, wiring, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
       

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.layer1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
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

import torch.nn.functional as F

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

