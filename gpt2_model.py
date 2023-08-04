# gpt2_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC
from ncps.wirings import AutoNCP

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
    def __init__(self, vocab_size, d_model=768, nhead=12, num_encoder_layers=12, dim_feedforward=3072, max_seq_length=1024, num_classes=10):
        super(GPT2Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = GPT2Encoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout=0.1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        START_TOKEN_INDEX = 0  # Replace with the actual index of your start token (e.g., <SOS>)
        max_seq_length = 512  # You can adjust this based on your preference and requirements
        temperature = 0.7  # You can experiment with different temperature values (e.g., 0.7, 1.0, 1.5, etc.)
        src_embedding = self.embedding(src)
        src_mask = None
        src_key_padding_mask = None
        encoder_output = self.encoder(src_embedding, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Initialize the output_ids with the start token
        start_token = torch.tensor([START_TOKEN_INDEX], dtype=torch.long, device=src.device)
        output_ids = start_token.unsqueeze(0).expand(src.size(0), -1)  # Shape: (batch_size, seq_len)

        # Generate tokens autoregressively
        for step in range(1, max_seq_length):
            input_tokens = output_ids[:, :step]
            input_embedding = self.embedding(input_tokens)
            decoder_output = self.encoder(input_embedding, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

            # Get the logits for the next token from the decoder_output
            next_token_logits = self.classifier(decoder_output)

            # Sample the next token (you can use temperature to control the randomness)
            next_token_probs = F.softmax(next_token_logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)

            # Append the generated token to the output_ids
            output_ids = torch.cat((output_ids, next_token.unsqueeze(1)), dim=-1)

        return output_ids

# Replace START_TOKEN_INDEX with the index of your start token (e.g., <SOS>).
