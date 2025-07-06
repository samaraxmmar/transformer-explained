# Introduction to Transformers

Welcome to the first notebook in the "Transformers Explained" series! In this notebook, we will explore the basics of the Transformer architecture, a model that has revolutionized natural language processing (NLP) and other fields.

## What is a Transformer?

The Transformer is a neural network architecture introduced in 2017 by Vaswani et al. in the paper "Attention Is All You Need." Unlike previous models such as RNNs (recurrent neural networks) and CNNs (convolutional neural networks), the Transformer relies entirely on attention mechanisms, thereby eliminating the need for recurrence or convolutions.

## Why are Transformers important?

Before Transformers, sequential models like RNNs (LSTM, GRU) dominated NLP tasks. However, they suffered from limitations, notably difficulty handling long-range dependencies and limited parallelization. Transformers solved these problems by introducing the self-attention mechanism, allowing the model to process all parts of a sequence simultaneously.

## General Architecture

A typical Transformer consists of two main parts:

1. **Encoder**: Processes the input sequence and produces a contextual representation.  
2. **Decoder**: Uses the encoder’s representation to generate the output sequence.

Each encoder and decoder is composed of several identical stacked layers.

### Encoder Blocks

Each encoder block contains two sublayers:

- **Multi-head self-attention mechanism**: Allows the model to weigh different parts of the input sequence.  
- **Position-wise feed-forward neural network**: Applies a linear transformation to each position.

### Decoder Blocks

Each decoder block contains three sublayers:

- **Masked multi-head self-attention mechanism**: Similar to the encoder but masks future positions to prevent cheating.  
- **Multi-head encoder-decoder attention mechanism**: Allows the decoder to focus on relevant parts of the encoder output.  
- **Position-wise feed-forward neural network**.

## Positional Encoding

Since Transformers contain no recurrence or convolution, they intrinsically have no notion of word order in the sequence. To address this, "positional encodings" are added to the input embeddings. These encodings provide information about the relative or absolute position of tokens in the sequence.

## Simplified Example (Pseudo-code)

```python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# A very simplified example of an encoder layer
class SimpleEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super(SimpleEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
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
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Model dimensions
d_model = 512  # Embedding dimension
num_heads = 8  # Number of attention heads
dim_feedforward = 2048  # Feed-forward network dimension
dropout = 0.1

# Create a simple encoder layer
encoder_layer = SimpleEncoderLayer(d_model, num_heads, dim_feedforward, dropout)

# Create a positional encoding
pos_encoder = PositionalEncoding(d_model)

# Example input (sequence_length, batch_size, d_model)
# Suppose a sequence of 10 words, batch size 2
input_sequence = torch.rand(10, 2, d_model)

# Add positional encoding
input_with_pos = pos_encoder(input_sequence)

# Pass through encoder layer
output = encoder_layer(input_with_pos)

print(f"Input shape: {input_sequence.shape}")
print(f"Encoder output shape: {output.shape}")
