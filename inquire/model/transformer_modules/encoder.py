from model.transformer_modules.mha import multi_head_attention
from model.transformer_modules.pffn import pffn

import torch
from torchsummary import summary

class encoder_layer(torch.nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_1 = torch.nn.LayerNorm(d_model)
        self.norm_2 = torch.nn.LayerNorm(d_model)
        self.attention = multi_head_attention(d_model, heads)
        self.ffn = pffn(d_model, d_ff)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Apply the multi-head attention layer
        attention = self.attention(x, x, x, mask)
        # Apply the dropout layer
        attention = self.dropout_1(attention)
        # Apply the residual connection and the layer normalization
        attention = self.norm_1(attention + x)
        # Apply the position wise feed forward network
        output = self.ffn(attention)
        # Apply the dropout layer
        output = self.dropout_2(output)
        # Apply the residual connection and the layer normalization
        output = self.norm_2(output + attention)
        return output

# Implementation of the encoder module of the transformer model
class Encoder(torch.nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, num_layers):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [encoder_layer(d_model, heads, d_ff, dropout) for _ in range(num_layers)]
        )
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
def main():
    # Test the encoder
    encoder = Encoder(512, 8, 2048, 0.1, 1)
    x = torch.rand(8, 10, 512)
    # mask = torch.ones(10, 1, 1)
    output = encoder(x)
    print(output.shape)

    # Torch Summary
    summary(encoder, (10, 512), device='cpu')

if __name__ == '__main__': 
    main()