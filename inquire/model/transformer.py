from model.transformer_modules.decoder import Decoder
from model.transformer_modules.encoder import Encoder

import torch

# Transformer layer with cross attention
class transformer_layer(torch.nn.Module):
    def __init__(self, d_input, d_model, heads, d_ff, dropout):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.encoder = Encoder(d_model, heads, d_ff, dropout, 1)
        self.decoder = Decoder(d_model, heads, d_ff, dropout, 1)

    def forward(self, src, trg, src_mask = None, trg_mask = None):
        # Apply the encoder
        e_outputs = self.encoder(src, src_mask)
        # Apply the decoder
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        return e_outputs, d_output

    

# Implementation of the transformer model with fix encoder input of 10 sequences and decoder input of 1 sequence
class Transformer(torch.nn.Module):
    def __init__(self, d_input, d_model, d_output, heads, d_ff, dropout, layers):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.layers = layers
        self.transformer_layers = torch.nn.ModuleList(
            [transformer_layer(d_input, d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.linear = torch.nn.Linear(d_model, d_output)

    def forward(self, src, trg, src_mask = None, trg_mask = None):  
        # Apply the transformer layers
        for layer in self.transformer_layers:
            src, trg = layer(src, trg, src_mask, trg_mask)
        # Apply the linear layer
        output = self.linear(trg)
        return output


def main():
    # Test the transformer
    transformer = Transformer(512, 512, 1, 8, 2048, 0.1, 1)
    src = torch.rand(8, 10, 512)
    trg = torch.rand(8, 1, 512)
    output = transformer(src, trg)
    print(output.shape)

    # Print the number of parameters
    print(sum(p.numel() for p in transformer.parameters() if p.requires_grad))

if __name__ == '__main__':
    main()