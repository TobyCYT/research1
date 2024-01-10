try:
    from model.transformer_modules.mha import multi_head_attention
    from model.transformer_modules.pffn import pffn
except:
    try:
        from transformer_modules.mha import multi_head_attention
        from transformer_modules.pffn import pffn
    except:
        from mha import multi_head_attention
        from pffn import pffn

import torch
from torchsummary import summary

class decoder_layer(torch.nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_1 = torch.nn.LayerNorm(d_model)
        self.norm_2 = torch.nn.LayerNorm(d_model)
        self.norm_3 = torch.nn.LayerNorm(d_model)
        self.attention_1 = multi_head_attention(d_model, heads)
        self.attention_2 = multi_head_attention(d_model, heads)
        self.ffn = pffn(d_model, d_ff)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.dropout_3 = torch.nn.Dropout(dropout)
        
    def forward(self, x, e_outputs = None, src_mask = None, trg_mask = None):
        # Apply the multi-head attention layer
        attention = self.attention_1(x, x, x, trg_mask)
        # Apply the dropout layer
        attention = self.dropout_1(attention)
        # Apply the residual connection and the layer normalization
        attention = self.norm_1(attention + x)
        if e_outputs is None:
            e_outputs = x
        # Apply the multi-head attention layer
        attention_2 = self.attention_2(attention, e_outputs, e_outputs, src_mask)
        # Apply the dropout layer
        attention_2 = self.dropout_2(attention_2)
        # Apply the residual connection and the layer normalization
        attention_2 = self.norm_2(attention_2 + attention)
        # Apply the position wise feed forward network
        output = self.ffn(attention_2)
        # Apply the dropout layer
        output = self.dropout_3(output)
        # Apply the residual connection and the layer normalization
        output = self.norm_3(output + attention_2)
        return output

# Implementation of the decoder module of the transformer model
class Decoder(torch.nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, num_layers):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [decoder_layer(d_model, heads, d_ff, dropout) for _ in range(num_layers)]
        )
        
    def forward(self, x, e_outputs = None, src_mask = None, trg_mask = None):
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        return x
    
def main():
    # Test the decoder
    decoder = Decoder(512, 8, 2048, 0.1, 1)
    x = torch.rand(8, 1, 512)
    # Cross attention from the encoder outputs
    e_outputs = torch.rand(8, 10, 512)
    output = decoder(x, e_outputs)
    print(output.shape)

    # Torch Summary
    summary(decoder, (1, 512), device='cpu')

if __name__ == '__main__': 
    main()