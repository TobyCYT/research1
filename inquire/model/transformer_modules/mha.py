from model.transformer_modules.attention import attention_head

import torch

# Implementation of the multi-head attention
class multi_head_attention(torch.nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.d_v = d_model // heads
        self.heads = heads
        self.attention_heads = torch.nn.ModuleList(
            [attention_head(d_model, self.d_k, self.d_v) for _ in range(heads)]
        )
        self.linear = torch.nn.Linear(d_model*heads, d_model)
    
    def forward(self, q, k, v, mask=None):
        # Concatenate the output of the attention heads
        concat = torch.cat([h(q, k, v, mask) for h in self.attention_heads], dim=-1)
        # Apply the linear layer to the concatenated output
        output = self.linear(concat)
        return output
    
def main():
    # Test the multi-head attention
    attention = multi_head_attention(512, 8)
    q = k = v = torch.rand(8, 10, 512)
    output = attention(q, k, v)
    print(output.shape)

if __name__ == '__main__':
    main()