import torch

# Implementation of a single head scaled dot product attention from the paper
class attention_head(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.q_linear = torch.nn.Linear(d_model, d_k)
        self.v_linear = torch.nn.Linear(d_model, d_v)
        self.k_linear = torch.nn.Linear(d_model, d_k)
        self.out = torch.nn.Linear(d_v, d_model)
    
    def forward(self, q, k, v, mask=None):
        # Apply the linear layers to the input
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        # Compute the scaled dot product attention
        weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        # Apply the mask
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        # Apply the softmax function
        normalized_weights = torch.nn.functional.softmax(weights, dim=-1)
        # Apply the attention to the value
        output = torch.matmul(normalized_weights, v)
        # Apply the linear layer to the output
        output = self.out(output)
        return output
    
def main():
    # Test the attention head
    attention = attention_head(512, 64, 64)
    q = k = v = torch.rand(8, 10, 512)
    output = attention(q, k, v)
    print(output.shape)

if __name__ == '__main__':  
    main()