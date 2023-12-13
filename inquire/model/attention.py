import torch

class attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_linear = torch.nn.Linear(512, 512) # Query
        self.v_linear = torch.nn.Linear(512, 512) # Query
        self.k_linear = torch.nn.Linear(512, 512) # Search Result

        self.pffn = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512)
        )

        self.out = torch.nn.Linear(512, 1)

        
    def forward(self, k, q):
        bs = q.shape[0]
        q = q.view(-1, 512)
        k = k.view(-1, 512)
        v = k
        # q = self.q_linear(q)
        # k = self.k_linear(k)
        # v = self.v_linear(v)

        weights = torch.matmul(q, k.T)
        weights = torch.nn.functional.softmax(weights, dim = -1)
        out = torch.matmul(weights, v)
        out = self.pffn(out)
        out = self.out(out)

        return out
    
def main():
    # Test the model
    model = attention()
    q = torch.randn(64, 1, 512)
    k = torch.randn(64, 10, 512)
    out = model(q, k)
    print(out.shape)

    # Number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

if __name__ == "__main__":
    main()