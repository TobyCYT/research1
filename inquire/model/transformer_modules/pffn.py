import torch

# Implementation of Position Wise Feed Forward Network
class pffn(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_ff)
        self.linear_2 = torch.nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear_2(x)
        return x
    
def main():
    # Test the position wise feed forward network
    model = pffn(512, 2048)
    x = torch.rand(10, 512)
    output = model(x)
    print(output.shape)

if __name__ == '__main__':
    main()