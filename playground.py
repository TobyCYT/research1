import torch

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(512, 1)

    def forward(self, k, q):
        bs = q.shape[0]
        x = k * q
        x = self.fc1(x)
        out = torch.max(x, dim=1).values
        return out

def main():
    ckpt = torch.load("inquire/ckpt/2023-12-06_18-39-17/VALaccFFN.pt")
    # Print model structure
    for key in ckpt.keys():
        print(key)
        print(ckpt[key].shape)

if __name__ == '__main__':
    main()