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
    linear = torch.nn.Linear(11, 1)
    # out_linear = torch.nn.Linear(10, 1)
    x = torch.tensor([x+1 for x in range(4)] ,dtype=torch.float32).unsqueeze(0)
    y = x
    x = x.repeat(1, 11, 1)
    print(x.shape)
    x = x.permute(0, 2, 1)
    print(x.shape)
    bs = x.shape[0]
    x = x.reshape(bs*4, 11)
    print(x)
    out = linear(x)
    print(out.shape)

    # Test the model

    # model = NeuralNet()
    # q = torch.randn(64, 1, 512)
    # k = torch.randn(64, 10, 512)
    # out = model(q, k)
    # print(out.shape)

if __name__ == '__main__':
    main()