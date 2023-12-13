import torch

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(11, 1)
        self.fc2 = torch.nn.Linear(512, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fc4 = torch.nn.Linear(512, 1)

    def forward(self, k, q):
        # k shape: (batch_size, 10, 512)
        # q shape: (batch_size, 1, 512)
        # x shape: (batch_size, 11, 512)
        # x.T shape: (batch_size, 512, 11)

        bs, _, _ = k.shape
        x = torch.cat((k, q), 1)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs*512, 11)
        x = self.fc1(x)
        x = torch.relu(x)
        x = x.reshape(bs, 512)
        x = x + self.fc2(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = x + self.fc3(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc4(x)

        return x
    
def main():
    # Test the model
    model = NeuralNet()
    q = torch.randn(64, 1, 512)
    k = torch.randn(64, 10, 512)
    out = model(q, k)
    out = torch.sigmoid(out)
    print(out.shape)

    # Number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))    

if __name__ == '__main__':
    main()