import torch

class NeuralNet(torch.nn.Module):
    def __init__(self,dropout=0.1):
        super(NeuralNet, self).__init__()
        # self.fc1_1 = torch.nn.Linear(11, 1)
        # self.do1 = torch.nn.Dropout(dropout)
        # self.fc1_2 = torch.nn.Linear(512, 1)

        self.fc2_1 = torch.nn.Linear(2, 1)
        self.fc2_2 = torch.nn.Linear(512, 1)
        # self.fc2_3 = torch.nn.Linear(10, 1)

        # self.lin = torch.nn.Linear(2, 1)

    def forward(self, k, q):
        # k shape: (batch_size, 10, 512)
        # q shape: (batch_size, 1, 512)
        # x shape: (batch_size, 11, 512)
        # x.T shape: (batch_size, 512, 11)

        bs, _, _ = k.shape
        # x1 = torch.cat((k, q), 1)

        # x1 = x1.permute(0, 2, 1)
        # x1 = x1.reshape(bs*512, 11)
        # x1 = self.fc1_1(x1)
        # x1 = torch.relu(x1)
        # x1 = x1.reshape(bs, 512)
        # x1 = self.do1(x1)
        # x1 = self.fc1_2(x1)

        q = q.repeat(1, 10, 1)
        # interlace q and k
        x2 = torch.cat((q, k), 2)
        # x2 shape: (batch_size, 10, 2, 512)
        x2 = x2.reshape(bs*10, 2, 512)
        x2 = x2.permute(0, 2, 1)
        x2 = self.fc2_1(x2)
        x2 = torch.relu(x2)
        x2 = x2.reshape(bs, 10, 512)
        x2 = self.fc2_2(x2)
        x2 = torch.relu(x2)
        x2 = x2.reshape(bs, 10)
        # x2 = self.fc2_3(x2)

        # x = torch.cat((x1, x2), 1)
        # x = torch.relu(x)
        # x = self.lin(x)

        x = torch.mean(x2, 1).unsqueeze(1)

        return x
    
def main():
    # Test the model
    model = NeuralNet()
    q = torch.randn(64, 1, 512)
    k = torch.randn(64, 10, 512)
    out = model(k, q)
    out = torch.sigmoid(out)
    print(out.shape)

    # Number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))    

if __name__ == '__main__':
    main()