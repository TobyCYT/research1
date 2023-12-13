import torch
import torchsummary

# Define a simple FFN model
class FFN(torch.nn.Module):
    def __init__(self, input_dim=512*11, hidden_dim=512, output_dim=1):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
def main():
    # test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FFN().to(device)
    torchsummary.summary(model, (512*11,))

if __name__ == '__main__':
    main()