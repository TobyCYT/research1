from utils.dataloader import get_loader
from model.ffn import FFN

import torch
from tqdm import tqdm

def val(model, val_loader, device='cpu'):
    model.eval()
    running_corrects = 0

    # Iterate over data
    for i, (sample, label, sample_label) in enumerate(tqdm(val_loader, total=len(val_loader), desc='Validation')):
        sample = sample.to(device)
        label = label.to(device)
        x = torch.cat((label, sample), 1)
        x = x.view(-1, 512*11)
        sample_label = sample_label.to(device).unsqueeze(1)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(x)
            outputs = torch.sigmoid(outputs)
            running_corrects += torch.sum((outputs > 0.5).float() == sample_label.data)

    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    print('Validation Acc: {:.4f}'.format(epoch_acc))
    return epoch_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader = get_loader('val', 1)
    model = FFN().to(device)
    model.load_state_dict(torch.load('inquire/ckpt/2023-12-01_18-38-46/accFFN.pt'))
    # Run validation 10 times and take the average
    acc = []
    for _ in range(10):
        acc.append(val(model, val_loader, device))
    print('Average Validation Acc: {:.4f}'.format(sum(acc)/len(acc)))

if __name__ == '__main__':
    main()