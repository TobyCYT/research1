# from model.transformer import Transformer
# from model.attention import attention
from model.neuralnet import NeuralNet

from utils.dataloader import get_loader

import os
import datetime

import torch
from tqdm import tqdm

# Define the training function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cpu'):
    best_val_acc = 0.0
    best_val_loss = 100.0
    best_train_acc = 0.0
    best_train_loss = 100.0
    # Create a folder in the checkpoint folder to save the best model with the current date time
    ckpt_path = 'inquire/ckpt/%s'%datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(ckpt_path, exist_ok=True)
    # Save the current settings into settings.txt, optimizer, scheduler, criterion, learning rate
    with open(ckpt_path+'/settings.txt', 'w') as f:
        f.write('Optimizer: %s\n'%optimizer)
        f.write('Scheduler: %s\n'%scheduler)
        f.write('Criterion: %s\n'%criterion)
        f.write('Learning rate: %s\n'%optimizer.param_groups[0]['lr'])
        
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for i, (sample, label, sample_label) in enumerate(tqdm(dataloader, total=len(dataloader), desc='Epoch %s'%epoch)):
                sample = sample.to(device)
                label = label.to(device)
                sample_label = sample_label.to(device).unsqueeze(1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(sample, label)
                    # outputs = outputs.squeeze(1)
                    outputs = torch.sigmoid(outputs)
                    loss = criterion(outputs, sample_label.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * sample.size(0)
                running_corrects += torch.sum(torch.round(torch.sigmoid(outputs)) == sample_label.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train' and scheduler is not None:
                scheduler.step(epoch_loss)

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), ckpt_path+'/VALaccFFN.pt')
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(model.state_dict(), ckpt_path+'/VALlossFFN.pt')

            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
                torch.save(model.state_dict(), ckpt_path+'/TRAINaccFFN.pt')
            if phase == 'train' and epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                torch.save(model.state_dict(), ckpt_path+'/TRAINlossFFN.pt')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    # Save the matrics in result.txt
    with open(ckpt_path+'/result.txt', 'w') as f:
        f.write('Best val acc: %s\n'%best_val_acc)
        f.write('Best val loss: %s\n'%best_val_loss)
        f.write('Best train acc: %s\n'%best_train_acc)
        f.write('Best train loss: %s\n'%best_train_loss)

def start_train(ckpt=None, num_epochs=100, warmup_epochs=0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = Transformer(512, 512, 1, 8, 2048, 0.1, 1).to(device)
    model = NeuralNet().to(device)
    train_loader = get_loader(mode='train', batch_size= 128)
    val_loader = get_loader(mode='val', batch_size= 128)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # scheduler = None
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))

    # warmup epoch for adam optimizer with very low learning rate
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 3e-4 ** (epoch / warmup_epochs))
        for _ in range(warmup_epochs):
            train(model, train_loader, val_loader, criterion, optimizer, warmup_scheduler, num_epochs=1, device=device)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)

def main():
    start_train(num_epochs=300)

if __name__ == '__main__':
    main()